
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from pytorch_wavelets import DWTForward
from torchsummary import summary

import torch_dct as DCT
from torch.autograd import Function
from torchdiffeq import odeint  


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0,stride=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride,padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.res1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.c256 = ConvBNReLU(512, 256, kernel_size=1)

        self.upconv3 = ConvBNReLU(256, 256, kernel_size=3, padding=1)


        self.c128 = ConvBNReLU(256, 128, kernel_size=1)

        self.upconv2 = ConvBNReLU(128, 128, kernel_size=3, padding=1)

        self.init_param()

    def forward(self, x1, x2, x3):

        x3_256 = self.c256(x3)                      

        x3up = F.interpolate(
            x3_256, 
            size=x2.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )                                             

        # x3up = self.upconv3(x3up)                     

        xm = x3up + x2                              

        x_m128 = self.c128(xm)                       
        x_mup = F.interpolate(
            x_m128, 
            size=x1.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        # x_mup = self.upconv2(x_mup)                  
        out = x_mup + x1                            

        y = self.res1(out)                            
        return y

    
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class UnifiedAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 reduction=4,
                 init_C1=0.01**2,
                 init_C2=0.03**2,
                 integration_steps=2, 
                 method='odeint',
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.heads = num_heads
        self.integration_steps = integration_steps
        self.method = method

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.temp     = nn.Parameter(torch.ones(num_heads))
        self.gate     = nn.Parameter(torch.tensor(0.5))
        self.C1       = nn.Parameter(torch.tensor(init_C1))
        self.C2       = nn.Parameter(torch.tensor(init_C2))

        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out    = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

        self.channel_att = nn.Sequential(
            nn.Linear(dim, dim // max(reduction, 2)),
            nn.ReLU(inplace=True),
            nn.Linear(dim // max(reduction, 2), dim),
            nn.Sigmoid()
        )
        self.ode_func = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, 1, kernel_size=3, padding=1)
        )
        self.proj_low = nn.Linear(dim, dim)

    def dynamics(self, t, A):
        return self.ode_func(A.unsqueeze(1)).squeeze(1)

    def forward(self, x, mode='low_fre'):
        """
        x: [B, N, C]
        mode: 'high_fre' 或 'low_fre'
        """
        B, N, C = x.shape

        qkv = self.qkv_proj(x).view(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, C]

        if mode == 'high_fre':

            w = rearrange(qkv[0], 'b n (h d) -> b h n d', h=self.heads)
            scale = torch.sqrt(torch.tensor(w.shape[-1], device=x.device)) + 1e-6

            energy    = torch.norm(w, p=2, dim=-1)**2
            Pi        = torch.softmax((energy/scale) * self.temp.view(1,-1,1), dim=-1)
            w_sq      = w**2
            Pi_norm   = Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)
            dots_sim  = (Pi_norm.unsqueeze(-2) @ w_sq)
            attn_sim  = torch.exp(-dots_sim)
            out_sim   = w * Pi.unsqueeze(-1) * attn_sim

            w_neg     = -w
            energy_n  = torch.norm(w_neg, p=2, dim=-1)**2
            Pi_n      = torch.softmax((energy_n/scale) * self.temp.view(1,-1,1), dim=-1)
            w_sq_n    = w_neg**2
            Pi_n_norm = Pi_n / (Pi_n.sum(dim=-1, keepdim=True) + 1e-8)
            dots_opp  = (Pi_n_norm.unsqueeze(-2) @ w_sq_n)
            attn_opp  = torch.exp(-dots_opp)
            out_opp   = w_neg * Pi_n.unsqueeze(-1) * attn_opp

            g   = torch.sigmoid(self.gate)
            out = g * out_sim + (1 - g) * out_opp               
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

        else:  # 'low_fre'

            mu_q = q.mean(dim=2, keepdim=True)
            mu_k = k.mean(dim=2, keepdim=True)
            qk   = (q - mu_q) * (k - mu_k)
            sigma_qk = qk.sum(dim=2, keepdim=True) / (C - 1)
            sigma_q2 = (q - mu_q).pow(2).sum(dim=2, keepdim=True) / (C - 1)
            sigma_k2 = (k - mu_k).pow(2).sum(dim=2, keepdim=True) / (C - 1)

            num = (2*mu_q*mu_k + self.C1) * (2*sigma_qk + self.C2)
            den = (mu_q.pow(2) + mu_k.pow(2) + self.C1) * (sigma_q2 + sigma_k2 + self.C2)
            attn_base = (num / (den + 1e-7)).pow(2).squeeze(-1)    # [B,N]

            if self.integration_steps > 0:
                t        = torch.linspace(0, 1, max(2,self.integration_steps), device=x.device)
                attn_evo = odeint(self.dynamics, attn_base, t, method=self.method)
                final_a  = torch.sigmoid(attn_evo[-1])
            else:
                final_a  = torch.sigmoid(attn_base)

            weights   = torch.softmax(final_a, dim=1)             # [B,N]
            ch_w      = self.channel_att(x.mean(dim=1, keepdim=True))  # [B,1,C]
            out_low   = weights.unsqueeze(-1) * v * ch_w          # [B,N,C]
            return self.proj_low(out_low)

        

class GlobalLocalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(GlobalLocalAttentionBlock, self).__init__()
        self.attention1 = UnifiedAttention(dim, num_heads)
        

    def forward(self, x, H, W ,mode='low_fre'):

        if mode == 'high_fre':    #########high_fre
            global_out = self.attention1(x, mode='high_fre')
        
        elif mode == 'low_fre':
            # global_out = self.attention1(x, H, W)
            global_out = self.attention1(x, mode='low_fre')

        
        return global_out



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttentionBlock(
            dim,
            num_heads=num_heads
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, mode='low_fre'):
        x = x + self.drop_path1(self.attn(self.norm1(x), H, W, mode=mode))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x



    
class Down_wt1(nn.Module):
    def __init__(self, in_ch, out_ch, initial_enhancement=1.5):
        super(Down_wt1, self).__init__()

        # 将增强因子设为可学习参数
        self.enhancement_factor = nn.Parameter(
            torch.tensor(initial_enhancement, dtype=torch.float32)
        )

        # 定义离散小波变换（DWT）模块
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv_bn_relu3 = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def enhance_high_freq(self, cH, cV, cD):

        factor = self.enhancement_factor.clamp(min=0.5, max=5.0) 
        return cH * factor, cV * factor, cD * factor

    def forward(self, x):
        yl, yH = self.wt(x) 
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]


        y_HL, y_LH, y_HH = self.enhance_high_freq(y_HL, y_LH, y_HH)


        yh = torch.cat([y_HL, y_LH, y_HH], dim=1)

        yl = self.conv_bn_relu1(yl)
        yh = self.conv_bn_relu3(yh)
        return yl, yh



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,deep = 0):     
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.deep = deep
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dwt = Down_wt1(in_chans,embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        B, C, H, W = x.shape     #torch.Size([1, 3, 256, 256])
        if self.deep == 0:
            # x = self.proj(x)   # torch.Size([1, 128, 64, 64])
            # x1 = x.flatten(2)    #torch.Size([1, 128, 4096])
            # x2 = x1.transpose(1, 2)    #torch.Size([1, 4096, 128])
            x = self.proj(x).flatten(2).transpose(1, 2)   
            y = x
        else:
            x, y = self.dwt(x)
            x = x.flatten(2).transpose(1, 2)
            y = y.flatten(2).transpose(1, 2)

        x = self.norm(x)
        y = self.norm(y)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x ,y ,(H, W)


# borrow from PVT https://github.com/whai362/PVT.git
class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = nn.ModuleList()       
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(len(depths)):       
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i],i))
            else:
                self.patch_embeds.append(
                    PatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i],i))    
            patch_num = self.patch_embeds[-1].num_patches + 1 if i == len(embed_dims) - 1 else self.patch_embeds[      
                -1].num_patches
            self.pos_embeds.append(nn.Parameter(torch.zeros(1, patch_num, embed_dims[i])))  
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([
                block_cls(
                    dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[k]
                    )
                for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        self.norm = norm_layer(embed_dims[-1])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))

        # classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        for pos_emb in self.pos_embeds:
            trunc_normal_(pos_emb, std=.02)
        self.apply(self._init_weights)   

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.blocks[k][i].drop_path.drop_prob = dpr[cur + i]
            cur += self.depths[k]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim, i,  s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.conv1 = nn.Conv2d(int(in_chans * (1 / 4)), int(embed_dim * (1 / 4)), 3, s, 1, bias=True)  #groups=int(in_chans * 1 / 4)
        self.conv2 = nn.Conv2d(int(in_chans - in_chans*(1/4)), int(embed_dim - embed_dim*(1/4)), 3, s, 1, bias=True)  #groups= int(in_chans - in_chans*(1/4))

        self.s = s
        self.i = i

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x    #将x给特征标记
        # feat_token_y = y
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)    #torch.Size([1, 128, 4096])    [torch.Size([1, 128, 64, 64])]
        if self.s == 1: 
            x = self.proj(cnn_feat) + cnn_feat      
            # y = self.proj(feat_token_y) + feat_token_y
        x = x.flatten(2).transpose(1, 2)     #torch.Size([1, 4096, 128])
        # y = y.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

    

class JSDAdaptiveFusion(nn.Module):
    """
    A fusion module that adaptively blends two feature maps using Jensen-Shannon divergence.

    Inputs:
        x1: torch.Tensor of shape (B, C, H, W) - first feature map (e.g., high-frequency components)
        x2: torch.Tensor of shape (B, C, H, W) - second feature map
    Output:
        torch.Tensor of shape (B, C, H, W) - fused feature map
    """
    def __init__(self, eps: float = 1e-6):
        super(JSDAdaptiveFusion, self).__init__()
        self.eps = eps
        # small refinement conv to smooth weight map and ensure values in [0,1]
        self.refine = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Compute channel-wise probability distributions via softmax
        P = F.softmax(x1, dim=1)
        Q = F.softmax(x2, dim=1)

        M = 0.5 * (P + Q)
        KL1 = torch.sum(P * (torch.log(P + self.eps) - torch.log(M + self.eps)), dim=1, keepdim=True)

        KL2 = torch.sum(Q * (torch.log(Q + self.eps) - torch.log(M + self.eps)), dim=1, keepdim=True)

        js_map = 0.5 * (KL1 + KL2) 

        weight = self.refine(js_map)

        out = weight * x1 + (1.0 - weight) * x2
        return out

class CPVTV2(PyramidVisionTransformer):
    """
    CPVT-based Pyramid Vision Transformer.
    Uses Position Encoding Generator (PEG) instead of fixed positional embeddings.
    Removes class token for improved performance on segmentation and detection tasks.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, 
                 embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], 
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], 
                 block_cls=Block):
        super(CPVTV2, self).__init__(img_size, patch_size, in_chans, num_classes, 
                                     embed_dims, num_heads, mlp_ratios, qkv_bias, 
                                     qk_scale, drop_rate, attn_drop_rate, drop_path_rate, 
                                     norm_layer, depths, sr_ratios, block_cls)

        del self.pos_embeds
        del self.cls_token

        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim, i) for i, embed_dim in enumerate(embed_dims)]
        )

        self.regression = Regression()

        self.fusion_modules = nn.ModuleList([
            JSDAdaptiveFusion() for _ in embed_dims
        ])

        self.apply(self._init_weights)

    def forward_features(self, x):
        outputs = []  

        B = x.shape[0]  

        for i in range(len(self.depths)):

            x, y, (H, W) = self.patch_embeds[i](x)  # x, y -> [B, N, C]  (N = H * W)

            x = self.pos_drops[i](x)
            y = self.pos_drops[i](y)

            mode_x = 'low_fre'
            mode_y = 'high_fre'

            # Transformer 块
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W, mode=mode_x)
                y = blk(y, H, W, mode=mode_y)
                if j == 0:
                    x = self.pos_block[i](x, H, W)
                    y = self.pos_block[i](y, H, W)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


            grad_x_ = self.fusion_modules[i](x, y)
            # grad_x_ = x + y

            outputs.append(grad_x_)

        return outputs

    def forward(self, x):
        x = self.forward_features(x)
        mu = self.regression(x[1], x[2], x[3])
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


class PCPVT(CPVTV2):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=Block):
        super(PCPVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                    mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                    norm_layer, depths, sr_ratios, block_cls)


class FSGformer(PCPVT):
    """
    alias Twins-SVT
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=Block):
        super(FSGformer, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                     mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                     norm_layer, depths, sr_ratios, block_cls)
        del self.blocks
 
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k],) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.apply(self._init_weights)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict
