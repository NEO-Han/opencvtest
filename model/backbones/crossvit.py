import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
    

class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
    
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape#[8,3,256,128]
        # print(B,C,H,W)
        # print(self.patch_size)
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # print(x.shape)#[8,768,16,8]
        x = x.flatten(2).transpose(1, 2) # [8,128,768]
        # print(x.shape)
        return x
    

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, 
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                       has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d+1) % num_branches]), act_layer(), nn.Linear(dim[(d+1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        #NOTE
        # print(len(x))
        # outs_b = [print(x_.shape) for x_, block in zip(x, self.blocks)]
        # outs_b = [print(block) for x_, block in zip(x, self.blocks)]
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]  #Given normalized_shape=[384], expected input with shape [*, 384], but got input of size[1000]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size,patches)]

class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu =1.0):
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = local_feature
        self.num_branches = len(patch_size)
        self.patch_embed=nn.ModuleList()
        if hybrid_backbone is not None:
            if hybrid_backbone =='t2t':
                # print(img_size)
                # img_size_t=to_2tuple(img_size)
                # print(img_size_t)
                # patch_size_t=to_2tuple(patch_size)
                # print(patch_size)
                # num_patches = _compute_num_patches(to_2tuple(img_size), to_2tuple(patch_size))
                num_patches = _compute_num_patches(img_size, patch_size)
                self.pos_embed = nn.ParameterList()
                from t2t import T2T,get_sinusoid_encoding
                tokens_type = 'transformer' 
                for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                    self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                    self.pos_embed.append(nn.Parameter(data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]), requires_grad=False))
                # num_patches = self.patch_embed.num_patches
            else:
                # self.patch_embed = PatchEmbed_overlap(
                # img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                # embed_dim=embed_dim)
                # num_patches = self.patch_embed.num_patches
                num_patches = _compute_num_patches(img_size, patch_size)
                # print(num_patches)#[448,64]
                self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
                for im_s, p,s, d in zip(img_size, patch_size, stride_size,embed_dim):
                    # print(im_s)
                    self.patch_embed.append(PatchEmbed_overlap(img_size=im_s, patch_size=p, stride_size=s,in_chans=in_chans, embed_dim=d))
        else:
                self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.ParameterList([nn.Parameter(torch.zeros(camera * view, 1, embed_dim[i])) for i in range(self.num_branches)])
            for i in range(self.num_branches):
                trunc_normal_(self.sie_embed[i], std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.ParameterList([nn.Parameter(torch.zeros(camera, 1, embed_dim[i])) for i in range(self.num_branches)])
            for i in range(self.num_branches):
                trunc_normal_(self.sie_embed[i], std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.ParameterList([nn.Parameter(torch.zeros(view, 1, embed_dim[i])) for i in range(self.num_branches)])
            for i in range(self.num_branches):
                trunc_normal_(self.sie_embed[i], std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # self.blocks = nn.ModuleList([
        #     MultiScaleBlock(
        #         dim=embed_dim, patches=num_patches,num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
        #     for i in range(depth)])
        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)
        # print(len(self.blocks))
        # print('========fen ge ================')
        
        # print(self.blocks[-1])
        # print(self.blocks[:-1])
        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        # Classifier head
        self.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])
        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)

        self.apply(self._init_weights)

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
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id):
        # B = x.shape[0]
        # x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        
        # if self.cam_num > 0 and self.view_num > 0:
        #     xs = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        # elif self.cam_num > 0:
        #     x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        # elif self.view_num > 0:
        #     x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        # else:
        #     x = x + self.pos_embed

        # x = self.pos_drop(x)

        # if self.local_feature:
        #     for blk in self.blocks[:-1]:
        #         x = blk(x)
        #     return x

        # else:
        #     for blk in self.blocks:
        #         x = blk(x)

        #     x = self.norm(x)

        #     return x[:, 0]
        
        # NOTE 以下为2023.8.31由crossvit处修改而来
        B, C, H, W = x.shape
        # print(B,C,H,W)
        xs=[]
        for i in range(self.num_branches):
            x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
            # print(x_.shape)
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            # tmp = tmp + self.pos_embed[i]
            # tmp = self.pos_drop(tmp)
            if self.cam_num > 0 and self.view_num > 0:
                tmp=tmp+self.pos_embed[i]+self.sie_xishu*self.sie_embed[i][camera_id * self.view_num + view_id]
                tmp = self.pos_drop(tmp)
                xs.append(tmp)
            elif self.cam_num > 0:
                tmp=tmp+self.pos_embed[i]+self.sie_xishu*self.sie_embed[i][camera_id]
                tmp = self.pos_drop(tmp)
                xs.append(tmp)
            elif self.view_num > 0:
                tmp=tmp+self.pos_embed[i]+self.sie_xishu*self.sie_embed[i][view_id]
                tmp = self.pos_drop(tmp)
                xs.append(tmp)
            else:
                tmp = tmp + self.pos_embed[i]
                tmp = self.pos_drop(tmp)
                xs.append(tmp)
        if self.local_feature:  

            # print([xs[i].shape for i in range(2)])
            for blk in self.blocks[:-1]:
                # print(blk)
                # print('@@@@@@@@@@@@@@@@@')
                # b+=1
                xs = blk(xs)
 
            # xs = [self.norm[i](x) for i, x in enumerate(xs)]
            return xs
        else:
            # a=0
            for blk in self.blocks:
                # print(blk)
                # print('@@@@@@@@@@@@@@@@@')
             
                
                xs = blk(xs)
            
            xs = [self.norm[i](x) for i, x in enumerate(xs)]
            out = [x[:, 0] for x in xs]

            return out


    def forward(self, x, cam_label=None, view_label=None):
        xs = self.forward_features(x, cam_label, view_label)
        # print('##############################')
        # print(xs[0].shape)
        # print(xs[1].shape)
         # NOTE 2023.8.31
        # print(self.head)
        # ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        # ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)

        # return ce_logits
        return xs

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        # print(model_path)
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            # print(' k:>> ',k,' v:>>',v.shape)
            if 'patch_embed.0.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed[0].proj.weight.shape
                v = v.reshape(O, -1, H, W)
                # print('resize k: ',k,'resize v:',v.shape)
            elif 'patch_embed.1.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed[1].proj.weight.shape
                v = v.reshape(O, -1, H, W)
                # print('resize k: ',k,'resize v:',v.shape)
            elif k == 'pos_embed.0' and v.shape != self.pos_embed[0].shape:
                # print('v:',v.shape)
                # To resize pos embedding when using model at different size from pretrained weights当使用与预训练权值不同大小的模型时，调整pos嵌入的大小
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed[0], self.patch_embed[0].num_y, self.patch_embed[0].num_x)

                # print('resize k: ',k,'resize v:',v.shape)
            elif k == 'pos_embed.1' and v.shape != self.pos_embed[1].shape:
                # print('v:',v.shape)
                # To resize pos embedding when using model at different size from pretrained weights当使用与预训练权值不同大小的模型时，调整pos嵌入的大小
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed[1], self.patch_embed[1].num_y, self.patch_embed[1].num_x)

                # print('resize k: ',k,'resize v:',v.shape)
            try:
                
                self.state_dict()[k].copy_(v)#The size of tensor a (226) must match the size of tensor b (197) at non-singleton dimension 1
                # print('hello world')
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def crossvit_base_224_TransReID(img_size=(240, 224), stride_size=[12,16], drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=[12, 16], stride_size=stride_size, embed_dim=[384,768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]], num_heads=[12, 12], mlp_ratio=[4, 4, 1], qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, hybrid_backbone='1',**kwargs)

    return model

if __name__=="__main__":
    model=crossvit_base_224_TransReID()
    print(model)