import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from models.swin_encoder import SwinTransformer


class Model(torch.nn.Module):

    def __init__(self, device, rgb_backbone_name='SwinTransformer', out_indices=None, checkpoint_path='',
                 pool_last=False, xyz_backbone_name='Point_MAE', group_size=128, num_group=1024):
        super().__init__()
        # 'vit_base_patch8_224_dino'
        # Determine if to output features.
        self.device = device

        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})
        ####sigma-backbone##########################################################
        # self.rgb_backbone=Backbone_VSSM(
        #     depths=[2, 2, 9, 2], 
        #     dims=96,
        #     pretrained='/root/M3DM/pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
        #     downsample_version='v1',
        #     drop_path_rate=0.2,
        #     num_classes=1000,
        #     norm_layer=nn.LayerNorm,                
        #     mlp_ratio=0.0,               
        #     ape=False,
        #     img_size=[224, 224],
        #     patch_size=4,
        # )
        ####sigma-backbone##########################################################
        
        ## RGB backbone
        ########swintransformer##############################################
        # self.rgb_backbone = SwinTransformer(img_size=224,
        #                        embed_dim=128,
        #                        depths=[2, 2, 18, 2],
        #                        num_heads=[4, 8, 16, 32],
        #                        window_size=7)
        # pretrained_dict = torch.load('/root/M3DM/checkpoints/swin_base_patch4_window7_224.pth')["model"]
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.rgb_backbone.state_dict()}
        # self.rgb_backbone.load_state_dict(pretrained_dict)
        ######################################################################
        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True, pretrained_cfg_overlay=dict(file='/root/.cache/huggingface/hub/models--timm--vit_base_patch8_224.dino/pytorch_model.bin'),checkpoint_path=checkpoint_path,
                                          **kwargs)
        
        ## XYZ backbone
        
        if xyz_backbone_name=='Point_MAE':
            self.xyz_backbone=PointTransformer(group_size=group_size, num_group=num_group)
            self.xyz_backbone.load_model_from_ckpt("checkpoints/pointmae_pretrain.pth")
        elif xyz_backbone_name=='Point_Bert':
            self.xyz_backbone=PointTransformer(group_size=group_size, num_group=num_group, encoder_dims=256)
            self.xyz_backbone.load_model_from_pb_ckpt("checkpoints/Point-BERT.pth")



    def forward_rgb_features(self, x):
        
        
        x = self.rgb_backbone.patch_embed(x)
        x = self.rgb_backbone._pos_embed(x)
        x = self.rgb_backbone.norm_pre(x)
        if self.rgb_backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.rgb_backbone.blocks(x)
        x = self.rgb_backbone.norm(x)
        #view自动调整view中一个参数定为-1，代表自动调整这个维度上的元素个数，以保证元素的总数不变。
        '''
        x[:, 1:]：表示对张量 x 的所有行（第一个维度）和从第二列开始的所有列（从第二个维度开始）进行切片操作。换句话说，它去除了 x 张量的第一列数据，保留了其余部分。
        .permute(0, 2, 1)：表示对切片后的张量进行维度重排列操作。具体来说，它将原始张量的第二个维度（列）移动到第三个位置，而将原始张量的第三个维度（原始的第二列）移动到第二个位置。
        原始张量的第一个维度保持不变。
        .view(1, -1, 28, 28)：表示对重排列后的张量进行形状重塑操作。其中 1 表示在第一个维度上增加一个维度，-1 表示在第二个维度上自动计算大小，
        以使得总元素数量不变，而 28, 28 则表示最终形状为 (1, height, width)，即张量具有一个样本，高度为 28，宽度为 28。
         综上所述，这行代码的作用是对张量 x 进行切片、维度重排列和形状重塑操作，最终得到一个形状为 (1, height, width) 的特征张量 feat。
        '''
        #去除位置编码x[:,1:]
        #permute(0, 2, 1) (1,784,768)------>(1,768,784)
        #view(1, -1, 28, 28)==(1,768,784)------>(1,768,28,28)
        feat = x[:,1:].permute(0, 2, 1).view(1, -1, 28, 28).clone()
        """ctx.save_for_backward(feat)
        feat = feat.clone()"""
        return feat
        
        # features,x1,x2,x3,x4 = self.rgb_backbone(x) 
        # x1,x2,x3,x4 = self.rgb_backbone(x)
#         x = self.rgb_backbone.patch_embed(x)
#         x = self.rgb_backbone._pos_embed(x)
#         x = self.rgb_backbone.norm_pre(x)
#         if self.rgb_backbone.grad_checkpointing and not torch.jit.is_scripting():
#             x = checkpoint_seq(self.blocks, x)
#         else:
#             x = self.rgb_backbone.blocks(x)
#         x = self.rgb_backbone.norm(x)

#         feat = x[:,1:].permute(0, 2, 1).view(1, -1, 28, 28).clone()
        # return feat
        # return x1,x2,x3,x4


    def forward(self, rgb, xyz):
        
#         x1,x2,x3,x4 = self.forward_rgb_features(rgb)

#         xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz)

#         # return rgb_features, xyz_features, center, ori_idx, center_idx
#         return x1,x2,x3,x4, xyz_features, center, ori_idx, center_idx
        rgb_features = self.forward_rgb_features(rgb)

        xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz)

        return rgb_features, xyz_features, center, ori_idx, center_idx


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center, center_idx = fps(xyz.contiguous(), self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class PointTransformer(nn.Module):
    def __init__(self, group_size=128, num_group=1024, encoder_dims=384):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6

        self.group_size = group_size
        self.num_group = num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            #if incompatible.missing_keys:
            #    print('missing_keys')
            #    print(
            #            incompatible.missing_keys
            #        )
            #if incompatible.unexpected_keys:
            #    print('unexpected_keys')
            #    print(
            #            incompatible.unexpected_keys

            #        )

            # print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                    incompatible.missing_keys
                )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                    incompatible.unexpected_keys

                )
                
        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')


    def forward(self, pts):
        if self.encoder_dims != self.trans_dim:
            B,C,N = pts.shape
            pts = pts.transpose(-1, -2) # B N 3
            # divide the point clo  ud in the same form. This is important
            neighborhood,  center, ori_idx, center_idx = self.group_divider(pts)
            # # generate mask
            # bool_masked_pos = self._mask_center(center, no_mask = False) # B G
            # encoder the input cloud blocks
            group_input_tokens = self.encoder(neighborhood)  #  B G N
            group_input_tokens = self.reduce_dim(group_input_tokens)
            # prepare cls
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
            # add pos embedding
            pos = self.pos_embed(center)
            # final input
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
            return x, center, ori_idx, center_idx 
        else:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)  # B N 3
            # divide the point clo  ud in the same form. This is important
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)

            group_input_tokens = self.encoder(neighborhood)  # B G N

            pos = self.pos_embed(center)
            # final input
            x = group_input_tokens
            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
            return x, center, ori_idx, center_idx
        