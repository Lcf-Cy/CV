import torch
from sklearn.decomposition import PCA
from torch import nn

from feature_extractors.features import Features
from utils.mvtec3d_util import *

import numpy as np
torch.set_printoptions(threshold=np.inf)
import math
import os
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * self.sigmoid(attn)


class ChannelAttentionBlock(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(ChannelAttentionBlock, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x) 
    
    
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y
        # return x * y.expand_as(x)




from sklearn.neighbors import NearestNeighbors

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
    

class SelfAttentionModule(nn.Module):
    def __init__(self, input_dim, head_dim, num_heads):
        super(SelfAttentionModule, self).__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, head_dim * num_heads)
        self.key = nn.Linear(input_dim, head_dim * num_heads)
        self.value = nn.Linear(input_dim, head_dim * num_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(head_dim * num_heads, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 计算查询、键和值
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # 应用softmax归一化
        attention_probs = self.softmax(attention_scores)

        # 加权求和
        weighted_sum = torch.matmul(attention_probs, value)

        # 将多个头级联
        weighted_sum = weighted_sum.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 线性变换
        output = self.linear(weighted_sum)

        return output
###########不对称卷积，扩大感受野#############################################################   
###########################################################################
class CropLayer(nn.Module):
    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class asyConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

class ERF(nn.Module):
    def __init__(self, x, y):
        super(ERF, self).__init__()
        self.asyConv = asyConv(in_channels=x, out_channels=y, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.oriConv = nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.atrConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=3, padding=3, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )
        self.conv2d = nn.Conv2d(y*2, y, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(y)
        self.res = BasicConv2d(x, y, 1)

    def forward(self, f):
        p2 = self.asyConv(f)
        p3 = self.atrConv(f)
        p  = torch.cat((p2, p3), 1)
        p  = F.relu(self.bn2d(self.conv2d(p)), inplace=True)

        return p

###########################################################################
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CSFF(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(CSFF, self).__init__()
        self.cat2 = BasicConv2d(hidden_channels * 2, out_channels, kernel_size=3, padding=1)
        self.param_free_norm = nn.BatchNorm2d(hidden_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, x, y, edge):
        xy = self.cat2(torch.cat((x, y), dim=1)) + y + x
        normalized = self.param_free_norm(xy)

        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(edge)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out

    
class RGBFeatures(Features):
#sample=[rgb_feature_maps, xyz_feature_maps, center, ori_idx, center_idx]
    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]#sample[1]=xyz_feature_maps
        # print(organized_pc[0].shape)
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()#(1,768,784)-->(768,784,1)
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)#(768,784,1)-->(768*784,1)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        #sample[0]---->(1,768,28,28)
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        #rgb_feature_maps---->(1,768,28,28)
        # rgb_feature_maps, xyz_feature_maps, _, _, center_idx, _ = self(sample[0],unorganized_pc_no_zeros.contiguous())
        #sigma
        # x1,x2,x3,x4, xyz_feature_maps, _, _, center_idx, _ = self(sample[0],unorganized_pc_no_zeros.contiguous())
        x4,x3,x2,x1, xyz_feature_maps, _, _, center_idx, _ = self(sample[0],unorganized_pc_no_zeros.contiguous())
        '''
        torch.Size([1, 128, 56, 56])
        torch.Size([1, 256, 28, 28])
        torch.Size([1, 512, 14, 14])
        torch.Size([1, 1024, 7, 7])
        '''
#         size = x2.size()[2:]
      
    
#         x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
#         rgb_patch=torch.cat([x3,x2],dim=1)
        cat2 = BasicConv2d(128, 640, kernel_size=3, padding=1)
        rgb_patch=cat2(x1)
        # print(rgb_patch.shape)
        
        
        # rgb_feature_maps---->(1,768,28,28)
        #rgb_patch----->(784,768)784个patch
        # rgb_patch=x1
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        # print(rgb_patch.shape)
        # print(x1.shape[-2:])

        self.patch_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]#sample[1]=xyz_feature_maps
        # print(organized_pc.shape)
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()#(1,768,784)-->(768,784,1)
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)#(768,784,1)-->(768*784,1)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        #sample[0]---->(1,768,28,28)
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        #rgb_feature_maps---->(1,768,28,28)
        # rgb_feature_maps, xyz_feature_maps, _, _, center_idx, _ = self(sample[0],unorganized_pc_no_zeros.contiguous())
        x4,x3,x2,x1,xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0],unorganized_pc_no_zeros.contiguous())
        
        
#         size = x2.size()[2:]
        
        
#         x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
#         rgb_patch=torch.cat([x2,x3],dim=1)
        cat2 = BasicConv2d(128, 640, kernel_size=3, padding=1)
        rgb_patch=cat2(x1)
        # print(rgb_patch.shape)
        
        
        # rgb_feature_maps---->(1,768,28,28)
        #rgb_patch----->(784,768)784个patch
        # rgb_patch=x1
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        

        self.compute_s_s_map(rgb_patch, x1.shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)
        self.mean = torch.mean(self.patch_lib)
        self.std = torch.std(self.patch_lib)
        self.patch_lib = (self.patch_lib - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        with torch.no_grad():
            if self.f_coreset < 1:
                self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                                n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                                eps=self.coreset_eps, )
                self.patch_lib = self.patch_lib[self.coreset_idx]


    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        patch = (patch - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

class PointFeatures(Features):

    def add_sample_to_mem_bank(self, sample,class_name=None):
        organized_pc = sample[1]
        # print(organized_pc)
        #squeeze()去除尺寸为1的维度，permute交换维度
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        # print(unorganized_pc)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        # print(len(nonzero_indices))
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        #创建一个与完整特征图相同大小的全零张量，并将插值后的点云数据填充到相应的位置。
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
 
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        # print(xyz_patch)
        #xyz_patch(3136,1152)
        self.patch_lib.append(xyz_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        self.compute_s_s_map(xyz_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.args.rm_zero_for_project:
            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib!=0, dim=1))[:,0]]

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]
            
        if self.args.rm_zero_for_project:

            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib!=0, dim=1))[:,0]]
            self.patch_lib = torch.cat((self.patch_lib, torch.zeros(1, self.patch_lib.shape[1])), 0)


    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''


        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

FUSION_BLOCK= True

class FusionFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        # print(organized_pc)
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))#28
        #交换重塑(768,56,56)
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        #(3136,768)
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, rgb_patch2], dim=1)

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_lib.append(fusion_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, rgb_patch2], dim=1)

        self.compute_s_s_map(fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps)
            self.patch_lib = self.patch_lib[self.coreset_idx]

class DoubleRGBPointFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb]])
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        s_map = s_map.view(1, 224, 224)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

class DoubleRGBPointFeatures_add(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)

        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
    
        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[s_xyz, s_rgb]])
        s_map = torch.cat([s_map_xyz, s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = s_xyz + s_rgb
        s_map = s_map_xyz + s_map_rgb
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map

class TripleFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)

        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:

            xyz_patch = torch.cat(xyz_feature_maps, 1)
            xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
            xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
            xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
            xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

            xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
            xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

            if FUSION_BLOCK:
                with torch.no_grad():
                    fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
                fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            else:
                fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

            self.patch_xyz_lib.append(xyz_patch)
            self.patch_fusion_lib.append(fusion_patch)
    

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    

        self.compute_s_s_map(xyz_patch, rgb_patch, fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 =  self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
    
        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')
        

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean)/self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]


        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib!=0, dim=1))[:,0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)


    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean)/self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size =  (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

  
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')

        s = torch.tensor([[self.args.xyz_s_lambda*s_xyz, self.args.rgb_s_lambda*s_rgb, self.args.fusion_s_lambda*s_fusion]])
 
        s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb, self.args.fusion_smap_lambda*s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)
 
        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
  
        s_map = s_map.view(1, self.image_size, self.image_size)


        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) 
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
    
    
    
    
    
    
    
    
    
    
    
###############################################################################################################################################
class KNNFusionFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # 转为numpy数据类型
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        # 构建rgb块特征以及块特征对应的索引 #######################################################
        # 点云快特征
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        # rgb块特征
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        # rgb_patch(784,768)个patch
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))  # rgb_patch_size=28
        # 交换重塑(768,56,56)
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        # (3136,768)
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T
        rgb_patch2_indices = torch.arange(rgb_patch2.shape[0])
        #####################################################################################

        #####xyz_patch#######################################################################
        # 构造0张量（1，interpolated_pc.shape[1]，224*224）
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        # 将interpolated_pc的值赋给xyz_patch_full中指定的非零索引位置，以更新点云数据的部分。
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        # resize(_,3136)
        # self.average = torch.nn.AvgPool2d(3, stride=1)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))  # 56*56
        # (3136,_)
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        # print(xyz_patch)
        # 点云快特征索引
        xyz_patch_indices = torch.arange(xyz_patch.shape[0])

        #################rgb_KDtree###################################################################
        # 定义 k 值
        k = 5
        # 构建 k 近邻模型
        knn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree')  # k+1，因为每个块本身也是最近的邻居
        knn_model.fit(rgb_patch2)
        # 寻找每个块的 k 近邻
        distances, indices = knn_model.kneighbors(rgb_patch2)
        # distances 为每个块与其 k 近邻块的距离列表，indices 为每个块的 k 近邻块的索引列表
        # 去除每个块自身作为最近邻的项
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        # # 提取 k 近邻特征
        knn_features_rgb = rgb_patch2[indices]
        # print(knn_features.shape)
        # # 将 k 近邻特征返回,每一块的5个k近邻特征
        # knn_features_rgb = knn_features_rgb.reshape(-1, k, block_size * block_size * 3)

        #################xyz_KDtree###################################################################
        # knn_model.fit(xyz_patch)
        # # 寻找每个块的 k 近邻
        # distances, indices = knn_model.kneighbors(xyz_patch)
        # # distances 为每个块与其 k 近邻块的距离列表，indices 为每个块的 k 近邻块的索引列表
        # # 去除每个块自身作为最近邻的项
        # distances = distances[:, 1:]
        # indices = indices[:, 1:]
        # # # 提取 k 近邻特征(3136,k,_)
        # knn_features_xyz = xyz_patch[indices]

        ############################rgb_patch和xyz_patch进行1-to-k拼接############################################################
        # adjusted_knn_features_xyz = np.expand_dims(knn_features_xyz, axis=2)
        # adjusted_rgb_patch = np.expand_dims(rgb_patch2, axis=1)  # （3136，1，768）
        # adjusted_rgb_patch1 = np.expand_dims(adjusted_rgb_patch, axis=1)
        # resized_rgb_patch1_array = np.resize(adjusted_rgb_patch1, adjusted_knn_features_xyz.shape)
        # concatenated_features_rgb_k_xyz = np.concatenate((resized_rgb_patch1_array, adjusted_knn_features_xyz), axis=2)

        # concatenated_features_rgb_k_xyz(3136,k,2,768)
        ############################xyz_patch和rgb_patch进行1-to-k拼接############################################################
        # adjusted_knn_features_rgb = np.expand_dims(knn_features_rgb, axis=2)#rgb的k近邻特征
        adjusted_xyz_patch = np.expand_dims(xyz_patch, axis=1)  # （3136，1，768）
        # adjusted_xyz_patch1 = np.expand_dims(adjusted_xyz_patch, axis=1)
        # resized_xyz_patch1_array = np.resize(adjusted_xyz_patch1, adjusted_knn_features_rgb.shape)
        # concatenated_features_xyz_k_rgb = np.concatenate((resized_xyz_patch1_array, adjusted_knn_features_rgb), axis=2)

        # concatenated_features_xyz_k_rgb(3136,k,2,768)
        ########################################################################################################################
        # 通道拼接rgb----k----xyz(3136,1,768)-->
        # x0 = knn_features_xyz.shape[0]
        # x1 = knn_features_xyz.shape[1]
        # x2 = knn_features_xyz.shape[2]
        # adjusted_rgb_patch_k = np.resize(adjusted_rgb_patch, (x0, x1, x2))
        # concatenated_rgb_k = torch.tensor(np.concatenate((adjusted_rgb_patch_k, knn_features_xyz), axis=2))
        # (3136,k,_)
        r0 = knn_features_rgb.shape[0]
        r1 = knn_features_rgb.shape[1]
        r2 = knn_features_rgb.shape[2]
        adjusted_xyz_patch_k = np.resize(adjusted_xyz_patch, (r0, r1, r2))
        concatenated_xyz_k = torch.from_numpy(np.concatenate((adjusted_xyz_patch_k, knn_features_rgb), axis=2))
        # (3136,5,1536)
        # print(concatenated_xyz_k.shape)
        ########################自注意力机制代码################################################################################################
        # self_attention1 = SelfAttentionModule(input_dim=concatenated_rgb_k.shape[2], head_dim=64, num_heads=12)
        # output_rgb = self_attention1.forward(concatenated_rgb_k)
        # fused_output_rgb = torch.mean(output_rgb, dim=1)  # (3136,2304)
        # print("fused_output_rgb.shape:", fused_output_rgb.shape)
        self_attention2 = SelfAttentionModule(input_dim=concatenated_xyz_k.shape[2], head_dim=64, num_heads=12)
        output_xyz = self_attention2.forward(concatenated_xyz_k)
        fused_output_xyz = torch.mean(output_xyz, dim=1)  # (3136,1536)
        # print("fused_output_xyz.shape:", fused_output_xyz.shape)
        with torch.no_grad():
            fusion_patch = torch.cat([rgb_patch2, fused_output_xyz], dim=1)
        # fusion_patch=fusion_patch.detach().numpy()
        # print(fusion_patch.shape)
            pooling = nn.AdaptiveAvgPool1d(768)

        # 将数据传递给平均池化层
            fusion_patch_reduced = pooling(fusion_patch.unsqueeze(0)).squeeze(0)
        # print(fusion_patch_reduced.shape)
        if class_name is not None:
            torch.save(fusion_patch_reduced, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1
        
        self.patch_lib.append(fusion_patch_reduced)


    ###################################################################################################################################
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # 转为numpy数据类型
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        # 构建rgb块特征以及块特征对应的索引 #######################################################
        # 点云快特征
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        # rgb块特征
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        # rgb_patch(784,768)个patch
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))  # rgb_patch_size=28
        # 交换重塑(768,56,56)
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        # (3136,768)
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T
        rgb_patch2_indices = torch.arange(rgb_patch2.shape[0])
        #####################################################################################

        #####xyz_patch#######################################################################
        # 构造0张量（1，interpolated_pc.shape[1]，224*224）
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        # 将interpolated_pc的值赋给xyz_patch_full中指定的非零索引位置，以更新点云数据的部分。
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        # resize(_,3136)
        # self.average = torch.nn.AvgPool2d(3, stride=1)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))  # 56*56
        # (3136,_)
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        # 点云快特征索引
        xyz_patch_indices = torch.arange(xyz_patch.shape[0])

        #################rgb_KDtree###################################################################
        # 定义 k 值
        k = 5
        # 构建 k 近邻模型
        knn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree')  # k+1，因为每个块本身也是最近的邻居
        knn_model.fit(rgb_patch2)
        # 寻找每个块的 k 近邻
        distances, indices = knn_model.kneighbors(rgb_patch2)
        # distances 为每个块与其 k 近邻块的距离列表，indices 为每个块的 k 近邻块的索引列表
        # 去除每个块自身作为最近邻的项
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        # # 提取 k 近邻特征
        knn_features_rgb = rgb_patch2[indices]
        # print(knn_features.shape)
        # # 将 k 近邻特征返回,每一块的5个k近邻特征
        # knn_features_rgb = knn_features_rgb.reshape(-1, k, block_size * block_size * 3)

        #################xyz_KDtree###################################################################
        # knn_model.fit(xyz_patch)
        # # 寻找每个块的 k 近邻
        # distances, indices = knn_model.kneighbors(xyz_patch)
        # # distances 为每个块与其 k 近邻块的距离列表，indices 为每个块的 k 近邻块的索引列表
        # # 去除每个块自身作为最近邻的项
        # distances = distances[:, 1:]
        # indices = indices[:, 1:]
        # # # 提取 k 近邻特征(3136,k,_)
        # knn_features_xyz = xyz_patch[indices]

        ############################rgb_patch和xyz_patch进行1-to-k拼接############################################################
        # adjusted_knn_features_xyz = np.expand_dims(knn_features_xyz, axis=2)
        # adjusted_rgb_patch = np.expand_dims(rgb_patch2, axis=1)  # （3136，1，768）
        # adjusted_rgb_patch1 = np.expand_dims(adjusted_rgb_patch, axis=1)
        # resized_rgb_patch1_array = np.resize(adjusted_rgb_patch1, adjusted_knn_features_xyz.shape)
        # concatenated_features_rgb_k_xyz = np.concatenate((resized_rgb_patch1_array, adjusted_knn_features_xyz), axis=2)

        # concatenated_features_rgb_k_xyz(3136,k,2,768)
        ############################xyz_patch和rgb_patch进行1-to-k拼接############################################################
        # adjusted_knn_features_rgb = np.expand_dims(knn_features_rgb, axis=2)#rgb的k近邻特征
        adjusted_xyz_patch = np.expand_dims(xyz_patch, axis=1)  # （3136，1，768）
        # adjusted_xyz_patch1 = np.expand_dims(adjusted_xyz_patch, axis=1)
        # resized_xyz_patch1_array = np.resize(adjusted_xyz_patch1, adjusted_knn_features_rgb.shape)
        # concatenated_features_xyz_k_rgb = np.concatenate((resized_xyz_patch1_array, adjusted_knn_features_rgb), axis=2)

        # concatenated_features_xyz_k_rgb(3136,k,2,768)
        ########################################################################################################################
        # 通道拼接rgb----k----xyz(3136,1,768)-->
        # x0 = knn_features_xyz.shape[0]
        # x1 = knn_features_xyz.shape[1]
        # x2 = knn_features_xyz.shape[2]
        # adjusted_rgb_patch_k = np.resize(adjusted_rgb_patch, (x0, x1, x2))
        # concatenated_rgb_k = torch.tensor(np.concatenate((adjusted_rgb_patch_k, knn_features_xyz), axis=2))
        # (3136,k,_)
        r0 = knn_features_rgb.shape[0]
        r1 = knn_features_rgb.shape[1]
        r2 = knn_features_rgb.shape[2]
        adjusted_xyz_patch_k = np.resize(adjusted_xyz_patch, (r0, r1, r2))
        concatenated_xyz_k = torch.tensor(np.concatenate((adjusted_xyz_patch_k, knn_features_rgb), axis=2))
        # (3136,5,1536)
        # print(concatenated_xyz_k.shape)
        ########################自注意力机制代码################################################################################################
        # self_attention1 = SelfAttentionModule(input_dim=concatenated_rgb_k.shape[2], head_dim=64, num_heads=12)
        # output_rgb = self_attention1.forward(concatenated_rgb_k)
        # fused_output_rgb = torch.mean(output_rgb, dim=1)  # (3136,2304)
        # print("fused_output_rgb.shape:", fused_output_rgb.shape)
        self_attention2 = SelfAttentionModule(input_dim=concatenated_xyz_k.shape[2], head_dim=64, num_heads=12)
        output_xyz = self_attention2.forward(concatenated_xyz_k)
        fused_output_xyz = torch.mean(output_xyz, dim=1)  # (3136,1536)
        # print("fused_output_xyz.shape:", fused_output_xyz.shape)
        with torch.no_grad():
            fusion_patch = torch.cat([rgb_patch2, fused_output_xyz], dim=1)
        # print(fusion_patch.shape)
            pooling = nn.AdaptiveAvgPool1d(768)

        # 将数据传递给平均池化层
            fusion_patch_reduced = pooling(fusion_patch.unsqueeze(0)).squeeze(0)

        self.compute_s_s_map(fusion_patch_reduced, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx,
                             nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz,
                        center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        # print(self.patch_lib.shape)
        # print(self.patch_lib)
        self.mean = torch.mean(self.patch_lib)
        self.std = torch.std(self.patch_lib)
        self.patch_lib = (self.patch_lib - self.mean)/self.std
        # print(self.patch_lib.shape)
        # print(self.patch_lib)
        # self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps)
            self.patch_lib = self.patch_lib[self.coreset_idx]

            
            
class MambaFusionFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        # print("organized_pc:",organized_pc.is_cuda)
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # 转为numpy数据类型
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        # 构建rgb块特征以及块特征对应的索引 #######################################################
        # 点云快特征
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        
        # print("xyz_patch:",xyz_patch.shape)
        # rgb块特征
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        # print(rgb_patch.shape)
        rgb_patch_two_dim=rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        rgb_patch_size = int(math.sqrt(rgb_patch_two_dim.shape[0]))  # 28
        # 交换重塑(768,56,56)
        rgb_patch2 = self.resize2(rgb_patch_two_dim.permute(1, 0).reshape(1,-1, rgb_patch_size, rgb_patch_size))
        # (3136,768)
        rgb_patch2_forKNN = rgb_patch2.reshape(rgb_patch.shape[1], -1).T#784,768
        
        # print(rgb_patch2_forKNN.shape)#1,768,28,28
   
        #####xyz_patch#######################################################################
        # 构造0张量（1，interpolated_pc.shape[1]，224*224）
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        # print("xyz_patch_full:",xyz_patch_full.shape)
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc
        
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        
        xyz_patch_full_resized = self.resize2(self.average(xyz_patch_full_2d))  # 56*56
        # print(xyz_patch_full_resized.shape)
        
        xyz_patch_forKNN = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T#3136,1152
        # print(xyz_patch_forKNN.shape)#784,1152
        ######################################################################################
        #KNN特征增强
        k = 6
        knn_model = NearestNeighbors(n_neighbors=k+1)  
        knn_model.fit(rgb_patch2_forKNN)
        distances, indices = knn_model.kneighbors(rgb_patch2_forKNN)
        indices = indices[:, 1:]
        # print(indices)
        # # 提取 k 近邻特征
        knn_features_xyz = xyz_patch_forKNN[indices]
        adjusted_rgb_patch_forKNN = torch.unsqueeze(rgb_patch2_forKNN, dim=1)
        adjusted_rgb_patch_k=adjusted_rgb_patch_forKNN.repeat(1, k, 1)
        concatenated_rgb_k=torch.cat([knn_features_xyz,adjusted_rgb_patch_k],dim=2)
        
        # mlp_ratio=4
        # xyz_norm = nn.LayerNorm(knn_features_xyz.shape[2])#1152
        # rgb_norm = nn.LayerNorm(adjusted_rgb_patch_k.shape[2])#768
        # xyz_mlp = Mlp(in_features=knn_features_xyz.shape[2], hidden_features=int(knn_features_xyz.shape[2] * mlp_ratio), act_layer=nn.GELU, drop=0.)
        # rgb_mlp = Mlp(in_features=adjusted_rgb_patch_k.shape[2], hidden_features=int(adjusted_rgb_patch_k.shape[2] * mlp_ratio), act_layer=nn.GELU, drop=0.)
        # xyz_feature  = xyz_mlp(xyz_norm(knn_features_xyz))
        # rgb_feature  = rgb_mlp(rgb_norm(adjusted_rgb_patch_k))
        # concatenated_rgb_k = torch.cat([xyz_feature, rgb_feature], dim=2)
        # concatenated_rgb_k = xyz_feature+rgb_feature
        self_attention2 = SelfAttentionModule(input_dim=concatenated_rgb_k.shape[2], head_dim=64, num_heads=16)
        output_rgb = self_attention2.forward(concatenated_rgb_k)
       
        fused_output_rgb,_= torch.max(output_rgb, dim=1)  # (784,768)
        # print(fused_output_rgb.shape)#784,1920
        fused_output_rgb=torch.cat([fused_output_rgb,rgb_patch2_forKNN],dim=1)
        
        #######################################################################################
        conv_layer = nn.Conv2d(1152, 768, kernel_size=1)
        conv_layer1 = nn.Conv2d(3456, 1536, kernel_size=1)
        xyz_patch=conv_layer(xyz_patch_full_resized)
        
        # print(xyz_patch.shape)
        # pooling = nn.AdaptiveAvgPool1d(768)
        # xyz_patch=pooling(xyz_patch_full_resized)
        
        # transposed_conv = nn.ConvTranspose2d(768, 1152, kernel_size=1, stride=1, padding=0)
        # rgb_patch=transposed_conv(rgb_patch)
        # print(rgb_patch.shape)
        
        # fusion_patch=torch.cat([rgb_patch,xyz_patch],dim=1)
        # fusion_patch=fusion_patch.reshape(fusion_patch.shape[1], -1).T
        # print(fusion_patch.shape)
        # print(rgb_patch2.shape)
        # print(xyz_patch.shape)
        rgb_patch_mamba= rgb_patch2.to(self.device)
        xyz_patch_mamba = xyz_patch.to(self.device)
        # outs_rgb_resize=outs_rgb_resize.to(self.device)
        rgb_patch = rgb_patch.to(self.device)
        
#         print(rgb_patch_mamba.shape)
#         print(xyz_patch_mamba.shape)
        
        # dims=192
        # with torch.no_grad():
        cross_mamba = CrossMambaFusionBlock(
                hidden_dim=768,
                # hidden_dim=1152,
                mlp_ratio=4.0,
                d_state=4,
            ) 

        cross_rgb, cross_x=cross_mamba(rgb_patch_mamba.permute(0, 2, 3, 1).contiguous(),xyz_patch_mamba.permute(0, 2, 3, 1).contiguous())


        # print(cross_xyz_chen.shape)
        #1,56,56,768
        cross_rgb_resize=cross_rgb.permute(0, 3, 1, 2).contiguous()#1,768,56,56
        cross_x_resize=cross_x.permute(0, 3, 1, 2).contiguous()

        channel_attention1 = ChannelAttention(channel=cross_rgb_resize.shape[1])
        attention_weights1 = channel_attention1(cross_rgb_resize)

        channel_attention2 = ChannelAttention(channel=cross_x_resize.shape[1])
        attention_weights2 = channel_attention2(cross_x_resize)

        cross_rgb_resize_enhance=cross_rgb_resize * attention_weights1.expand_as(cross_rgb_resize)
        # print(cross_rgb_resize_enhance.shape)

        cross_x_resize_enhance=cross_x_resize * attention_weights2.expand_as(cross_x_resize)
            # x_fuse =cross_rgb_resize_enhance+cross_x_resize_enhance
        with torch.no_grad():
            x_fuse=torch.cat([cross_rgb_resize_enhance,cross_x_resize_enhance],dim=1)
            
            
            
#             channel_attn_mamba =  ConcatMambaFusionBlock(
#                     hidden_dim=768,
#                     # hidden_dim=1152,
#                     mlp_ratio=4.0,
#                     d_state=4,
#                 ) 
#             x_fuse = channel_attn_mamba(cross_rgb, cross_x).permute(0, 3, 1, 2).contiguous()
            # x_fuse = x_fuse.permute(0, 3, 1, 2).contiguous()
            # rgb_patch=rgb_patch.reshape(x_fuse.shape[1], -1).T
            # print(rgb_patch.shape)
            # x_fuse = x_fuse.reshape(x_fuse.shape[1], -1).T.cpu()
            # print(x_fuse.shape)
            ########################################################
            x_fuse=x_fuse.cpu()
            
            # x_fuse=torch.cat([x_fuse,rgb_patch],dim=1)
          
            # spatial_attn = SpatialAttention(kernel_size=3)
            # attention_weights_space=spatial_attn(x_fuse)
            # # print(attention_weights_space.shape)
            # x_fuse=x_fuse * attention_weights_space.expand_as(x_fuse)
            #########################################################
            # channel_attention = ChannelAttention(channel=x_fuse.shape[1])
            # attention_weights = channel_attention(x_fuse)
            
            # channel_attention = ChannelAttention(channel=rgb_patch.shape[1])
            
          
            # attention_weights = channel_attention(rgb_patch)
            
            # x_fuse=xyz_patch * attention_weights.expand_as(xyz_patch)
            
            # x_fuse=x_fuse+xyz_patch
            # x_fuse=rgb_patch * attention_weights.expand_as(rgb_patch)
            
            # x_fuse=xyz_patch
            # rgb=cross_rgb_resize_enhance.reshape(x_fuse.shape[1], -1).T
            # print(rgb.shape)
            # xyz=cross_x_resize_enhance.reshape(x_fuse.shape[1], -1).T
            x_fuse=x_fuse.reshape(x_fuse.shape[1], -1).T
            fusion=torch.cat([x_fuse,fused_output_rgb],dim=1)
            # fusion=fused_output_rgb
            # print(fusion.shape)
            
            # fusion=rgb_patch2
            # fusion=xyz_patch.reshape(xyz_patch.shape[1], -1).T
            # x_fuse=attention_weights*rgb_patch
            # print(x_fuse.shape)
            # print(attention_weights.shape)
            
            # x_fuse=fusion_patch
        if class_name is not None:
            torch.save(fusion, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1
        
        self.patch_lib.append(fusion)
        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch2)
        torch.cuda.empty_cache()

    ###################################################################################################################################
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        # print("organized_pc:",organized_pc.is_cuda)
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # 转为numpy数据类型
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())
        # 构建rgb块特征以及块特征对应的索引 #######################################################
        # 点云快特征
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        
        # print("xyz_patch:",xyz_patch.shape)
        # rgb块特征
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        # print(rgb_patch.shape)
        rgb_patch_two_dim=rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        rgb_patch_size = int(math.sqrt(rgb_patch_two_dim.shape[0]))  # 28
        # 交换重塑(768,56,56)
        rgb_patch2 = self.resize2(rgb_patch_two_dim.permute(1, 0).reshape(1,-1, rgb_patch_size, rgb_patch_size))
        # (3136,768)
        rgb_patch2_forKNN = rgb_patch2.reshape(rgb_patch.shape[1], -1).T#784,768
        
        # print(rgb_patch2_forKNN.shape)#1,768,28,28
   
        #####xyz_patch#######################################################################
        # 构造0张量（1，interpolated_pc.shape[1]，224*224）
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        # print("xyz_patch_full:",xyz_patch_full.shape)
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc
        
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        
        xyz_patch_full_resized = self.resize2(self.average(xyz_patch_full_2d))  # 56*56
        # print(xyz_patch_full_resized.shape)
        
        xyz_patch_forKNN = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T#3136,1152
        # print(xyz_patch_forKNN.shape)#784,1152
        ######################################################################################
        #KNN特征增强
        k = 6
        knn_model = NearestNeighbors(n_neighbors=k+1)  
        knn_model.fit(rgb_patch2_forKNN)
        distances, indices = knn_model.kneighbors(rgb_patch2_forKNN)
        indices = indices[:, 1:]
        # print(indices)
        # # 提取 k 近邻特征
        knn_features_xyz = xyz_patch_forKNN[indices]
        adjusted_rgb_patch_forKNN = torch.unsqueeze(rgb_patch2_forKNN, dim=1)
        adjusted_rgb_patch_k=adjusted_rgb_patch_forKNN.repeat(1, k, 1)
        concatenated_rgb_k=torch.cat([knn_features_xyz,adjusted_rgb_patch_k],dim=2)
        
        # mlp_ratio=4
        # xyz_norm = nn.LayerNorm(knn_features_xyz.shape[2])#1152
        # rgb_norm = nn.LayerNorm(adjusted_rgb_patch_k.shape[2])#768
        # xyz_mlp = Mlp(in_features=knn_features_xyz.shape[2], hidden_features=int(knn_features_xyz.shape[2] * mlp_ratio), act_layer=nn.GELU, drop=0.)
        # rgb_mlp = Mlp(in_features=adjusted_rgb_patch_k.shape[2], hidden_features=int(adjusted_rgb_patch_k.shape[2] * mlp_ratio), act_layer=nn.GELU, drop=0.)
        # xyz_feature  = xyz_mlp(xyz_norm(knn_features_xyz))
        # rgb_feature  = rgb_mlp(rgb_norm(adjusted_rgb_patch_k))
        # concatenated_rgb_k = torch.cat([xyz_feature, rgb_feature], dim=2)
        # concatenated_rgb_k = xyz_feature+rgb_feature
        self_attention2 = SelfAttentionModule(input_dim=concatenated_rgb_k.shape[2], head_dim=64, num_heads=16)
        output_rgb = self_attention2.forward(concatenated_rgb_k)
       
        fused_output_rgb,_= torch.max(output_rgb, dim=1)  # (784,768)
        # print(fused_output_rgb.shape)#784,1920
        fused_output_rgb=torch.cat([fused_output_rgb,rgb_patch2_forKNN],dim=1)
        
        #######################################################################################
        conv_layer = nn.Conv2d(1152, 768, kernel_size=1)
        conv_layer1 = nn.Conv2d(3456, 1536, kernel_size=1)
        xyz_patch=conv_layer(xyz_patch_full_resized)
        
        # print(xyz_patch.shape)
        # pooling = nn.AdaptiveAvgPool1d(768)
        # xyz_patch=pooling(xyz_patch_full_resized)
        
        # transposed_conv = nn.ConvTranspose2d(768, 1152, kernel_size=1, stride=1, padding=0)
        # rgb_patch=transposed_conv(rgb_patch)
        # print(rgb_patch.shape)
        
        # fusion_patch=torch.cat([rgb_patch,xyz_patch],dim=1)
        # fusion_patch=fusion_patch.reshape(fusion_patch.shape[1], -1).T
        # print(fusion_patch.shape)
        # print(rgb_patch2.shape)
        # print(xyz_patch.shape)
        rgb_patch_mamba= rgb_patch2.to(self.device)
        xyz_patch_mamba = xyz_patch.to(self.device)
        # outs_rgb_resize=outs_rgb_resize.to(self.device)
        rgb_patch = rgb_patch.to(self.device)
        
#         print(rgb_patch_mamba.shape)
#         print(xyz_patch_mamba.shape)
        
        # dims=192
        # with torch.no_grad():
        cross_mamba = CrossMambaFusionBlock(
                hidden_dim=768,
                # hidden_dim=1152,
                mlp_ratio=4.0,
                d_state=4,
            ) 

        cross_rgb, cross_x=cross_mamba(rgb_patch_mamba.permute(0, 2, 3, 1).contiguous(),xyz_patch_mamba.permute(0, 2, 3, 1).contiguous())


        # print(cross_xyz_chen.shape)
        #1,56,56,768
        cross_rgb_resize=cross_rgb.permute(0, 3, 1, 2).contiguous()#1,768,56,56
        cross_x_resize=cross_x.permute(0, 3, 1, 2).contiguous()

        channel_attention1 = ChannelAttention(channel=cross_rgb_resize.shape[1])
        attention_weights1 = channel_attention1(cross_rgb_resize)

        channel_attention2 = ChannelAttention(channel=cross_x_resize.shape[1])
        attention_weights2 = channel_attention2(cross_x_resize)

        cross_rgb_resize_enhance=cross_rgb_resize * attention_weights1.expand_as(cross_rgb_resize)
        # print(cross_rgb_resize_enhance.shape)

        cross_x_resize_enhance=cross_x_resize * attention_weights2.expand_as(cross_x_resize)
            # x_fuse =cross_rgb_resize_enhance+cross_x_resize_enhance
        with torch.no_grad():
            x_fuse=torch.cat([cross_rgb_resize_enhance,cross_x_resize_enhance],dim=1)
            
            
            
#             channel_attn_mamba =  ConcatMambaFusionBlock(
#                     hidden_dim=768,
#                     # hidden_dim=1152,
#                     mlp_ratio=4.0,
#                     d_state=4,
#                 ) 
#             x_fuse = channel_attn_mamba(cross_rgb, cross_x).permute(0, 3, 1, 2).contiguous()
            # x_fuse = x_fuse.permute(0, 3, 1, 2).contiguous()
            # rgb_patch=rgb_patch.reshape(x_fuse.shape[1], -1).T
            # print(rgb_patch.shape)
            # x_fuse = x_fuse.reshape(x_fuse.shape[1], -1).T.cpu()
            # print(x_fuse.shape)
            ########################################################
            x_fuse=x_fuse.cpu()
            
            # x_fuse=torch.cat([x_fuse,rgb_patch],dim=1)
          
            # spatial_attn = SpatialAttention(kernel_size=3)
            # attention_weights_space=spatial_attn(x_fuse)
            # # print(attention_weights_space.shape)
            # x_fuse=x_fuse * attention_weights_space.expand_as(x_fuse)
            #########################################################
            # channel_attention = ChannelAttention(channel=x_fuse.shape[1])
            # attention_weights = channel_attention(x_fuse)
            
            # channel_attention = ChannelAttention(channel=rgb_patch.shape[1])
            
          
            # attention_weights = channel_attention(rgb_patch)
            
            # x_fuse=xyz_patch * attention_weights.expand_as(xyz_patch)
            
            # x_fuse=x_fuse+xyz_patch
            # x_fuse=rgb_patch * attention_weights.expand_as(rgb_patch)
            
            # x_fuse=xyz_patch
            # rgb=cross_rgb_resize_enhance.reshape(x_fuse.shape[1], -1).T
            # print(rgb.shape)
            # xyz=cross_x_resize_enhance.reshape(x_fuse.shape[1], -1).T
            x_fuse=x_fuse.reshape(x_fuse.shape[1], -1).T
            fusion=torch.cat([x_fuse,fused_output_rgb],dim=1)
            # fusion=fused_output_rgb
            
            # print(fusion.shape)
            
            
            
        self.compute_s_s_map(fusion, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx,
                             nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        torch.cuda.empty_cache()

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz,
                        center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        # print(self.patch_lib.shape)
        # print(self.patch_lib)
        self.mean = torch.mean(self.patch_lib)
        self.std = torch.std(self.patch_lib)
        self.patch_lib = (self.patch_lib - self.mean)/self.std
        # print(self.patch_lib.shape)
        # print(self.patch_lib)
        # self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps)
            self.patch_lib = self.patch_lib[self.coreset_idx]
