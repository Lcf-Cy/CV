import torch
from torch import nn

from feature_extractors.features import Features
from utils.mvtec3d_util import *
import numpy as np
import math
import os
from sklearn.neighbors import NearestNeighbors
# 多头注意力机制
FUSION_BLOCK=True
class SelfAttention(nn.Module):
    def __init__(self, input_dim, head_dim, num_heads):
        super(SelfAttention, self).__init__()
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

class KNNFusionFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()#转为numpy数据类型
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                   unorganized_pc_no_zeros.contiguous())
        # 构建rgb块特征以及块特征对应的索引 #######################################################
        #点云快特征
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        #rgb块特征
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        #rgb_patch(784,768)个patch
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))  # rgb_patch_size=28
        # 交换重塑(768,56,56)
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        # (3136,768)
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T
        rgb_patch2_indices = torch.arange(rgb_patch2.shape[0])
        #####################################################################################


        #####xyz_patch#######################################################################
        #构造0张量（1，interpolated_pc.shape[1]，224*224）
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        #将interpolated_pc的值赋给xyz_patch_full中指定的非零索引位置，以更新点云数据的部分。
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        #resize(_,3136)
        #self.average = torch.nn.AvgPool2d(3, stride=1)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))#56*56
        #(3136,_)
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        #点云快特征索引
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
        knn_model.fit(xyz_patch)
        # 寻找每个块的 k 近邻
        distances, indices = knn_model.kneighbors(xyz_patch)
        # distances 为每个块与其 k 近邻块的距离列表，indices 为每个块的 k 近邻块的索引列表
        # 去除每个块自身作为最近邻的项
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        # # 提取 k 近邻特征(3136,k,_)
        knn_features_xyz = xyz_patch[indices]

        ############################rgb_patch和xyz_patch进行1-to-k拼接############################################################
        adjusted_knn_features_xyz = np.expand_dims(knn_features_xyz, axis=2)
        adjusted_rgb_patch = np.expand_dims(rgb_patch2, axis=1)#（3136，1，768）
        adjusted_rgb_patch1 = np.expand_dims(adjusted_rgb_patch, axis=1)
        resized_rgb_patch1_array = np.resize(adjusted_rgb_patch1, adjusted_knn_features_xyz.shape)
        concatenated_features_rgb_k_xyz = np.concatenate((resized_rgb_patch1_array, adjusted_knn_features_xyz), axis=2)

        # concatenated_features_rgb_k_xyz(3136,k,2,768)
        ############################xyz_patch和rgb_patch进行1-to-k拼接############################################################
        adjusted_knn_features_rgb = np.expand_dims(knn_features_rgb, axis=2)
        adjusted_xyz_patch = np.expand_dims(xyz_patch, axis=1)#（3136，1，768）
        adjusted_xyz_patch1 = np.expand_dims(adjusted_xyz_patch, axis=1)
        resized_xyz_patch1_array = np.resize(adjusted_xyz_patch1, adjusted_knn_features_rgb.shape)
        concatenated_features_xyz_k_rgb = np.concatenate((resized_xyz_patch1_array, adjusted_knn_features_rgb), axis=2)

        #concatenated_features_xyz_k_rgb(3136,k,2,768)
        ########################################################################################################################
        #通道拼接rgb----k----xyz(3136,1,768)-->
        x0=knn_features_xyz.shape[0]
        x1 = knn_features_xyz.shape[1]
        x2 = knn_features_xyz.shape[2]
        adjusted_rgb_patch_k=np.resize(adjusted_rgb_patch, (x0,x1,x2))
        concatenated_rgb_k = np.concatenate((adjusted_rgb_patch_k, knn_features_xyz), axis=2)
        #(3136,k,_)
        r0 = knn_features_rgb.shape[0]
        r1 = knn_features_rgb.shape[1]
        r2 = knn_features_rgb.shape[2]
        adjusted_xyz_patch_k = np.resize(adjusted_xyz_patch, (r0,r1,r2))
        concatenated_xyz_k = np.concatenate((adjusted_xyz_patch_k, knn_features_rgb), axis=2)
        ########################自注意力机制代码################################################################################################
        self_attention1 = SelfAttention(input_dim=concatenated_rgb_k.shape[2], head_dim=64, num_heads=12)
        output_rgb = self_attention1(concatenated_rgb_k)
        fused_output_rgb = torch.mean(output_rgb, dim=1)

        self_attention2 = SelfAttention(input_dim=concatenated_xyz_k.shape[2], head_dim=64, num_heads=12)
        output_xyz = self_attention2(concatenated_xyz_k)
        fused_output_xyz = torch.mean(output_xyz, dim=1)

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), fused_output_rgb.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, fused_output_rgb], dim=1)

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_lib.append(fusion_patch)
###################################################################################################################################
    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(xyz_patch.unsqueeze(0), rgb_patch2.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([xyz_patch, rgb_patch2], dim=1)

        self.compute_s_s_map(fusion_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx,
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

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps)
            self.patch_lib = self.patch_lib[self.coreset_idx]














