import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import logging
import time
import math
import pandas as pd

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Dataset:
    """数据集类，处理RT和TP数据"""

    def __init__(self, density, userlist_path, wslist_path, data_dir):
        # 记录用户有效特征的编码数量
        self.user_id_num = 0
        self.user_ip_address_num = 0
        self.user_country_num = 0
        self.user_ip_number_num = 0
        self.user_as_num = 0
        self.user_latitude_num = 0
        self.user_longitude_num = 0

        # 记录服务有效特征的编码数量
        self.ws_id_num = 0
        self.ws_wsdl_address_num = 0
        self.ws_provider_num = 0
        self.ws_ip_address_num = 0
        self.ws_country_num = 0
        self.ws_ip_number_num = 0
        self.ws_as_num = 0
        self.ws_latitude_num = 0
        self.ws_longitude_num = 0

        # 文件路径
        self.userlist_path = userlist_path
        self.wslist_path = wslist_path
        self.data_dir = data_dir

        # 设置RT和TP矩阵路径
        rtMatrix_train_path = os.path.join(data_dir, f"sparse/rtMatrix{density}.csv")
        rtMatrix_test_path = os.path.join(data_dir, f"sparse/rtMatrix{100 - density}.csv")
        tpMatrix_train_path = os.path.join(data_dir, f"sparse/tpMatrix{density}.csv")
        tpMatrix_test_path = os.path.join(data_dir, f"sparse/tpMatrix{100 - density}.csv")

        # 读取用户和服务特征
        userlist_df = pd.read_csv(userlist_path)
        wslist_df = pd.read_csv(wslist_path)

        # 读取训练和测试矩阵
        rtMatrix_train_df = pd.read_csv(rtMatrix_train_path, header=None)
        rtMatrix_test_df = pd.read_csv(rtMatrix_test_path, header=None)
        tpMatrix_train_df = pd.read_csv(tpMatrix_train_path, header=None)
        tpMatrix_test_df = pd.read_csv(tpMatrix_test_path, header=None)

        logger.info("文件预读取完毕")

        # 获取用户和服务信息
        self.user_info = self.get_user_info(userlist_df)
        self.ws_info = self.get_ws_info(wslist_df)

        # 获取矩阵数据
        self.train_rt_info = self.get_rt_info(rtMatrix_train_df)
        self.test_rt_info = self.get_rt_info(rtMatrix_test_df)
        self.train_tp_info = self.get_tp_info(tpMatrix_train_df)
        self.test_tp_info = self.get_tp_info(tpMatrix_test_df)

        # 构建数据集
        self.build_dataset()

    def get_user_info(self, userlist_df):
        """获取并编码用户特征"""
        userlist_df["user_id"] = userlist_df["user_id"].astype(int)

        other_features = ["ip_address", "country", "ip_number", "as", "latitude", "longitude"]
        for feat in other_features:
            lbe = LabelEncoder()
            userlist_df[feat] = lbe.fit_transform(userlist_df[feat])

        max_values = userlist_df[["user_id", "ip_address", "country", "ip_number", "as", "latitude", "longitude"]].max()
        self.user_id_num = max_values["user_id"] + 1
        self.user_ip_address_num = max_values["ip_address"] + 1
        self.user_country_num = max_values["country"] + 1
        self.user_ip_number_num = max_values["ip_number"] + 1
        self.user_as_num = max_values["as"] + 1
        self.user_latitude_num = max_values["latitude"] + 1
        self.user_longitude_num = max_values["longitude"] + 1

        user_info = {
            row["user_id"]: {
                'user_id': row["user_id"],
                'ip_address': row["ip_address"],
                'country': row["country"],
                'ip_number': row["ip_number"],
                'as': row["as"],
                'latitude': row["latitude"],
                'longitude': row["longitude"]
            }
            for _, row in userlist_df.iterrows()
        }

        logger.info("user_info获取完毕")
        return user_info

    def get_ws_info(self, wslist_df):
        """获取并编码Web服务特征"""
        wslist_df["ws_id"] = wslist_df["ws_id"].astype(int)

        other_features = ["wsdl_address", "provider", "ip_address", "country", "ip_number", "as", "latitude",
                          "longitude"]
        for feat in other_features:
            lbe = LabelEncoder()
            wslist_df[feat] = lbe.fit_transform(wslist_df[feat])

        max_values = wslist_df[
            ["ws_id", "wsdl_address", "provider", "ip_address", "country", "ip_number", "as", "latitude",
             "longitude"]].max()
        self.ws_id_num = max_values["ws_id"] + 1
        self.ws_wsdl_address_num = max_values["wsdl_address"] + 1
        self.ws_provider_num = max_values["provider"] + 1
        self.ws_ip_address_num = max_values["ip_address"] + 1
        self.ws_country_num = max_values["country"] + 1
        self.ws_ip_number_num = max_values["ip_number"] + 1
        self.ws_as_num = max_values["as"] + 1
        self.ws_latitude_num = max_values["latitude"] + 1
        self.ws_longitude_num = max_values["longitude"] + 1

        ws_info = {
            row["ws_id"]: {
                'ws_id': row["ws_id"],
                'wsdl_address': row["wsdl_address"],
                'provider': row["provider"],
                'ip_address': row["ip_address"],
                'country': row["country"],
                'ip_number': row["ip_number"],
                'as': row["as"],
                'latitude': row["latitude"],
                'longitude': row["longitude"]
            }
            for _, row in wslist_df.iterrows()
        }

        logger.info("ws_info获取完毕")
        return ws_info

    def get_rt_info(self, rtMatrix_df):
        """处理响应时间矩阵"""
        rtMatrix_array = rtMatrix_df.values
        rows, cols = np.where(rtMatrix_array != 0)
        rts = rtMatrix_array[rows, cols]

        rt_info = {}
        for i, j, rt in zip(rows, cols, rts):
            if i not in rt_info:
                rt_info[i] = {}
            rt_info[i][j] = float(rt)

        logger.info("rt_info获取完毕，非零元素数量：{}".format(len(rts)))
        return rt_info

    def get_tp_info(self, tpMatrix_df):
        """处理吞吐量矩阵"""
        tpMatrix_array = tpMatrix_df.values
        rows, cols = np.where(tpMatrix_array != 0)
        tps = tpMatrix_array[rows, cols]

        tp_info = {}
        for i, j, tp in zip(rows, cols, tps):
            if i not in tp_info:
                tp_info[i] = {}
            tp_info[i][j] = float(tp)

        logger.info("tp_info获取完毕，非零元素数量：{}".format(len(tps)))
        return tp_info

    def build_dataset(self):
        """构建数据集"""
        # 构建训练数据集
        self.train_dataset, self.rt_mean, self.rt_std, self.tp_mean, self.tp_std = self.get_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            rt_info=self.train_rt_info,
            tp_info=self.train_tp_info,
            compute_stats=True
        )

        # 构建测试数据集
        self.valid_dataset = self.get_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            rt_info=self.test_rt_info,
            tp_info=self.test_tp_info,
            compute_stats=False,
            rt_mean=self.rt_mean,
            rt_std=self.rt_std,
            tp_mean=self.tp_mean,
            tp_std=self.tp_std
        )

        logger.info(f"dataset statistics:")
        logger.info(f"train dataset size: {len(self.train_dataset)}")
        logger.info(f"valid dataset size: {len(self.valid_dataset)}")

    def get_dataset(self, user_info, ws_info, rt_info, tp_info, compute_stats=True,
                    rt_mean=None, rt_std=None, tp_mean=None, tp_std=None):
        """构建数据集，包含同时有RT和TP的样本"""

        # 找到同时有RT和TP数据的(user_id, ws_id)对
        common_pairs = []
        for user_id in rt_info.keys():
            if user_id in tp_info:
                rt_services = set(rt_info[user_id].keys())
                tp_services = set(tp_info[user_id].keys())
                # 只保留同时存在于RT和TP中的服务
                common_services = rt_services & tp_services
                for ws_id in common_services:
                    common_pairs.append((user_id, ws_id))

        logger.info(f"找到同时包含RT和TP的样本对数量: {len(common_pairs)}")

        # 如果需要计算统计量
        if compute_stats:
            all_rts = []
            all_tps = []

            for user_id, ws_id in common_pairs:
                all_rts.append(rt_info[user_id][ws_id])
                all_tps.append(tp_info[user_id][ws_id])

            rt_mean, rt_std = np.mean(all_rts), np.std(all_rts)
            tp_mean, tp_std = np.mean(all_tps), np.std(all_tps)

            logger.info(f"RT归一化参数 - 均值: {rt_mean:.4f}, 标准差: {rt_std:.4f}")
            logger.info(f"TP归一化参数 - 均值: {tp_mean:.4f}, 标准差: {tp_std:.4f}")

        # 构建数据集
        dataset = []
        for user_id, ws_id in common_pairs:
            # 获取RT和TP值
            rt_raw = rt_info[user_id][ws_id]
            tp_raw = tp_info[user_id][ws_id]

            # 归一化
            normalized_rt = (rt_raw - rt_mean) / (rt_std + 1e-8)
            normalized_tp = (tp_raw - tp_mean) / (tp_std + 1e-8)

            item = {
                'user_info': user_info[user_id],
                'ws_info': ws_info[ws_id],
                'rt': normalized_rt,
                'tp': normalized_tp,
                'rt_raw': rt_raw,
                'tp_raw': tp_raw
            }

            dataset.append(item)

        if compute_stats:
            return dataset, rt_mean, rt_std, tp_mean, tp_std
        else:
            return dataset


class TorchDataset(torch.utils.data.Dataset):
    """PyTorch数据集类"""

    def __init__(self, dataset, user_info_dim, ws_info_dim):
        self.dataset = dataset
        self.user_info_dim = user_info_dim
        self.ws_info_dim = ws_info_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 将用户信息转换为张量
        user_info = torch.zeros(self.user_info_dim)
        user_info[0] = item['user_info']['user_id']
        user_info[1] = item['user_info']['ip_address']
        user_info[2] = item['user_info']['country']
        user_info[3] = item['user_info']['ip_number']
        user_info[4] = item['user_info']['as']
        user_info[5] = item['user_info']['latitude']
        user_info[6] = item['user_info']['longitude']

        # 将服务信息转换为张量
        ws_info = torch.zeros(self.ws_info_dim)
        ws_info[0] = item['ws_info']['ws_id']
        ws_info[1] = item['ws_info']['wsdl_address']
        ws_info[2] = item['ws_info']['provider']
        ws_info[3] = item['ws_info']['ip_address']
        ws_info[4] = item['ws_info']['country']
        ws_info[5] = item['ws_info']['ip_number']
        ws_info[6] = item['ws_info']['as']
        ws_info[7] = item['ws_info']['latitude']
        ws_info[8] = item['ws_info']['longitude']

        return {
            'user_info': user_info,
            'ws_info': ws_info,
            'rt': torch.tensor(item['rt'], dtype=torch.float),
            'tp': torch.tensor(item['tp'], dtype=torch.float),
            'rt_raw': torch.tensor(item['rt_raw'], dtype=torch.float),
            'tp_raw': torch.tensor(item['tp_raw'], dtype=torch.float)
        }


class MLP_Block(nn.Module):
    """多层感知机模块，仅包含线性层和ReLU激活函数"""

    def __init__(self, input_dim, hidden_units=None, output_dim=None):
        super(MLP_Block, self).__init__()

        if not hidden_units:
            hidden_units = []

        layers = []

        # 隐藏层
        hidden_units = [input_dim] + hidden_units
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(nn.ReLU())

        # 输出层
        if output_dim is not None:
            layers.append(nn.Linear(hidden_units[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class STEM_Layer(nn.Module):
    """STEM层实现"""

    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim,
                 expert_hidden_units, gate_hidden_units):
        super(STEM_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks

        # 任务特定专家
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                MLP_Block(
                    input_dim=input_dim,
                    hidden_units=expert_hidden_units,
                ) for _ in range(self.num_specific_experts)
            ]) for _ in range(num_tasks)
        ])

        # 共享专家
        self.shared_experts = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                hidden_units=expert_hidden_units,
            ) for _ in range(self.num_shared_experts)
        ])

        # 门控网络
        self.gate = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                output_dim=num_specific_experts * num_tasks + num_shared_experts,
                hidden_units=gate_hidden_units,
            ) for i in range(self.num_tasks + 1)
        ])

        self.gate_activation = nn.Softmax(dim=-1)

    # forward 方法保持不变
    def forward(self, x, return_gate=False):
        """
        前向传播
        x: list, len(x)==num_tasks+1
        """
        specific_expert_outputs = []
        shared_expert_outputs = []

        # 任务特定专家的输出
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](x[i]))
            specific_expert_outputs.append(task_expert_outputs)

        # 共享专家的输出
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))

        # 门控机制
        stem_outputs = []
        stem_gates = []

        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                # 对任务特定专家
                gate_input = []
                for j in range(self.num_tasks):
                    if j == i:
                        gate_input.extend(specific_expert_outputs[j])
                    else:
                        specific_expert_outputs_j = specific_expert_outputs[j]
                        # 对其他任务的专家使用stop_gradient操作
                        specific_expert_outputs_j = [out.detach() for out in specific_expert_outputs_j]
                        gate_input.extend(specific_expert_outputs_j)

                gate_input.extend(shared_expert_outputs)
                gate_input = torch.stack(gate_input,
                                         dim=1)  # (batch_size, num_specific_experts*num_tasks+num_shared_experts, dim)
                gate = self.gate_activation(
                    self.gate[i](x[i] + x[-1]))  # (batch_size, num_specific_experts*num_tasks+num_shared_experts)

                if return_gate:
                    specific_gate = gate[:, :self.num_specific_experts * self.num_tasks].mean(0)
                    task_gate = torch.chunk(specific_gate, chunks=self.num_tasks)
                    specific_gate_list = []
                    for tg in task_gate:
                        specific_gate_list.append(torch.sum(tg))
                    shared_gate = gate[:, -self.num_shared_experts:].mean(0).sum()
                    target_task_gate = torch.stack(specific_gate_list + [shared_gate], dim=0).view(-1)  # (num_task+1,1)
                    assert len(target_task_gate) == self.num_tasks + 1
                    stem_gates.append(target_task_gate)

                stem_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)  # (batch_size, dim)
                stem_outputs.append(stem_output)
            else:
                # 对共享专家
                gate_input = []
                for j in range(self.num_tasks):
                    gate_input.extend(specific_expert_outputs[j])
                gate_input.extend(shared_expert_outputs)
                gate_input = torch.stack(gate_input,
                                         dim=1)  # (batch_size, num_specific_experts*num_tasks+num_shared_experts, dim)
                gate = self.gate_activation(
                    self.gate[i](x[-1]))  # (batch_size, num_specific_experts*num_tasks+num_shared_experts)
                stem_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)  # (batch_size, dim)
                stem_outputs.append(stem_output)

        if return_gate:
            return stem_outputs, stem_gates
        else:
            return stem_outputs


class STEMModel(nn.Module):
    """STEM模型实现：Shared and Task-specific EMbeddings"""

    def __init__(self, user_feature_info, ws_feature_info, embedding_dim=16,
                 num_layers=1, num_shared_experts=1, num_specific_experts=1,
                 expert_hidden_units=[512, 256, 128], gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64]):
        super(STEMModel, self).__init__()

        self.num_tasks = 2  # RT和TP两个任务
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 用户特征嵌入 - 共享嵌入
        self.user_shared_embeddings = nn.ModuleDict({
            'user_id': nn.Embedding(user_feature_info['user_id_num'], embedding_dim),
            'ip_address': nn.Embedding(user_feature_info['user_ip_address_num'], embedding_dim),
            'country': nn.Embedding(user_feature_info['user_country_num'], embedding_dim),
            'ip_number': nn.Embedding(user_feature_info['user_ip_number_num'], embedding_dim),
            'as': nn.Embedding(user_feature_info['user_as_num'], embedding_dim),
            'latitude': nn.Embedding(user_feature_info['user_latitude_num'], embedding_dim),
            'longitude': nn.Embedding(user_feature_info['user_longitude_num'], embedding_dim)
        })

        # 服务特征嵌入 - 共享嵌入
        self.ws_shared_embeddings = nn.ModuleDict({
            'ws_id': nn.Embedding(ws_feature_info['ws_id_num'], embedding_dim),
            'wsdl_address': nn.Embedding(ws_feature_info['ws_wsdl_address_num'], embedding_dim),
            'provider': nn.Embedding(ws_feature_info['ws_provider_num'], embedding_dim),
            'ip_address': nn.Embedding(ws_feature_info['ws_ip_address_num'], embedding_dim),
            'country': nn.Embedding(ws_feature_info['ws_country_num'], embedding_dim),
            'ip_number': nn.Embedding(ws_feature_info['ws_ip_number_num'], embedding_dim),
            'as': nn.Embedding(ws_feature_info['ws_as_num'], embedding_dim),
            'latitude': nn.Embedding(ws_feature_info['ws_latitude_num'], embedding_dim),
            'longitude': nn.Embedding(ws_feature_info['ws_longitude_num'], embedding_dim)
        })

        # 用户特征嵌入 - 任务特定嵌入 (RT和TP两个任务)
        self.user_task_embeddings = nn.ModuleList([
            # RT任务嵌入
            nn.ModuleDict({
                'user_id': nn.Embedding(user_feature_info['user_id_num'], embedding_dim),
                'ip_address': nn.Embedding(user_feature_info['user_ip_address_num'], embedding_dim),
                'country': nn.Embedding(user_feature_info['user_country_num'], embedding_dim),
                'ip_number': nn.Embedding(user_feature_info['user_ip_number_num'], embedding_dim),
                'as': nn.Embedding(user_feature_info['user_as_num'], embedding_dim),
                'latitude': nn.Embedding(user_feature_info['user_latitude_num'], embedding_dim),
                'longitude': nn.Embedding(user_feature_info['user_longitude_num'], embedding_dim)
            }),
            # TP任务嵌入
            nn.ModuleDict({
                'user_id': nn.Embedding(user_feature_info['user_id_num'], embedding_dim),
                'ip_address': nn.Embedding(user_feature_info['user_ip_address_num'], embedding_dim),
                'country': nn.Embedding(user_feature_info['user_country_num'], embedding_dim),
                'ip_number': nn.Embedding(user_feature_info['user_ip_number_num'], embedding_dim),
                'as': nn.Embedding(user_feature_info['user_as_num'], embedding_dim),
                'latitude': nn.Embedding(user_feature_info['user_latitude_num'], embedding_dim),
                'longitude': nn.Embedding(user_feature_info['user_longitude_num'], embedding_dim)
            })
        ])

        # 服务特征嵌入 - 任务特定嵌入 (RT和TP两个任务)
        self.ws_task_embeddings = nn.ModuleList([
            # RT任务嵌入
            nn.ModuleDict({
                'ws_id': nn.Embedding(ws_feature_info['ws_id_num'], embedding_dim),
                'wsdl_address': nn.Embedding(ws_feature_info['ws_wsdl_address_num'], embedding_dim),
                'provider': nn.Embedding(ws_feature_info['ws_provider_num'], embedding_dim),
                'ip_address': nn.Embedding(ws_feature_info['ws_ip_address_num'], embedding_dim),
                'country': nn.Embedding(ws_feature_info['ws_country_num'], embedding_dim),
                'ip_number': nn.Embedding(ws_feature_info['ws_ip_number_num'], embedding_dim),
                'as': nn.Embedding(ws_feature_info['ws_as_num'], embedding_dim),
                'latitude': nn.Embedding(ws_feature_info['ws_latitude_num'], embedding_dim),
                'longitude': nn.Embedding(ws_feature_info['ws_longitude_num'], embedding_dim)
            }),
            # TP任务嵌入
            nn.ModuleDict({
                'ws_id': nn.Embedding(ws_feature_info['ws_id_num'], embedding_dim),
                'wsdl_address': nn.Embedding(ws_feature_info['ws_wsdl_address_num'], embedding_dim),
                'provider': nn.Embedding(ws_feature_info['ws_provider_num'], embedding_dim),
                'ip_address': nn.Embedding(ws_feature_info['ws_ip_address_num'], embedding_dim),
                'country': nn.Embedding(ws_feature_info['ws_country_num'], embedding_dim),
                'ip_number': nn.Embedding(ws_feature_info['ws_ip_number_num'], embedding_dim),
                'as': nn.Embedding(ws_feature_info['ws_as_num'], embedding_dim),
                'latitude': nn.Embedding(ws_feature_info['ws_latitude_num'], embedding_dim),
                'longitude': nn.Embedding(ws_feature_info['ws_longitude_num'], embedding_dim)
            })
        ])

        # 计算嵌入后的特征维度 (7个用户特征 + 9个服务特征) * embedding_dim
        self.input_dim = (7 + 9) * embedding_dim

        # STEM层
        self.stem_layers = nn.ModuleList([
            STEM_Layer(
                num_shared_experts=num_shared_experts,
                num_specific_experts=num_specific_experts,
                num_tasks=self.num_tasks,
                input_dim=self.input_dim if i == 0 else expert_hidden_units[-1],
                expert_hidden_units=expert_hidden_units,
                gate_hidden_units=gate_hidden_units,
            ) for i in range(num_layers)
        ])

        # 任务塔
        self.towers = nn.ModuleList([
            MLP_Block(
                input_dim=expert_hidden_units[-1],
                output_dim=1,
                hidden_units=tower_hidden_units,
            ) for _ in range(self.num_tasks)
        ])

    # get_embeddings 和 forward 方法保持不变
    def get_embeddings(self, user_features, ws_features):
        """获取共享和任务特定的嵌入表示"""
        batch_size = user_features.size(0)

        # 共享嵌入
        shared_embeddings = []

        # 用户特征共享嵌入
        user_id_emb = self.user_shared_embeddings['user_id'](user_features[:, 0].long())
        user_ip_address_emb = self.user_shared_embeddings['ip_address'](user_features[:, 1].long())
        user_country_emb = self.user_shared_embeddings['country'](user_features[:, 2].long())
        user_ip_number_emb = self.user_shared_embeddings['ip_number'](user_features[:, 3].long())
        user_as_emb = self.user_shared_embeddings['as'](user_features[:, 4].long())
        user_latitude_emb = self.user_shared_embeddings['latitude'](user_features[:, 5].long())
        user_longitude_emb = self.user_shared_embeddings['longitude'](user_features[:, 6].long())

        # 服务特征共享嵌入
        ws_id_emb = self.ws_shared_embeddings['ws_id'](ws_features[:, 0].long())
        ws_wsdl_address_emb = self.ws_shared_embeddings['wsdl_address'](ws_features[:, 1].long())
        ws_provider_emb = self.ws_shared_embeddings['provider'](ws_features[:, 2].long())
        ws_ip_address_emb = self.ws_shared_embeddings['ip_address'](ws_features[:, 3].long())
        ws_country_emb = self.ws_shared_embeddings['country'](ws_features[:, 4].long())
        ws_ip_number_emb = self.ws_shared_embeddings['ip_number'](ws_features[:, 5].long())
        ws_as_emb = self.ws_shared_embeddings['as'](ws_features[:, 6].long())
        ws_latitude_emb = self.ws_shared_embeddings['latitude'](ws_features[:, 7].long())
        ws_longitude_emb = self.ws_shared_embeddings['longitude'](ws_features[:, 8].long())

        # 拼接共享嵌入
        shared_emb = torch.cat([
            user_id_emb, user_ip_address_emb, user_country_emb, user_ip_number_emb,
            user_as_emb, user_latitude_emb, user_longitude_emb,
            ws_id_emb, ws_wsdl_address_emb, ws_provider_emb, ws_ip_address_emb,
            ws_country_emb, ws_ip_number_emb, ws_as_emb, ws_latitude_emb, ws_longitude_emb
        ], dim=1)

        # 任务特定嵌入
        task_embeddings = []

        for task_idx in range(self.num_tasks):
            # 用户任务特定嵌入
            user_id_emb_task = self.user_task_embeddings[task_idx]['user_id'](user_features[:, 0].long())
            user_ip_address_emb_task = self.user_task_embeddings[task_idx]['ip_address'](user_features[:, 1].long())
            user_country_emb_task = self.user_task_embeddings[task_idx]['country'](user_features[:, 2].long())
            user_ip_number_emb_task = self.user_task_embeddings[task_idx]['ip_number'](user_features[:, 3].long())
            user_as_emb_task = self.user_task_embeddings[task_idx]['as'](user_features[:, 4].long())
            user_latitude_emb_task = self.user_task_embeddings[task_idx]['latitude'](user_features[:, 5].long())
            user_longitude_emb_task = self.user_task_embeddings[task_idx]['longitude'](user_features[:, 6].long())

            # 服务任务特定嵌入
            ws_id_emb_task = self.ws_task_embeddings[task_idx]['ws_id'](ws_features[:, 0].long())
            ws_wsdl_address_emb_task = self.ws_task_embeddings[task_idx]['wsdl_address'](ws_features[:, 1].long())
            ws_provider_emb_task = self.ws_task_embeddings[task_idx]['provider'](ws_features[:, 2].long())
            ws_ip_address_emb_task = self.ws_task_embeddings[task_idx]['ip_address'](ws_features[:, 3].long())
            ws_country_emb_task = self.ws_task_embeddings[task_idx]['country'](ws_features[:, 4].long())
            ws_ip_number_emb_task = self.ws_task_embeddings[task_idx]['ip_number'](ws_features[:, 5].long())
            ws_as_emb_task = self.ws_task_embeddings[task_idx]['as'](ws_features[:, 6].long())
            ws_latitude_emb_task = self.ws_task_embeddings[task_idx]['latitude'](ws_features[:, 7].long())
            ws_longitude_emb_task = self.ws_task_embeddings[task_idx]['longitude'](ws_features[:, 8].long())

            # 拼接任务特定嵌入
            task_emb = torch.cat([
                user_id_emb_task, user_ip_address_emb_task, user_country_emb_task, user_ip_number_emb_task,
                user_as_emb_task, user_latitude_emb_task, user_longitude_emb_task,
                ws_id_emb_task, ws_wsdl_address_emb_task, ws_provider_emb_task, ws_ip_address_emb_task,
                ws_country_emb_task, ws_ip_number_emb_task, ws_as_emb_task, ws_latitude_emb_task, ws_longitude_emb_task
            ], dim=1)

            task_embeddings.append(task_emb)

        # 返回任务特定嵌入和共享嵌入
        return task_embeddings, shared_emb

    def forward(self, user_features, ws_features):
        """前向传播"""
        # 获取任务特定嵌入和共享嵌入
        task_embeddings, shared_emb = self.get_embeddings(user_features, ws_features)

        # 合并所有嵌入作为STEM层的输入
        stem_inputs = task_embeddings + [shared_emb]  # [rt_emb, tp_emb, shared_emb]

        # 通过STEM层
        for i in range(self.num_layers):
            stem_outputs = self.stem_layers[i](stem_inputs)
            stem_inputs = stem_outputs

        # 任务塔预测
        rt_pred = self.towers[0](stem_outputs[0])  # RT任务预测
        tp_pred = self.towers[1](stem_outputs[1])  # TP任务预测

        return rt_pred.squeeze(), tp_pred.squeeze()


def train_stem_model(model, train_loader, valid_loader, optimizer, scheduler, device,
                     num_epochs, eval_epochs=10, early_stopping_patience=10,
                     rt_mean=0, rt_std=1, tp_mean=0, tp_std=1):
    """训练STEM模型"""
    logger.info("开始训练STEM模型")

    criterion = nn.SmoothL1Loss()

    best_metrics = {
        "rt_mae": float('inf'),
        "rt_rmse": float('inf'),
        "tp_mae": float('inf'),
        "tp_rmse": float('inf'),
    }
    no_improvement_epochs = 0
    best_epoch = 0

    total_train_time = 0
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        rt_losses = 0.0
        tp_losses = 0.0
        batch_count = 0

        for batch in train_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            rt_targets = batch['rt'].to(device)
            tp_targets = batch['tp'].to(device)

            optimizer.zero_grad()

            # 前向传播
            rt_pred, tp_pred = model(user_features, ws_features)

            # 计算单独的损失
            rt_loss = criterion(rt_pred, rt_targets)
            tp_loss = criterion(tp_pred, tp_targets)

            # 直接相加所有任务损失
            total_loss = rt_loss + tp_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            rt_losses += rt_loss.item()
            tp_losses += tp_loss.item()
            batch_count += 1

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_rt_loss = rt_losses / batch_count if batch_count > 0 else 0
        avg_tp_loss = tp_losses / batch_count if batch_count > 0 else 0

        # 计算本轮训练时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        total_train_time += epoch_time

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"训练损失: {avg_train_loss:.4f}, "
            f"RT损失: {avg_rt_loss:.4f}, "
            f"TP损失: {avg_tp_loss:.4f}, "
            f"训练时间: {epoch_time:.2f}秒"
        )

        # 验证阶段
        if (epoch + 1) % eval_epochs == 0:
            valid_start_time = time.time()

            valid_metrics = evaluate_stem_model(
                model, valid_loader, device, rt_mean, rt_std, tp_mean, tp_std
            )

            valid_time = time.time() - valid_start_time

            logger.info(
                f"验证损失: {valid_metrics['total_loss']:.4f}, "
                f"RT MAE: {valid_metrics['rt_mae']:.4f}, RT RMSE: {valid_metrics['rt_rmse']:.4f}, "
                f"TP MAE: {valid_metrics['tp_mae']:.4f}, TP RMSE: {valid_metrics['tp_rmse']:.4f}, "
                f"验证时间: {valid_time:.2f}秒"
            )

            # 判断是否为更好的模型（参考原始代码的评估方式）
            avg_improvement = -1

            # 判断是否为首次评估
            if math.isinf(best_metrics['rt_mae']) or math.isinf(best_metrics['tp_mae']):
                is_better = True
            else:
                # 计算平均相对提升
                rt_mae_improvement = (valid_metrics['rt_mae'] - best_metrics['rt_mae']) / best_metrics['rt_mae']
                rt_rmse_improvement = (valid_metrics['rt_rmse'] - best_metrics['rt_rmse']) / best_metrics['rt_rmse']
                tp_mae_improvement = (valid_metrics['tp_mae'] - best_metrics['tp_mae']) / best_metrics['tp_mae']
                tp_rmse_improvement = (valid_metrics['tp_rmse'] - best_metrics['tp_rmse']) / best_metrics['tp_rmse']

                # 计算平均相对提升率
                avg_improvement = (
                                              rt_mae_improvement + rt_rmse_improvement + tp_mae_improvement + tp_rmse_improvement) / 4

                # 检查是否有改进
                is_better = avg_improvement < 0

            # 使用判断结果
            if is_better:
                best_metrics = valid_metrics
                best_epoch = epoch + 1
                no_improvement_epochs = 0

                logger.info(f"✓ 找到更好的模型! (Epoch {best_epoch}), 指标平均提升: {avg_improvement:.4f}")
            else:
                no_improvement_epochs += 1
                logger.info(f"× 连续 {no_improvement_epochs} 轮评估未改进")

            # 早停检查
            if early_stopping_patience > 0 and no_improvement_epochs >= early_stopping_patience:
                logger.info(f"早停触发，连续 {no_improvement_epochs} 轮无显著提升")
                break

    # 计算平均每轮训练时间
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    logger.info("训练完成")
    logger.info(f"总训练时间: {total_train_time:.2f}秒, 平均每轮训练时间: {avg_epoch_time:.2f}秒")
    logger.info(
        f"最佳验证指标 (Epoch {best_epoch}), "
        f"RT MAE: {best_metrics['rt_mae']:.4f}, RT RMSE: {best_metrics['rt_rmse']:.4f}, "
        f"TP MAE: {best_metrics['tp_mae']:.4f}, TP RMSE: {best_metrics['tp_rmse']:.4f}, "
    )

    return best_metrics


def evaluate_stem_model(model, valid_loader, device, rt_mean, rt_std, tp_mean, tp_std):
    """评估STEM模型性能"""
    model.eval()
    criterion = nn.SmoothL1Loss()

    valid_loss = 0.0
    rt_losses = 0.0
    tp_losses = 0.0
    batch_count = 0

    rt_preds = []
    tp_preds = []
    rt_targets_list = []
    tp_targets_list = []

    with torch.no_grad():
        for batch in valid_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            rt_targets = batch['rt'].to(device)
            tp_targets = batch['tp'].to(device)

            # 前向传播
            rt_pred, tp_pred = model(user_features, ws_features)

            # 计算损失
            rt_loss = criterion(rt_pred, rt_targets)
            tp_loss = criterion(tp_pred, tp_targets)

            # 直接相加所有任务损失
            total_loss = rt_loss + tp_loss

            valid_loss += total_loss.item()
            rt_losses += rt_loss.item()
            tp_losses += tp_loss.item()
            batch_count += 1

            # 收集预测和目标
            rt_preds.append(rt_pred)
            tp_preds.append(tp_pred)
            rt_targets_list.append(rt_targets)
            tp_targets_list.append(tp_targets)

    # 合并所有批次的预测和目标
    rt_preds = torch.cat(rt_preds)
    tp_preds = torch.cat(tp_preds)
    rt_targets = torch.cat(rt_targets_list)
    tp_targets = torch.cat(tp_targets_list)

    # 反归一化
    rt_preds_denorm = rt_preds * rt_std + rt_mean
    tp_preds_denorm = tp_preds * tp_std + tp_mean
    rt_targets_denorm = rt_targets * rt_std + rt_mean
    tp_targets_denorm = tp_targets * tp_std + tp_mean

    # 计算指标
    rt_mae = torch.mean(torch.abs(rt_preds_denorm - rt_targets_denorm)).item()
    rt_mse = torch.mean((rt_preds_denorm - rt_targets_denorm) ** 2).item()
    rt_rmse = torch.sqrt(torch.tensor(rt_mse)).item()

    tp_mae = torch.mean(torch.abs(tp_preds_denorm - tp_targets_denorm)).item()
    tp_mse = torch.mean((tp_preds_denorm - tp_targets_denorm) ** 2).item()
    tp_rmse = torch.sqrt(torch.tensor(tp_mse)).item()

    # 计算平均损失
    avg_valid_loss = valid_loss / batch_count if batch_count > 0 else 0
    avg_rt_loss = rt_losses / batch_count if batch_count > 0 else 0
    avg_tp_loss = tp_losses / batch_count if batch_count > 0 else 0

    return {
        'total_loss': avg_valid_loss,
        'rt_loss': avg_rt_loss,
        'tp_loss': avg_tp_loss,
        'rt_mae': rt_mae,
        'rt_rmse': rt_rmse,
        'tp_mae': tp_mae,
        'tp_rmse': tp_rmse
    }


def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"已设置随机种子: {seed}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='STEM模型用于QoS预测')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/QoS_Dataset/')
    parser.add_argument('--userlist_path', type=str, default='data/QoS_Dataset/userlist.csv')
    parser.add_argument('--wslist_path', type=str, default='data/QoS_Dataset/wslist.csv')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--density', type=int, default=4, help='数据密度百分比')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument("--eval_epochs", type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # STEM模型参数
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_shared_experts', type=int, default=4)
    parser.add_argument('--num_specific_experts', type=int, default=4)
    parser.add_argument('--expert_hidden_dims', type=str, default='512,256')
    parser.add_argument('--gate_hidden_dims', type=str, default='256,128')
    parser.add_argument('--tower_hidden_dims', type=str, default='128,64')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 解析维度参数
    expert_hidden_dims = [int(x) for x in args.expert_hidden_dims.split(',')]
    gate_hidden_dims = [int(x) for x in args.gate_hidden_dims.split(',')]
    tower_hidden_dims = [int(x) for x in args.tower_hidden_dims.split(',')]

    # 检查GPU可用性
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info(f"正在加载STEM数据，密度: {args.density}%...")
    dataset = Dataset(
        density=args.density,
        userlist_path=args.userlist_path,
        wslist_path=args.wslist_path,
        data_dir=args.data_dir
    )

    # 特征维度
    user_feature_dim = 7
    ws_feature_dim = 9

    # 特征信息字典
    user_feature_info = {
        'user_id_num': dataset.user_id_num,
        'user_ip_address_num': dataset.user_ip_address_num,
        'user_country_num': dataset.user_country_num,
        'user_ip_number_num': dataset.user_ip_number_num,
        'user_as_num': dataset.user_as_num,
        'user_latitude_num': dataset.user_latitude_num,
        'user_longitude_num': dataset.user_longitude_num,
    }

    ws_feature_info = {
        'ws_id_num': dataset.ws_id_num,
        'ws_wsdl_address_num': dataset.ws_wsdl_address_num,
        'ws_provider_num': dataset.ws_provider_num,
        'ws_ip_address_num': dataset.ws_ip_address_num,
        'ws_country_num': dataset.ws_country_num,
        'ws_ip_number_num': dataset.ws_ip_number_num,
        'ws_as_num': dataset.ws_as_num,
        'ws_latitude_num': dataset.ws_latitude_num,
        'ws_longitude_num': dataset.ws_longitude_num,
    }

    # 创建PyTorch数据集
    train_torch_dataset = TorchDataset(dataset.train_dataset, user_feature_dim, ws_feature_dim)
    valid_torch_dataset = TorchDataset(dataset.valid_dataset, user_feature_dim, ws_feature_dim)

    # 创建数据加载器
    train_loader = DataLoader(
        train_torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_torch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info("===== STEM模型配置 =====")
    logger.info(f"嵌入维度: {args.embedding_dim}")
    logger.info(f"层数: {args.num_layers}")
    logger.info(f"共享专家数量: {args.num_shared_experts}")
    logger.info(f"任务特定专家数量: {args.num_specific_experts}")
    logger.info(f"专家隐藏层维度: {expert_hidden_dims}")
    logger.info(f"门控隐藏层维度: {gate_hidden_dims}")
    logger.info(f"任务塔隐藏层维度: {tower_hidden_dims}")
    logger.info("========================")

    # 创建STEM模型 - 移除不再使用的参数
    model = STEMModel(
        user_feature_info=user_feature_info,
        ws_feature_info=ws_feature_info,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_shared_experts=args.num_shared_experts,
        num_specific_experts=args.num_specific_experts,
        expert_hidden_units=expert_hidden_dims,
        gate_hidden_units=gate_hidden_dims,
        tower_hidden_units=tower_hidden_dims,
    ).to(device)

    # 统计模型参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数数量: {trainable_params:,}")

    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-5
    )

    # 训练STEM模型
    best_metrics = train_stem_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        eval_epochs=args.eval_epochs,
        early_stopping_patience=args.early_stopping_patience,
        rt_mean=dataset.rt_mean,
        rt_std=dataset.rt_std,
        tp_mean=dataset.tp_mean,
        tp_std=dataset.tp_std
    )

    return best_metrics


if __name__ == '__main__':
    main()