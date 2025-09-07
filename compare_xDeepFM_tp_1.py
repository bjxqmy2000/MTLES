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


# QoS数据集类，用于加载和处理数据
class QoSDataset:
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

    def get_user_info(self, userlist_df):
        """获取并编码用户特征"""
        # 将user_id转为int类型
        userlist_df["user_id"] = userlist_df["user_id"].astype(int)

        # 使用LabelEncoder对分类特征进行编码
        other_features = ["ip_address", "country", "ip_number", "as", "latitude", "longitude"]
        for feat in other_features:
            lbe = LabelEncoder()
            userlist_df[feat] = lbe.fit_transform(userlist_df[feat])

        # 获取各特征的最大编码值+1（即特征的类别数量）
        max_values = userlist_df[["user_id", "ip_address", "country", "ip_number", "as", "latitude", "longitude"]].max()
        self.user_id_num = max_values["user_id"] + 1
        self.user_ip_address_num = max_values["ip_address"] + 1
        self.user_country_num = max_values["country"] + 1
        self.user_ip_number_num = max_values["ip_number"] + 1
        self.user_as_num = max_values["as"] + 1
        self.user_latitude_num = max_values["latitude"] + 1
        self.user_longitude_num = max_values["longitude"] + 1

        # 构建用户信息字典
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
        # 将ws_id转为int类型
        wslist_df["ws_id"] = wslist_df["ws_id"].astype(int)

        # 使用LabelEncoder对分类特征进行编码
        other_features = ["wsdl_address", "provider", "ip_address", "country", "ip_number", "as", "latitude",
                          "longitude"]
        for feat in other_features:
            lbe = LabelEncoder()
            wslist_df[feat] = lbe.fit_transform(wslist_df[feat])

        # 获取各特征的最大编码值+1（即特征的类别数量）
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

        # 构建服务信息字典
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
        # 转换为numpy数组
        rtMatrix_array = rtMatrix_df.values

        # 获取非零元素的索引和值
        rows, cols = np.where(rtMatrix_array != 0)
        rts = rtMatrix_array[rows, cols]

        # 构建响应时间字典
        rt_info = {}
        for i, j, rt in zip(rows, cols, rts):
            if i not in rt_info:
                rt_info[i] = {}
            rt_info[i][j] = float(rt)

        logger.info("rt_info获取完毕，非零元素数量：{}".format(len(rts)))
        return rt_info

    def get_tp_info(self, tpMatrix_df):
        """处理吞吐量矩阵"""
        # 转换为numpy数组
        tpMatrix_array = tpMatrix_df.values

        # 获取非零元素的索引和值
        rows, cols = np.where(tpMatrix_array != 0)
        tps = tpMatrix_array[rows, cols]

        # 构建吞吐量字典
        tp_info = {}
        for i, j, tp in zip(rows, cols, tps):
            if i not in tp_info:
                tp_info[i] = {}
            tp_info[i][j] = float(tp)

        logger.info("tp_info获取完毕，非零元素数量：{}".format(len(tps)))
        return tp_info

    def build_rt_task_dataset(self):
        """构建RT单任务数据集并计算归一化参数"""
        # 构建训练数据集并计算归一化参数
        self.train_dataset, self.rt_mean, self.rt_std = self.get_rt_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            rt_info=self.train_rt_info,
            compute_stats=True
        )

        # 构建测试数据集，使用训练集的归一化参数
        self.valid_dataset = self.get_rt_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            rt_info=self.test_rt_info,
            compute_stats=False,
            rt_mean=self.rt_mean,
            rt_std=self.rt_std
        )

        print(f"RT-only task dataset statistics:")
        print(f"train dataset size: {len(self.train_dataset)}")
        print(f"valid dataset size: {len(self.valid_dataset)}")

    def build_tp_task_dataset(self):
        """构建TP单任务数据集并计算归一化参数"""
        # 构建训练数据集并计算归一化参数
        self.train_dataset, self.tp_mean, self.tp_std = self.get_tp_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            tp_info=self.train_tp_info,
            compute_stats=True
        )

        # 构建测试数据集，使用训练集的归一化参数
        self.valid_dataset = self.get_tp_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            tp_info=self.test_tp_info,
            compute_stats=False,
            tp_mean=self.tp_mean,
            tp_std=self.tp_std
        )

        print(f"TP-only task dataset statistics:")
        print(f"train dataset size: {len(self.train_dataset)}")
        print(f"valid dataset size: {len(self.valid_dataset)}")

    def get_rt_dataset(self, user_info, ws_info, rt_info, compute_stats=True,
                       rt_mean=None, rt_std=None):
        """构建RT单任务数据集并进行归一化处理"""
        # 如果需要计算统计量
        if compute_stats:
            all_rts = []
            # 收集所有有效的RT值
            for user_id in rt_info.keys():
                for ws_id in rt_info[user_id].keys():
                    all_rts.append(rt_info[user_id][ws_id])

            # 计算均值和标准差用于归一化
            rt_mean, rt_std = np.mean(all_rts), np.std(all_rts)
            logger.info(f"RT归一化参数 - 均值: {rt_mean:.4f}, 标准差: {rt_std:.4f}")

        # 构建数据集
        dataset = []
        # 处理所有用户
        for user_id in rt_info.keys():
            # 获取该用户的RT数据
            user_rts = rt_info[user_id]

            for ws_id in user_rts.keys():
                # 准备数据项
                item = {
                    'user_info': user_info[user_id],
                    'ws_info': ws_info[ws_id]
                }

                # 处理RT
                rt_raw = user_rts[ws_id]
                normalized_rt = (rt_raw - rt_mean) / (rt_std + 1e-8)
                item['rt'] = normalized_rt
                item['rt_raw'] = rt_raw

                # 添加到数据集
                dataset.append(item)

        if compute_stats:
            return dataset, rt_mean, rt_std
        else:
            return dataset

    def get_tp_dataset(self, user_info, ws_info, tp_info, compute_stats=True,
                       tp_mean=None, tp_std=None):
        """构建TP单任务数据集并进行归一化处理"""
        # 如果需要计算统计量
        if compute_stats:
            all_tps = []
            # 收集所有有效的TP值
            for user_id in tp_info.keys():
                for ws_id in tp_info[user_id].keys():
                    all_tps.append(tp_info[user_id][ws_id])

            # 计算均值和标准差用于归一化
            tp_mean, tp_std = np.mean(all_tps), np.std(all_tps)
            logger.info(f"TP归一化参数 - 均值: {tp_mean:.4f}, 标准差: {tp_std:.4f}")

        # 构建数据集
        dataset = []
        # 处理所有用户
        for user_id in tp_info.keys():
            # 获取该用户的TP数据
            user_tps = tp_info[user_id]

            for ws_id in user_tps.keys():
                # 准备数据项
                item = {
                    'user_info': user_info[user_id],
                    'ws_info': ws_info[ws_id]
                }

                # 处理TP
                tp_raw = user_tps[ws_id]
                normalized_tp = (tp_raw - tp_mean) / (tp_std + 1e-8)
                item['tp'] = normalized_tp
                item['tp_raw'] = tp_raw

                # 添加到数据集
                dataset.append(item)

        if compute_stats:
            return dataset, tp_mean, tp_std
        else:
            return dataset


class QoSRtTorchDataset(Dataset):
    """RT单任务PyTorch数据集类，用于加载QoS响应时间数据"""

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
            'rt_raw': torch.tensor(item['rt_raw'], dtype=torch.float)
        }


class QoSTpTorchDataset(Dataset):
    """TP单任务PyTorch数据集类，用于加载QoS吞吐量数据"""

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
            'tp': torch.tensor(item['tp'], dtype=torch.float),
            'tp_raw': torch.tensor(item['tp_raw'], dtype=torch.float)
        }


# 特征嵌入层，用于单任务嵌入，共享嵌入
class SharedEmbedding(nn.Module):
    """特征嵌入层，同时为Linear组件、CIN组件和Deep组件提供共享嵌入"""

    def __init__(self, user_feature_info, ws_feature_info, embedding_dim=128):
        super().__init__()

        # 用户特征嵌入层
        self.user_id_embed = nn.Embedding(user_feature_info['user_id_num'], embedding_dim)
        self.user_ip_address_embed = nn.Embedding(user_feature_info['user_ip_address_num'], embedding_dim)
        self.user_country_embed = nn.Embedding(user_feature_info['user_country_num'], embedding_dim)
        self.user_ip_number_embed = nn.Embedding(user_feature_info['user_ip_number_num'], embedding_dim)
        self.user_as_embed = nn.Embedding(user_feature_info['user_as_num'], embedding_dim)
        self.user_latitude_embed = nn.Embedding(user_feature_info['user_latitude_num'], embedding_dim)
        self.user_longitude_embed = nn.Embedding(user_feature_info['user_longitude_num'], embedding_dim)

        # 服务特征嵌入层
        self.ws_id_embed = nn.Embedding(ws_feature_info['ws_id_num'], embedding_dim)
        self.ws_wsdl_address_embed = nn.Embedding(ws_feature_info['ws_wsdl_address_num'], embedding_dim)
        self.ws_provider_embed = nn.Embedding(ws_feature_info['ws_provider_num'], embedding_dim)
        self.ws_ip_address_embed = nn.Embedding(ws_feature_info['ws_ip_address_num'], embedding_dim)
        self.ws_country_embed = nn.Embedding(ws_feature_info['ws_country_num'], embedding_dim)
        self.ws_ip_number_embed = nn.Embedding(ws_feature_info['ws_ip_number_num'], embedding_dim)
        self.ws_as_embed = nn.Embedding(ws_feature_info['ws_as_num'], embedding_dim)
        self.ws_latitude_embed = nn.Embedding(ws_feature_info['ws_latitude_num'], embedding_dim)
        self.ws_longitude_embed = nn.Embedding(ws_feature_info['ws_longitude_num'], embedding_dim)

        # 特征数量和嵌入维度
        self.num_features = 16  # 7个用户特征 + 9个服务特征
        self.embedding_dim = embedding_dim

    def forward(self, user_features, ws_features):
        # 获取用户特征嵌入
        user_id_emb = self.user_id_embed(user_features[:, 0].long())
        user_ip_address_emb = self.user_ip_address_embed(user_features[:, 1].long())
        user_country_emb = self.user_country_embed(user_features[:, 2].long())
        user_ip_number_emb = self.user_ip_number_embed(user_features[:, 3].long())
        user_as_emb = self.user_as_embed(user_features[:, 4].long())
        user_latitude_emb = self.user_latitude_embed(user_features[:, 5].long())
        user_longitude_emb = self.user_longitude_embed(user_features[:, 6].long())

        # 获取服务特征嵌入
        ws_id_emb = self.ws_id_embed(ws_features[:, 0].long())
        ws_wsdl_address_emb = self.ws_wsdl_address_embed(ws_features[:, 1].long())
        ws_provider_emb = self.ws_provider_embed(ws_features[:, 2].long())
        ws_ip_address_emb = self.ws_ip_address_embed(ws_features[:, 3].long())
        ws_country_emb = self.ws_country_embed(ws_features[:, 4].long())
        ws_ip_number_emb = self.ws_ip_number_embed(ws_features[:, 5].long())
        ws_as_emb = self.ws_as_embed(ws_features[:, 6].long())
        ws_latitude_emb = self.ws_latitude_embed(ws_features[:, 7].long())
        ws_longitude_emb = self.ws_longitude_embed(ws_features[:, 8].long())

        # 收集特征嵌入为列表
        feature_embeddings = [
            user_id_emb, user_ip_address_emb, user_country_emb, user_ip_number_emb,
            user_as_emb, user_latitude_emb, user_longitude_emb,
            ws_id_emb, ws_wsdl_address_emb, ws_provider_emb, ws_ip_address_emb,
            ws_country_emb, ws_ip_number_emb, ws_as_emb, ws_latitude_emb, ws_longitude_emb
        ]

        # 堆叠特征嵌入为张量 [batch_size, num_features, embedding_dim]
        features_stacked = torch.stack(feature_embeddings, dim=1)

        return features_stacked


# CIN组件 (Compressed Interaction Network) - 修复版本
class CINLayer(nn.Module):
    """压缩交互网络组件，实现高阶特征交互"""

    def __init__(self, field_num, cin_size):
        super().__init__()
        self.field_num = [field_num] + cin_size  # 每层的特征个数(包括第0层)

        # 创建权重矩阵
        self.cin_W = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.field_num[0] * self.field_num[i],
                out_channels=self.field_num[i + 1],
                kernel_size=1
            )
            for i in range(len(self.field_num) - 1)
        ])

    def forward(self, embedded_features):
        """
        计算CIN部分的输出
        Args:
            embedded_features: [batch_size, field_num[0], embedding_dim]
        Returns:
            CIN输出: [batch_size, sum(field_num[1:])]
        """
        batch_size = embedded_features.shape[0]
        embedding_dim = embedded_features.shape[2]

        # 初始化结果列表
        res_list = []
        X0 = embedded_features  # [batch_size, field_num[0], embedding_dim]
        X_i = X0  # 当前层的输入

        for i in range(len(self.field_num) - 1):
            # 计算外积并重组
            # X0: [batch_size, field_num[0], embedding_dim]
            # X_i: [batch_size, field_num[i], embedding_dim]

            # 扩展维度以便计算外积
            # X0_expanded: [batch_size, field_num[0], 1, embedding_dim]
            # X_i_expanded: [batch_size, 1, field_num[i], embedding_dim]
            X0_expanded = X0.unsqueeze(2)
            X_i_expanded = X_i.unsqueeze(1)

            # 计算外积
            # outer_product: [batch_size, field_num[0], field_num[i], embedding_dim]
            outer_product = X0_expanded * X_i_expanded

            # 重塑为适合卷积的形状
            # Z: [batch_size, field_num[0] * field_num[i], embedding_dim]
            Z = outer_product.view(batch_size, self.field_num[0] * self.field_num[i], embedding_dim)

            # 使用1D卷积进行变换
            # 输入: [batch_size, field_num[0] * field_num[i], embedding_dim]
            # 输出: [batch_size, field_num[i+1], embedding_dim]
            X_i = self.cin_W[i](Z)

            # 将当前层的结果添加到列表中
            res_list.append(X_i)

        # 合并所有层的结果
        # 首先将每层的结果在嵌入维度上求和，移除嵌入维度
        res = torch.cat([x.sum(2) for x in res_list], dim=1)

        return res


# Deep组件
class DeepLayer(nn.Module):
    """深度组件，用于学习高阶特征交互"""

    def __init__(self, input_dim, deep_hidden_dims=[256, 128, 64]):
        super().__init__()

        # 创建深度网络层
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in deep_hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            prev_dim = hidden_dim

    def forward(self, x):
        """
        计算Deep部分的输出
        Args:
            x: [batch_size, input_dim]
        Returns:
            深度网络输出: [batch_size, deep_hidden_dims[-1]]
        """
        for layer in self.layers:
            x = layer(x)
        return x


# xDeepFM模型
class xDeepFM(nn.Module):
    """xDeepFM模型，结合Linear、CIN和深度神经网络"""

    def __init__(self, user_feature_info, ws_feature_info, embedding_dim=128,
                 cin_size=[128, 128], deep_hidden_dims=[256, 128, 64],
                 prediction_hidden_dims=[128, 64]):
        super().__init__()

        # 共享嵌入层
        self.embedding = SharedEmbedding(
            user_feature_info,
            ws_feature_info,
            embedding_dim
        )

        # 计算维度
        self.num_features = 16  # 7个用户特征 + 9个服务特征
        self.embedding_dim = embedding_dim
        self.input_dim = self.num_features * embedding_dim

        # CIN组件
        self.cin = CINLayer(self.num_features, cin_size)

        # Deep组件
        self.deep = DeepLayer(self.input_dim, deep_hidden_dims)

        # 计算cin输出维度（所有cin层的节点数之和）
        cin_output_dim = sum(cin_size)

        # 预测头 - 使用prediction_hidden_dims构建多层预测网络
        self.prediction_layers = nn.ModuleList()
        prev_dim = deep_hidden_dims[-1] + cin_output_dim  # CIN输出维度为cin_output_dim，Deep输出维度为deep_hidden_dims[-1]

        for hidden_dim in prediction_hidden_dims:
            self.prediction_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.prediction_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 最终输出层
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, user_features, ws_features):
        """
        xDeepFM前向传播
        Args:
            user_features: 用户特征 [batch_size, user_feature_dim]
            ws_features: 服务特征 [batch_size, ws_feature_dim]
        Returns:
            预测结果: [batch_size, 1]
        """
        # 获取特征嵌入
        embedded_features = self.embedding(user_features, ws_features)  # [batch_size, num_features, embedding_dim]
        batch_size = embedded_features.size(0)

        # CIN部分
        cin_output = self.cin(embedded_features)  # [batch_size, sum(cin_size)]

        # Deep部分
        deep_input = embedded_features.view(batch_size, -1)  # [batch_size, num_features * embedding_dim]
        deep_output = self.deep(deep_input)  # [batch_size, deep_hidden_dims[-1]]

        # 合并Linear、CIN和Deep的输出
        combined = torch.cat([cin_output, deep_output], dim=1)  # [batch_size, sum(cin_size) + deep_hidden_dims[-1]]

        # 通过预测头
        x = combined
        for layer in self.prediction_layers:
            x = layer(x)

        # 最终输出
        output = self.output_layer(x)

        return output.squeeze()


# 用于RT任务的xDeepFM模型训练
def train_xdeepfm_rt_model(model, train_loader, valid_loader, criterion,
                           optimizer, scheduler, num_epochs, device,
                           early_stopping_patience=10, rt_mean=0, rt_std=1, eval_epochs=10):
    """训练RT任务的xDeepFM模型"""
    logger = logging.getLogger(__name__)
    logger.info("开始训练RT任务的xDeepFM模型")

    best_metrics = {
        "rt_mae": float('inf'),
        "rt_rmse": float('inf')
    }
    no_improvement_epochs = 0
    best_epoch = 0

    # 记录总训练时间和每轮时间
    total_train_time = 0
    epoch_times = []

    for epoch in range(num_epochs):
        # 记录开始时间
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            rt_targets = batch['rt'].to(device)

            optimizer.zero_grad()

            # 前向传播
            rt_pred = model(user_features, ws_features)

            # 计算损失
            loss = criterion(rt_pred, rt_targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0

        # 计算本轮训练时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        total_train_time += epoch_time

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"训练损失: {avg_train_loss:.4f}, "
            f"训练时间: {epoch_time:.2f}秒"
        )

        # 验证阶段
        if (epoch + 1) % eval_epochs == 0:  # 每k个epoch进行一次验证
            # 记录验证开始时间
            valid_start_time = time.time()

            valid_metrics = evaluate_xdeepfm_rt_model(
                model, valid_loader, criterion, device, rt_mean, rt_std
            )

            # 计算验证时间
            valid_time = time.time() - valid_start_time

            valid_loss = valid_metrics['loss']

            logger.info(
                f"验证损失: {valid_loss:.4f}, "
                f"RT MAE: {valid_metrics['rt_mae']:.4f}, RT RMSE: {valid_metrics['rt_rmse']:.4f}, "
                f"验证时间: {valid_time:.2f}秒"
            )

            avg_improvement = -1

            # 判断是否为首次评估
            if math.isinf(best_metrics['rt_mae']):
                is_better = True
            else:
                # 计算平均相对提升
                rt_mae_improvement = (valid_metrics['rt_mae'] - best_metrics['rt_mae']) / best_metrics['rt_mae']
                rt_rmse_improvement = (valid_metrics['rt_rmse'] - best_metrics['rt_rmse']) / best_metrics['rt_rmse']

                # 计算平均相对提升率
                avg_improvement = (rt_mae_improvement + rt_rmse_improvement) / 2

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
        f"RT MAE: {best_metrics['rt_mae']:.4f}, RT RMSE: {best_metrics['rt_rmse']:.4f}"
    )

    return best_metrics


# 用于RT任务的xDeepFM模型评估
def evaluate_xdeepfm_rt_model(model, valid_loader, criterion, device, rt_mean, rt_std):
    """评估RT任务的xDeepFM模型性能"""
    model.eval()
    valid_loss = 0.0
    batch_count = 0

    rt_preds = []
    rt_targets_list = []

    with torch.no_grad():
        for batch in valid_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            rt_targets = batch['rt'].to(device)

            # 前向传播
            rt_pred = model(user_features, ws_features)

            # 计算损失
            loss = criterion(rt_pred, rt_targets)
            valid_loss += loss.item()
            batch_count += 1

            # 收集预测和目标
            rt_preds.append(rt_pred)
            rt_targets_list.append(rt_targets)

    # 合并所有批次的预测和目标
    rt_preds = torch.cat(rt_preds)
    rt_targets = torch.cat(rt_targets_list)

    # 反归一化
    rt_preds_denorm = rt_preds * rt_std + rt_mean
    rt_targets_denorm = rt_targets * rt_std + rt_mean

    # 计算指标
    rt_mae = torch.mean(torch.abs(rt_preds_denorm - rt_targets_denorm)).item()
    rt_mse = torch.mean((rt_preds_denorm - rt_targets_denorm) ** 2).item()
    rt_rmse = torch.sqrt(torch.tensor(rt_mse)).item()

    # 计算平均损失
    avg_valid_loss = valid_loss / batch_count if batch_count > 0 else 0

    return {
        'loss': avg_valid_loss,
        'rt_mae': rt_mae,
        'rt_rmse': rt_rmse
    }


# 用于TP任务的xDeepFM模型训练
def train_xdeepfm_tp_model(model, train_loader, valid_loader, criterion,
                           optimizer, scheduler, num_epochs, device,
                           early_stopping_patience=10, tp_mean=0, tp_std=1, eval_epochs=10):
    """训练TP任务的xDeepFM模型"""
    logger = logging.getLogger(__name__)
    logger.info("开始训练TP任务的xDeepFM模型")

    best_metrics = {
        "tp_mae": float('inf'),
        "tp_rmse": float('inf')
    }
    no_improvement_epochs = 0
    best_epoch = 0

    # 记录总训练时间和每轮时间
    total_train_time = 0
    epoch_times = []

    for epoch in range(num_epochs):
        # 记录开始时间
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            tp_targets = batch['tp'].to(device)

            optimizer.zero_grad()

            # 前向传播
            tp_pred = model(user_features, ws_features)

            # 计算损失
            loss = criterion(tp_pred, tp_targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0

        # 计算本轮训练时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        total_train_time += epoch_time

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"训练损失: {avg_train_loss:.4f}, "
            f"训练时间: {epoch_time:.2f}秒"
        )

        # 验证阶段
        if (epoch + 1) % eval_epochs == 0:  # 每k个epoch进行一次验证
            # 记录验证开始时间
            valid_start_time = time.time()

            valid_metrics = evaluate_xdeepfm_tp_model(
                model, valid_loader, criterion, device, tp_mean, tp_std
            )

            # 计算验证时间
            valid_time = time.time() - valid_start_time

            valid_loss = valid_metrics['loss']

            logger.info(
                f"验证损失: {valid_loss:.4f}, "
                f"TP MAE: {valid_metrics['tp_mae']:.4f}, TP RMSE: {valid_metrics['tp_rmse']:.4f}, "
                f"验证时间: {valid_time:.2f}秒"
            )

            avg_improvement = -1

            # 判断是否为首次评估
            if math.isinf(best_metrics['tp_mae']):
                is_better = True
            else:
                # 计算平均相对提升
                tp_mae_improvement = (valid_metrics['tp_mae'] - best_metrics['tp_mae']) / best_metrics['tp_mae']
                tp_rmse_improvement = (valid_metrics['tp_rmse'] - best_metrics['tp_rmse']) / best_metrics['tp_rmse']

                # 计算平均相对提升率
                avg_improvement = (tp_mae_improvement + tp_rmse_improvement) / 2

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
        f"TP MAE: {best_metrics['tp_mae']:.4f}, TP RMSE: {best_metrics['tp_rmse']:.4f}"
    )

    return best_metrics


# 用于TP任务的xDeepFM模型评估
def evaluate_xdeepfm_tp_model(model, valid_loader, criterion, device, tp_mean, tp_std):
    """评估TP任务的xDeepFM模型性能"""
    model.eval()
    valid_loss = 0.0
    batch_count = 0

    tp_preds = []
    tp_targets_list = []

    with torch.no_grad():
        for batch in valid_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            tp_targets = batch['tp'].to(device)

            # 前向传播
            tp_pred = model(user_features, ws_features)

            # 计算损失
            loss = criterion(tp_pred, tp_targets)
            valid_loss += loss.item()
            batch_count += 1

            # 收集预测和目标
            tp_preds.append(tp_pred)
            tp_targets_list.append(tp_targets)

    # 合并所有批次的预测和目标
    tp_preds = torch.cat(tp_preds)
    tp_targets = torch.cat(tp_targets_list)

    # 反归一化
    tp_preds_denorm = tp_preds * tp_std + tp_mean
    tp_targets_denorm = tp_targets * tp_std + tp_mean

    # 计算指标
    tp_mae = torch.mean(torch.abs(tp_preds_denorm - tp_targets_denorm)).item()
    tp_mse = torch.mean((tp_preds_denorm - tp_targets_denorm) ** 2).item()
    tp_rmse = torch.sqrt(torch.tensor(tp_mse)).item()

    # 计算平均损失
    avg_valid_loss = valid_loss / batch_count if batch_count > 0 else 0

    return {
        'loss': avg_valid_loss,
        'tp_mae': tp_mae,
        'tp_rmse': tp_rmse
    }


def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)  # Python的random模块
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU单卡
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU多卡

    # 为了完全的确定性，可以添加以下设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"已设置随机种子: {seed}")


def main():
    """主函数"""
    # 参数设置
    parser = argparse.ArgumentParser(description='xDeepFM模型用于QoS预测')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/QoS_Dataset/')
    parser.add_argument('--userlist_path', type=str, default='data/QoS_Dataset/userlist.csv')
    parser.add_argument('--wslist_path', type=str, default='data/QoS_Dataset/wslist.csv')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--density', type=int, default=1)

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument("--eval_epochs", type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--cin_size', type=str, default='16,16')
    parser.add_argument('--deep_hidden_dims', type=str, default='512,256')
    parser.add_argument('--prediction_hidden_dims', type=str, default='128,64')

    # 任务模式
    parser.add_argument('--task', type=str, default='tp_only', choices=['rt_only', 'tp_only'])

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 解析维度参数
    cin_size = [int(x) for x in args.cin_size.split(',')]
    deep_hidden_dims = [int(x) for x in args.deep_hidden_dims.split(',')]
    prediction_hidden_dims = [int(x) for x in args.prediction_hidden_dims.split(',')]

    # 检查GPU可用性
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info(f"正在加载数据，密度: {args.density}...")
    qos_dataset = QoSDataset(
        density=args.density,
        userlist_path=args.userlist_path,
        wslist_path=args.wslist_path,
        data_dir=args.data_dir
    )

    # 特征维度计算
    user_feature_dim = 7  # user_id, ip_address, country, ip_number, as, latitude, longitude
    ws_feature_dim = 9  # ws_id, wsdl_address, provider, ip_address, country, ip_number, as, latitude, longitude

    # 特征信息字典
    user_feature_info = {
        'user_id_num': qos_dataset.user_id_num,
        'user_ip_address_num': qos_dataset.user_ip_address_num,
        'user_country_num': qos_dataset.user_country_num,
        'user_ip_number_num': qos_dataset.user_ip_number_num,
        'user_as_num': qos_dataset.user_as_num,
        'user_latitude_num': qos_dataset.user_latitude_num,
        'user_longitude_num': qos_dataset.user_longitude_num,
    }

    ws_feature_info = {
        'ws_id_num': qos_dataset.ws_id_num,
        'ws_wsdl_address_num': qos_dataset.ws_wsdl_address_num,
        'ws_provider_num': qos_dataset.ws_provider_num,
        'ws_ip_address_num': qos_dataset.ws_ip_address_num,
        'ws_country_num': qos_dataset.ws_country_num,
        'ws_ip_number_num': qos_dataset.ws_ip_number_num,
        'ws_as_num': qos_dataset.ws_as_num,
        'ws_latitude_num': qos_dataset.ws_latitude_num,
        'ws_longitude_num': qos_dataset.ws_longitude_num,
    }

    # 根据任务类型创建并训练模型
    if args.task == 'rt_only':
        # 构建RT单任务数据集
        qos_dataset.build_rt_task_dataset()

        # 创建PyTorch数据集
        train_torch_dataset = QoSRtTorchDataset(qos_dataset.train_dataset, user_feature_dim, ws_feature_dim)
        valid_torch_dataset = QoSRtTorchDataset(qos_dataset.valid_dataset, user_feature_dim, ws_feature_dim)

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

        logger.info("===== xDeepFM RT单任务模型配置 =====")
        logger.info(f"嵌入维度: {args.embedding_dim}")
        logger.info(f"CIN层大小: {cin_size}")
        logger.info(f"深度网络隐藏层维度: {deep_hidden_dims}")
        logger.info(f"预测头隐藏层维度: {prediction_hidden_dims}")
        logger.info("===================")

        # 创建xDeepFM模型
        model = xDeepFM(
            user_feature_info=user_feature_info,
            ws_feature_info=ws_feature_info,
            embedding_dim=args.embedding_dim,
            cin_size=cin_size,
            deep_hidden_dims=deep_hidden_dims,
            prediction_hidden_dims=prediction_hidden_dims
        ).to(device)

        # 统计模型参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"可训练参数数量: {trainable_params}")

        # 定义损失函数
        criterion = nn.SmoothL1Loss()

        # 创建优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=1e-5
        )

        # 训练xDeepFM模型
        best_metrics = train_xdeepfm_rt_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            device=device,
            early_stopping_patience=args.early_stopping_patience,
            rt_mean=qos_dataset.rt_mean,
            rt_std=qos_dataset.rt_std,
            eval_epochs=args.eval_epochs
        )

    elif args.task == 'tp_only':
        # 构建TP单任务数据集
        qos_dataset.build_tp_task_dataset()

        # 创建PyTorch数据集
        train_torch_dataset = QoSTpTorchDataset(qos_dataset.train_dataset, user_feature_dim, ws_feature_dim)
        valid_torch_dataset = QoSTpTorchDataset(qos_dataset.valid_dataset, user_feature_dim, ws_feature_dim)

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

        logger.info("===== xDeepFM TP单任务模型配置 =====")
        logger.info(f"嵌入维度: {args.embedding_dim}")
        logger.info(f"CIN层大小: {cin_size}")
        logger.info(f"深度网络隐藏层维度: {deep_hidden_dims}")
        logger.info(f"预测头隐藏层维度: {prediction_hidden_dims}")
        logger.info("===================")

        # 创建xDeepFM模型
        model = xDeepFM(
            user_feature_info=user_feature_info,
            ws_feature_info=ws_feature_info,
            embedding_dim=args.embedding_dim,
            cin_size=cin_size,
            deep_hidden_dims=deep_hidden_dims,
            prediction_hidden_dims=prediction_hidden_dims
        ).to(device)

        # 统计模型参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"可训练参数数量: {trainable_params}")

        # 定义损失函数
        criterion = nn.SmoothL1Loss()

        # 创建优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=1e-5
        )

        # 训练xDeepFM模型
        best_metrics = train_xdeepfm_tp_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            device=device,
            early_stopping_patience=args.early_stopping_patience,
            tp_mean=qos_dataset.tp_mean,
            tp_std=qos_dataset.tp_std,
            eval_epochs=args.eval_epochs
        )

    # 返回最佳指标
    return best_metrics


if __name__ == '__main__':
    main()