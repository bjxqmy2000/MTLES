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


class WAMTLDataset:
    """WAMTL数据集类，专门处理同时包含RT和TP的样本"""

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

        # 构建WAMTL数据集（只使用同时有RT和TP的样本）
        self.build_wamtl_dataset()

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

    def build_wamtl_dataset(self):
        """构建WAMTL数据集，只使用同时包含RT和TP的样本"""
        # 构建训练数据集
        self.train_dataset, self.rt_mean, self.rt_std, self.tp_mean, self.tp_std = self.get_wamtl_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            rt_info=self.train_rt_info,
            tp_info=self.train_tp_info,
            compute_stats=True
        )

        # 构建测试数据集
        self.valid_dataset = self.get_wamtl_dataset(
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

        logger.info(f"WAMTL dataset statistics:")
        logger.info(f"train dataset size: {len(self.train_dataset)} (只包含同时有RT和TP的样本)")
        logger.info(f"valid dataset size: {len(self.valid_dataset)} (只包含同时有RT和TP的样本)")

    def get_wamtl_dataset(self, user_info, ws_info, rt_info, tp_info, compute_stats=True,
                          rt_mean=None, rt_std=None, tp_mean=None, tp_std=None):
        """构建WAMTL数据集，只包含同时有RT和TP的样本"""

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


class WAMTLTorchDataset(Dataset):
    """WAMTL PyTorch数据集类"""

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


class ExpertNetwork(nn.Module):
    """专家网络"""

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GatingNetwork(nn.Module):
    """门控网络，用于计算专家权重"""

    def __init__(self, input_dim, num_experts, hidden_dims, dropout_rate=0.2):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(current_dim, num_experts))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PredictionHead(nn.Module):
    """预测头网络"""

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        # 最终输出层
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class WAMTLModel(nn.Module):
    """WAMTL模型：Weight Adaptive Multi-Task Learning"""

    def __init__(self, user_feature_info, ws_feature_info, embedding_dim=64,
                 num_experts=4, expert_hidden_dims=[256, 128],
                 gate_hidden_dims=[128, 64], prediction_hidden_dims=[64, 32],
                 dropout_rate=0.2):
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

        # 计算嵌入后的特征维度
        self.input_dim = 16 * embedding_dim  # 7个用户特征 + 9个服务特征
        self.num_experts = num_experts

        # 创建专家网络
        self.experts = nn.ModuleList([
            ExpertNetwork(self.input_dim, expert_hidden_dims, dropout_rate)
            for _ in range(num_experts)
        ])

        # 专家输出维度
        self.expert_output_dim = expert_hidden_dims[-1]

        # RT任务的门控网络
        self.rt_gate = GatingNetwork(
            input_dim=self.input_dim,
            num_experts=num_experts,
            hidden_dims=gate_hidden_dims,
            dropout_rate=dropout_rate
        )

        # TP任务的门控网络
        self.tp_gate = GatingNetwork(
            input_dim=self.input_dim,
            num_experts=num_experts,
            hidden_dims=gate_hidden_dims,
            dropout_rate=dropout_rate
        )

        # RT预测头
        self.rt_prediction = PredictionHead(
            input_dim=self.expert_output_dim,
            hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate
        )

        # TP预测头
        self.tp_prediction = PredictionHead(
            input_dim=self.expert_output_dim,
            hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate
        )

        # 用于动态权重调整的历史学习率
        self.rt_loss_history = []
        self.tp_loss_history = []

    def get_embeddings(self, user_features, ws_features):
        """获取用户和服务的嵌入表示"""
        # 用户特征嵌入
        user_id_emb = self.user_id_embed(user_features[:, 0].long())
        user_ip_address_emb = self.user_ip_address_embed(user_features[:, 1].long())
        user_country_emb = self.user_country_embed(user_features[:, 2].long())
        user_ip_number_emb = self.user_ip_number_embed(user_features[:, 3].long())
        user_as_emb = self.user_as_embed(user_features[:, 4].long())
        user_latitude_emb = self.user_latitude_embed(user_features[:, 5].long())
        user_longitude_emb = self.user_longitude_embed(user_features[:, 6].long())

        # 服务特征嵌入
        ws_id_emb = self.ws_id_embed(ws_features[:, 0].long())
        ws_wsdl_address_emb = self.ws_wsdl_address_embed(ws_features[:, 1].long())
        ws_provider_emb = self.ws_provider_embed(ws_features[:, 2].long())
        ws_ip_address_emb = self.ws_ip_address_embed(ws_features[:, 3].long())
        ws_country_emb = self.ws_country_embed(ws_features[:, 4].long())
        ws_ip_number_emb = self.ws_ip_number_embed(ws_features[:, 5].long())
        ws_as_emb = self.ws_as_embed(ws_features[:, 6].long())
        ws_latitude_emb = self.ws_latitude_embed(ws_features[:, 7].long())
        ws_longitude_emb = self.ws_longitude_embed(ws_features[:, 8].long())

        # 拼接所有特征嵌入
        all_embeddings = torch.cat([
            user_id_emb, user_ip_address_emb, user_country_emb, user_ip_number_emb,
            user_as_emb, user_latitude_emb, user_longitude_emb,
            ws_id_emb, ws_wsdl_address_emb, ws_provider_emb, ws_ip_address_emb,
            ws_country_emb, ws_ip_number_emb, ws_as_emb, ws_latitude_emb, ws_longitude_emb
        ], dim=1)

        return all_embeddings

    def forward(self, user_features, ws_features):
        # 获取嵌入特征
        embeddings = self.get_embeddings(user_features, ws_features)

        # 通过所有专家网络
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(embeddings))

        # 堆叠专家输出 [batch_size, num_experts, expert_output_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 获取门控权重
        rt_gate_weights = self.rt_gate(embeddings)  # [batch_size, num_experts]
        tp_gate_weights = self.tp_gate(embeddings)  # [batch_size, num_experts]

        # 加权专家输出
        rt_weighted = torch.bmm(rt_gate_weights.unsqueeze(1), expert_outputs).squeeze(1)
        tp_weighted = torch.bmm(tp_gate_weights.unsqueeze(1), expert_outputs).squeeze(1)

        # 预测输出
        rt_pred = self.rt_prediction(rt_weighted)
        tp_pred = self.tp_prediction(tp_weighted)

        return rt_pred, tp_pred

    def compute_adaptive_weights(self, rt_loss, tp_loss, temperature=2.0):
        """计算自适应权重"""
        # 记录损失历史
        self.rt_loss_history.append(rt_loss.item())
        self.tp_loss_history.append(tp_loss.item())

        # 保留最近几次的损失记录
        max_history = 10
        if len(self.rt_loss_history) > max_history:
            self.rt_loss_history = self.rt_loss_history[-max_history:]
            self.tp_loss_history = self.tp_loss_history[-max_history:]

        # 如果历史记录不足，使用等权重
        if len(self.rt_loss_history) < 2:
            return 0.5, 0.5

        # 计算学习率（损失变化率）
        rt_rate = abs(self.rt_loss_history[-1] - self.rt_loss_history[-2])
        tp_rate = abs(self.tp_loss_history[-1] - self.tp_loss_history[-2])

        # 避免除零
        rt_rate = max(rt_rate, 1e-8)
        tp_rate = max(tp_rate, 1e-8)

        # 计算自适应权重（学习率高的任务权重低）
        rt_weight = math.exp(-rt_rate / temperature)
        tp_weight = math.exp(-tp_rate / temperature)

        # 归一化权重
        total_weight = rt_weight + tp_weight
        rt_weight = rt_weight / total_weight
        tp_weight = tp_weight / total_weight

        return rt_weight, tp_weight


def train_wamtl_model(model, train_loader, valid_loader, optimizer, scheduler, device,
                      num_epochs, eval_epochs=10, early_stopping_patience=10,
                      rt_mean=0, rt_std=1, tp_mean=0, tp_std=1):
    """训练WAMTL模型"""
    logger.info("开始训练WAMTL模型")

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
            rt_loss = criterion(rt_pred.squeeze(), rt_targets)
            tp_loss = criterion(tp_pred.squeeze(), tp_targets)

            # 计算自适应权重
            rt_weight, tp_weight = model.compute_adaptive_weights(rt_loss, tp_loss)

            # 加权总损失
            total_loss = rt_weight * rt_loss + tp_weight * tp_loss

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

            valid_metrics = evaluate_wamtl_model(
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
                avg_improvement = (rt_mae_improvement + rt_rmse_improvement + tp_mae_improvement + tp_rmse_improvement) / 4

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


def evaluate_wamtl_model(model, valid_loader, device, rt_mean, rt_std, tp_mean, tp_std):
    """评估WAMTL模型性能"""
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
            rt_loss = criterion(rt_pred.squeeze(), rt_targets)
            tp_loss = criterion(tp_pred.squeeze(), tp_targets)

            # 使用等权重计算总损失（评估时）
            total_loss = 0.5 * rt_loss + 0.5 * tp_loss

            valid_loss += total_loss.item()
            rt_losses += rt_loss.item()
            tp_losses += tp_loss.item()
            batch_count += 1

            # 收集预测和目标
            rt_preds.append(rt_pred.squeeze())
            tp_preds.append(tp_pred.squeeze())
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
    parser = argparse.ArgumentParser(description='WAMTL模型用于QoS预测')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/QoS_Dataset/')
    parser.add_argument('--userlist_path', type=str, default='data/QoS_Dataset/userlist.csv')
    parser.add_argument('--wslist_path', type=str, default='data/QoS_Dataset/wslist.csv')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--density', type=int, default=2, help='数据密度百分比')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument("--eval_epochs", type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout_rate', type=float, default=0.3)

    # WAMTL模型参数
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--expert_hidden_dims', type=str, default='512,256')
    parser.add_argument('--gate_hidden_dims', type=str, default='256,128')
    parser.add_argument('--prediction_hidden_dims', type=str, default='128,64')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 解析维度参数
    expert_hidden_dims = [int(x) for x in args.expert_hidden_dims.split(',')]
    gate_hidden_dims = [int(x) for x in args.gate_hidden_dims.split(',')]
    prediction_hidden_dims = [int(x) for x in args.prediction_hidden_dims.split(',')]

    # 检查GPU可用性
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info(f"正在加载WAMTL数据，密度: {args.density}%...")
    wamtl_dataset = WAMTLDataset(
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
        'user_id_num': wamtl_dataset.user_id_num,
        'user_ip_address_num': wamtl_dataset.user_ip_address_num,
        'user_country_num': wamtl_dataset.user_country_num,
        'user_ip_number_num': wamtl_dataset.user_ip_number_num,
        'user_as_num': wamtl_dataset.user_as_num,
        'user_latitude_num': wamtl_dataset.user_latitude_num,
        'user_longitude_num': wamtl_dataset.user_longitude_num,
    }

    ws_feature_info = {
        'ws_id_num': wamtl_dataset.ws_id_num,
        'ws_wsdl_address_num': wamtl_dataset.ws_wsdl_address_num,
        'ws_provider_num': wamtl_dataset.ws_provider_num,
        'ws_ip_address_num': wamtl_dataset.ws_ip_address_num,
        'ws_country_num': wamtl_dataset.ws_country_num,
        'ws_ip_number_num': wamtl_dataset.ws_ip_number_num,
        'ws_as_num': wamtl_dataset.ws_as_num,
        'ws_latitude_num': wamtl_dataset.ws_latitude_num,
        'ws_longitude_num': wamtl_dataset.ws_longitude_num,
    }

    # 创建PyTorch数据集
    train_torch_dataset = WAMTLTorchDataset(wamtl_dataset.train_dataset, user_feature_dim, ws_feature_dim)
    valid_torch_dataset = WAMTLTorchDataset(wamtl_dataset.valid_dataset, user_feature_dim, ws_feature_dim)

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

    logger.info("===== WAMTL模型配置 =====")
    logger.info(f"嵌入维度: {args.embedding_dim}")
    logger.info(f"专家数量: {args.num_experts}")
    logger.info(f"专家隐藏层维度: {expert_hidden_dims}")
    logger.info(f"门控隐藏层维度: {gate_hidden_dims}")
    logger.info(f"预测头隐藏层维度: {prediction_hidden_dims}")
    logger.info("========================")

    # 创建WAMTL模型
    model = WAMTLModel(
        user_feature_info=user_feature_info,
        ws_feature_info=ws_feature_info,
        embedding_dim=args.embedding_dim,
        num_experts=args.num_experts,
        expert_hidden_dims=expert_hidden_dims,
        gate_hidden_dims=gate_hidden_dims,
        prediction_hidden_dims=prediction_hidden_dims,
        dropout_rate=args.dropout_rate
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

    # 训练WAMTL模型
    best_metrics = train_wamtl_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        eval_epochs=args.eval_epochs,
        early_stopping_patience=args.early_stopping_patience,
        rt_mean=wamtl_dataset.rt_mean,
        rt_std=wamtl_dataset.rt_std,
        tp_mean=wamtl_dataset.tp_mean,
        tp_std=wamtl_dataset.tp_std
    )

    return best_metrics


if __name__ == '__main__':
    main()