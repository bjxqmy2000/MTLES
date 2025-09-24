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


class MultiTaskEmbedding(nn.Module):
    """多任务嵌入层，用于嵌入用户和服务特征，生成RT、TP和共享三组嵌入"""

    def __init__(self, user_feature_info, ws_feature_info, embedding_dim=128, use_attention=True, num_heads=4, dropout_rate=0.2):
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

        self.use_attention = use_attention

        if use_attention:
            # 添加自注意力模块
            self.rt_attention_block = SelfAttentionBlock(embedding_dim, num_heads, dropout_rate)
            self.tp_attention_block = SelfAttentionBlock(embedding_dim, num_heads, dropout_rate)
            self.shared_attention_block = SelfAttentionBlock(embedding_dim, num_heads, dropout_rate)

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
        batch_size = features_stacked.size(0)

        # 不使用自注意力机制
        if not self.use_attention:
            # 展平并返回特征
            features_flat = features_stacked.reshape(batch_size, -1)  # [batch_size, num_features * embedding_dim]
            return features_flat, features_flat, features_flat

        # 通过自注意力块处理三组特征
        rt_attention_output = self.rt_attention_block(features_stacked)
        tp_attention_output = self.tp_attention_block(features_stacked)
        shared_attention_output = self.shared_attention_block(features_stacked)

        # 展平每组特征
        batch_size = rt_attention_output.size(0)
        rt_flat = rt_attention_output.reshape(batch_size, -1)  # [batch_size, num_features * embedding_dim]
        tp_flat = tp_attention_output.reshape(batch_size, -1)  # [batch_size, num_features * embedding_dim]
        shared_flat = shared_attention_output.reshape(batch_size, -1)  # [batch_size, num_features * embedding_dim]

        return rt_flat, tp_flat, shared_flat


class SingleTaskEmbedding(nn.Module):
    """单任务嵌入层，用于嵌入用户和服务特征，仅生成一组嵌入"""

    def __init__(self, user_feature_info, ws_feature_info, embedding_dim=128, num_heads=4, dropout_rate=0.2):
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

        # 添加自注意力模块
        self.attention_block = SelfAttentionBlock(embedding_dim, num_heads, dropout_rate)

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

        # 通过自注意力块处理特征
        attention_output = self.attention_block(features_stacked)

        # 展平特征
        batch_size = attention_output.size(0)
        flat_features = attention_output.reshape(batch_size, -1)  # [batch_size, num_features * embedding_dim]

        return flat_features


class SelfAttentionBlock(nn.Module):
    """自注意力块 - 结合自注意力机制和前馈神经网络的Transformer块"""

    def __init__(self, embedding_dim, num_heads=4, dropout_rate=0.2):
        super().__init__()

        # 层归一化，用于注意力前的处理
        self.norm1 = nn.LayerNorm(embedding_dim)

        # 多头自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # 层归一化，用于前馈网络前的处理
        self.norm2 = nn.LayerNorm(embedding_dim)

        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, x):
        """
        输入形状: [batch_size, seq_len, embedding_dim]
        输出形状: [batch_size, seq_len * embedding_dim]
        """
        # 应用自注意力机制（带残差连接）
        attn_norm_output = self.norm1(x)
        attn_output, _ = self.self_attention(
            attn_norm_output, attn_norm_output, attn_norm_output
        )
        attn_output = x + attn_output

        # 应用前馈神经网络（带残差连接）
        ffn_norm_output = self.norm2(attn_output)
        ffn_output = self.ffn(ffn_norm_output)
        output = attn_output + ffn_output

        batch_size = output.size(0)
        return output.reshape(batch_size, -1)  # [batch_size, seq_len * embedding_dim]

class ExpertNetwork(nn.Module):
    """专家网络 - 深度神经网络"""

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(current_dim))
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GateNetwork(nn.Module):
    """门控网络 - 用于分配专家权重"""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(current_dim))
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # 输出层，生成专家权重
        layers.append(nn.Linear(current_dim, output_dim))
        # Softmax确保权重和为1
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PredictionHead(nn.Module):
    """预测头网络 - 用于预测任务的最终输出"""

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(current_dim))
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # 最终输出层，预测单一值
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AFESMultiModel(nn.Module):
    """AFES多任务模型，实现自适应特征和专家选择"""

    def __init__(self, user_feature_info, ws_feature_info,
                 embedding_dim=128, num_task_experts=4, num_shared_experts=4,
                 expert_hidden_dims=[512, 256], prediction_hidden_dims=[256, 128],
                 gate_hidden_dims=[256, 128], dropout_rate=0.2, num_heads=4,
                 use_attention=True, use_experts=True):
        super().__init__()

        # 特征嵌入层（包含自注意力处理）
        self.embedding = MultiTaskEmbedding(
            user_feature_info,
            ws_feature_info,
            embedding_dim,
            use_attention,
            num_heads,
            dropout_rate
        )

        # 计算嵌入后的特征维度
        self.input_dim = 16 * embedding_dim  # num_features * embedding_dim

        # 保存专家数量和控制参数
        self.num_task_experts = num_task_experts
        self.num_shared_experts = num_shared_experts
        self.use_attention = use_attention
        self.use_experts = use_experts

        # 如果使用专家网络，则初始化专家相关组件
        if use_experts:
            # 每个任务只能访问自己的专家和共享专家
            self.rt_experts_count = num_task_experts + num_shared_experts
            self.tp_experts_count = num_task_experts + num_shared_experts

            # RT任务专家
            self.rt_experts = nn.ModuleList([
                ExpertNetwork(self.input_dim, expert_hidden_dims, dropout_rate)
                for _ in range(num_task_experts)
            ])

            # TP任务专家
            self.tp_experts = nn.ModuleList([
                ExpertNetwork(self.input_dim, expert_hidden_dims, dropout_rate)
                for _ in range(num_task_experts)
            ])

            # 共享专家
            self.shared_experts = nn.ModuleList([
                ExpertNetwork(self.input_dim, expert_hidden_dims, dropout_rate)
                for _ in range(num_shared_experts)
            ])

            # 专家输出维度
            self.expert_output_dim = expert_hidden_dims[-1]

            # RT任务的门控网络
            self.rt_gate = GateNetwork(
                input_dim=self.input_dim,
                hidden_dims=gate_hidden_dims,
                output_dim=self.rt_experts_count,
                dropout_rate=dropout_rate
            )

            # TP任务的门控网络
            self.tp_gate = GateNetwork(
                input_dim=self.input_dim,
                hidden_dims=gate_hidden_dims,
                output_dim=self.tp_experts_count,
                dropout_rate=dropout_rate
            )

            # RT预测塔
            self.rt_prediction_head = PredictionHead(
                input_dim=self.expert_output_dim,
                hidden_dims=prediction_hidden_dims,
                dropout_rate=dropout_rate
            )

            # TP预测塔
            self.tp_prediction_head = PredictionHead(
                input_dim=self.expert_output_dim,
                hidden_dims=prediction_hidden_dims,
                dropout_rate=dropout_rate
            )
        else:
            # 如果不使用专家网络，直接接预测头
            # 加和后的特征维度与原始维度相同
            self.rt_prediction_head = PredictionHead(
                input_dim=self.input_dim,
                hidden_dims=prediction_hidden_dims,
                dropout_rate=dropout_rate
            )

            self.tp_prediction_head = PredictionHead(
                input_dim=self.input_dim,
                hidden_dims=prediction_hidden_dims,
                dropout_rate=dropout_rate
            )

    def forward(self, user_features, ws_features):
        # 嵌入特征并通过自注意力处理，获取三组展平后的特征
        rt_flat, tp_flat, shared_flat = self.embedding(user_features, ws_features)

        if self.use_experts:
            # 通过各专家网络
            rt_expert_outputs = [expert(rt_flat) for expert in self.rt_experts]
            tp_expert_outputs = [expert(tp_flat) for expert in self.tp_experts]
            shared_expert_outputs = [expert(shared_flat) for expert in self.shared_experts]

            # 合并RT专家和共享专家的输出，用于RT任务的门控网络
            rt_gate_experts = rt_expert_outputs + shared_expert_outputs
            rt_gate_experts_stacked = torch.stack(rt_gate_experts, dim=1)

            # 合并TP专家和共享专家的输出，用于TP任务的门控网络
            tp_gate_experts = tp_expert_outputs + shared_expert_outputs
            tp_gate_experts_stacked = torch.stack(tp_gate_experts, dim=1)

            # 计算门控网络的输出
            rt_gate_output = self.rt_gate(rt_flat)
            tp_gate_output = self.tp_gate(tp_flat)

            # 使用门控网络加权专家输出
            rt_final = torch.bmm(rt_gate_output.unsqueeze(1), rt_gate_experts_stacked).squeeze(1)
            tp_final = torch.bmm(tp_gate_output.unsqueeze(1), tp_gate_experts_stacked).squeeze(1)

            # 应用预测塔
            rt_pred = self.rt_prediction_head(rt_final)
            tp_pred = self.tp_prediction_head(tp_final)
        else:
            # 不使用专家网络，直接将任务注意力和共享注意力的输出加和
            # 对RT任务，加和rt_flat和shared_flat
            rt_sum = rt_flat + shared_flat
            # 对TP任务，加和tp_flat和shared_flat
            tp_sum = tp_flat + shared_flat

            # 直接应用预测塔
            rt_pred = self.rt_prediction_head(rt_sum)
            tp_pred = self.tp_prediction_head(tp_sum)

        return rt_pred, tp_pred


class RtTaskModel(nn.Module):
    """RT单任务模型，使用门控网络加权自身专家的输出"""

    def __init__(self, user_feature_info, ws_feature_info,
                 embedding_dim=128, num_experts=6,
                 expert_hidden_dims=[512, 256], prediction_hidden_dims=[256, 128],
                 gate_hidden_dims=[256, 128],
                 dropout_rate=0.2, num_heads=4):
        super().__init__()

        # 特征嵌入层（包含自注意力处理）- 使用单任务嵌入
        self.embedding = SingleTaskEmbedding(
            user_feature_info,
            ws_feature_info,
            embedding_dim,
            num_heads,
            dropout_rate
        )

        # 计算嵌入后的特征维度
        self.input_dim = 16 * embedding_dim  # num_features * embedding_dim

        # 保存专家数量
        self.num_experts = num_experts

        # 创建RT任务专家
        self.experts = nn.ModuleList([
            ExpertNetwork(self.input_dim, expert_hidden_dims, dropout_rate)
            for _ in range(num_experts)
        ])

        # 专家输出维度
        self.expert_output_dim = expert_hidden_dims[-1]

        # 创建门控网络来加权专家输出
        self.gate = GateNetwork(
            input_dim=self.input_dim,
            hidden_dims=gate_hidden_dims,
            output_dim=num_experts,
            dropout_rate=dropout_rate
        )

        # 预测塔
        self.prediction_head = PredictionHead(
            input_dim=self.expert_output_dim,
            hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate
        )

    def forward(self, user_features, ws_features):
        # 嵌入特征并通过自注意力处理，获取展平后的特征
        flat_features = self.embedding(user_features, ws_features)  # [batch_size, num_features * embedding_dim]

        # 通过专家网络
        expert_outputs = [expert(flat_features) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, expert_output_dim]

        # 使用门控网络加权专家输出
        gate_output = self.gate(flat_features)  # [batch_size, num_experts]
        weighted_output = torch.bmm(gate_output.unsqueeze(1), expert_outputs_stacked).squeeze(1)  # [batch_size, expert_output_dim]

        # 应用预测塔
        pred = self.prediction_head(weighted_output)

        return pred


class TpTaskModel(nn.Module):
    """TP单任务模型，使用门控网络加权自身专家的输出"""

    def __init__(self, user_feature_info, ws_feature_info,
                 embedding_dim=128, num_experts=6,
                 expert_hidden_dims=[512, 256], prediction_hidden_dims=[256, 128],
                 gate_hidden_dims=[256, 128],
                 dropout_rate=0.2, num_heads=4):
        super().__init__()

        # 特征嵌入层（包含自注意力处理）- 使用单任务嵌入
        self.embedding = SingleTaskEmbedding(
            user_feature_info,
            ws_feature_info,
            embedding_dim,
            num_heads,
            dropout_rate
        )

        # 计算嵌入后的特征维度
        self.input_dim = 16 * embedding_dim  # num_features * embedding_dim

        # 保存专家数量
        self.num_experts = num_experts

        # 创建TP任务专家
        self.experts = nn.ModuleList([
            ExpertNetwork(self.input_dim, expert_hidden_dims, dropout_rate)
            for _ in range(num_experts)
        ])

        # 专家输出维度
        self.expert_output_dim = expert_hidden_dims[-1]

        # 创建门控网络来加权专家输出
        self.gate = GateNetwork(
            input_dim=self.input_dim,
            hidden_dims=gate_hidden_dims,
            output_dim=num_experts,
            dropout_rate=dropout_rate
        )

        # 预测塔
        self.prediction_head = PredictionHead(
            input_dim=self.expert_output_dim,
            hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate
        )

    def forward(self, user_features, ws_features):
        # 嵌入特征并通过自注意力处理，获取展平后的特征
        flat_features = self.embedding(user_features, ws_features)  # [batch_size, num_features * embedding_dim]

        # 通过专家网络
        expert_outputs = [expert(flat_features) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, expert_output_dim]

        # 使用门控网络加权专家输出
        gate_output = self.gate(flat_features)  # [batch_size, num_experts]
        weighted_output = torch.bmm(gate_output.unsqueeze(1), expert_outputs_stacked).squeeze(1)  # [batch_size, expert_output_dim]

        # 应用预测塔
        pred = self.prediction_head(weighted_output)

        return pred


def train_multi_task_model(model, train_loader, valid_loader, rt_criterion, tp_criterion,
                           optimizer, scheduler, device, num_epochs, eval_epochs=10, early_stopping_patience=10,
                           rt_mean=0, rt_std=1, tp_mean=0, tp_std=1):
    """训练多任务AFES模型"""
    logger.info("开始训练多任务AFES模型")

    best_metrics = {
        "rt_mae": float('inf'),
        "rt_rmse": float('inf'),
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
        rt_losses = 0.0
        tp_losses = 0.0
        rt_samples = 0
        tp_samples = 0
        both_samples = 0
        batch_count = 0

        for batch in train_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            rt_targets = batch['rt'].to(device)
            tp_targets = batch['tp'].to(device)
            has_rt = batch['has_rt'].to(device)
            has_tp = batch['has_tp'].to(device)

            optimizer.zero_grad()

            # 前向传播
            rt_pred, tp_pred = model(user_features, ws_features)

            # 初始化损失
            loss = 0.0
            batch_rt_loss = 0.0
            batch_tp_loss = 0.0

            # 计算RT损失（仅对有RT值的样本）
            if has_rt.any():
                # 过滤有RT的样本
                valid_rt_preds = rt_pred.squeeze()[has_rt]
                valid_rt_targets = rt_targets[has_rt]

                rt_loss = rt_criterion(valid_rt_preds, valid_rt_targets)
                batch_rt_loss = rt_loss.item()
                loss += rt_loss
                rt_samples += has_rt.sum().item()

            # 计算TP损失（仅对有TP值的样本）
            if has_tp.any():
                # 过滤有TP的样本
                valid_tp_preds = tp_pred.squeeze()[has_tp]
                valid_tp_targets = tp_targets[has_tp]

                tp_loss = tp_criterion(valid_tp_preds, valid_tp_targets)
                batch_tp_loss = tp_loss.item()
                loss += tp_loss
                tp_samples += has_tp.sum().item()

            # 统计同时有RT和TP的样本数
            both = has_rt & has_tp
            both_samples += both.sum().item()

            # 只有在有损失时才进行反向传播
            if loss > 0:
                # 反向传播
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                rt_losses += batch_rt_loss
                tp_losses += batch_tp_loss
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
            f"RT损失: {avg_rt_loss:.4f} (样本数: {rt_samples}), "
            f"TP损失: {avg_tp_loss:.4f} (样本数: {tp_samples}), "
            f"同时有RT和TP的样本数: {both_samples}, "
            f"训练时间: {epoch_time:.2f}秒"
        )

        # 验证阶段
        if (epoch + 1) % eval_epochs == 0:  # 每k个epoch进行一次验证
            # 记录验证开始时间
            valid_start_time = time.time()

            valid_metrics = evaluate_multi_task_model(
                model, valid_loader, rt_criterion, tp_criterion,
                device, rt_mean, rt_std, tp_mean, tp_std
            )

            # 计算验证时间
            valid_time = time.time() - valid_start_time

            logger.info(
                f"验证损失: {valid_metrics['total_loss']:.4f}, "
                f"RT损失: {valid_metrics['rt_loss']:.4f} (样本数: {valid_metrics['rt_count']}), "
                f"TP损失: {valid_metrics['tp_loss']:.4f} (样本数: {valid_metrics['tp_count']}), "
                f"同时有RT和TP的样本数: {valid_metrics['both_count']}, "
                f"验证时间: {valid_time:.2f}秒"
            )

            logger.info(
                f"RT MAE: {valid_metrics['rt_mae']:.4f}, RT RMSE: {valid_metrics['rt_rmse']:.4f}, "
                f"TP MAE: {valid_metrics['tp_mae']:.4f}, TP RMSE: {valid_metrics['tp_rmse']:.4f}, "
            )

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
        f"TP MAE: {best_metrics['tp_mae']:.4f}, TP RMSE: {best_metrics['tp_rmse']:.4f}"
    )

    return best_metrics

def evaluate_multi_task_model(model, valid_loader, rt_criterion, tp_criterion, device,
                             rt_mean, rt_std, tp_mean, tp_std):
    """评估多任务AFES模型性能，只评估有数据的样本"""
    model.eval()
    valid_loss = 0.0
    rt_losses = 0.0
    tp_losses = 0.0
    batch_count = 0

    rt_preds = []
    tp_preds = []
    rt_targets_list = []
    tp_targets_list = []
    has_rt_list = []
    has_tp_list = []
    both_count = 0

    with torch.no_grad():
        for batch in valid_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            rt_targets = batch['rt'].to(device)
            tp_targets = batch['tp'].to(device)
            has_rt = batch['has_rt'].to(device)
            has_tp = batch['has_tp'].to(device)

            # 前向传播
            rt_pred, tp_pred = model(user_features, ws_features)

            # 初始化损失
            loss = 0.0
            batch_rt_loss = 0.0
            batch_tp_loss = 0.0

            # 计算RT损失（仅对有RT值的样本）
            if has_rt.any():
                # 过滤有RT的样本
                valid_rt_preds = rt_pred.squeeze()[has_rt]
                valid_rt_targets = rt_targets[has_rt]

                rt_loss = rt_criterion(valid_rt_preds, valid_rt_targets)
                batch_rt_loss = rt_loss.item()
                loss += rt_loss

            # 计算TP损失（仅对有TP值的样本）
            if has_tp.any():
                # 过滤有TP的样本
                valid_tp_preds = tp_pred.squeeze()[has_tp]
                valid_tp_targets = tp_targets[has_tp]

                tp_loss = tp_criterion(valid_tp_preds, valid_tp_targets)
                batch_tp_loss = tp_loss.item()
                loss += tp_loss

            # 统计同时有RT和TP的样本数
            both = has_rt & has_tp
            both_count += both.sum().item()

            # 只有在有损失时才计入总损失
            if loss > 0:
                valid_loss += loss.item()
                rt_losses += batch_rt_loss
                tp_losses += batch_tp_loss
                batch_count += 1

            # 收集预测和目标（包括标记）
            rt_preds.append(rt_pred.squeeze())
            tp_preds.append(tp_pred.squeeze())
            rt_targets_list.append(rt_targets)
            tp_targets_list.append(tp_targets)
            has_rt_list.append(has_rt)
            has_tp_list.append(has_tp)

    # 合并所有批次的预测和目标
    rt_preds = torch.cat(rt_preds)
    tp_preds = torch.cat(tp_preds)
    rt_targets = torch.cat(rt_targets_list)
    tp_targets = torch.cat(tp_targets_list)
    has_rt = torch.cat(has_rt_list)
    has_tp = torch.cat(has_tp_list)

    # 统计有效样本数
    rt_count = has_rt.sum().item()
    tp_count = has_tp.sum().item()

    # 反归一化
    rt_preds_denorm = rt_preds * rt_std + rt_mean
    tp_preds_denorm = tp_preds * tp_std + tp_mean
    rt_targets_denorm = rt_targets * rt_std + rt_mean
    tp_targets_denorm = tp_targets * tp_std + tp_mean

    # 初始化指标
    rt_mae = 0.0
    rt_rmse = 0.0
    tp_mae = 0.0
    tp_rmse = 0.0

    # 使用反归一化的值计算指标（仅对有数据的样本）
    if rt_count > 0:
        valid_rt_preds = rt_preds_denorm[has_rt]
        valid_rt_targets = rt_targets_denorm[has_rt]

        rt_mae = torch.mean(torch.abs(valid_rt_preds - valid_rt_targets)).item()
        rt_mse = torch.mean((valid_rt_preds - valid_rt_targets) ** 2).item()
        rt_rmse = torch.sqrt(torch.tensor(rt_mse)).item()

    if tp_count > 0:
        valid_tp_preds = tp_preds_denorm[has_tp]
        valid_tp_targets = tp_targets_denorm[has_tp]

        tp_mae = torch.mean(torch.abs(valid_tp_preds - valid_tp_targets)).item()
        tp_mse = torch.mean((valid_tp_preds - valid_tp_targets) ** 2).item()
        tp_rmse = torch.sqrt(torch.tensor(tp_mse)).item()

    # 计算平均损失
    avg_valid_loss = valid_loss / batch_count if batch_count > 0 else 0
    avg_rt_loss = rt_losses / batch_count if batch_count > 0 else 0
    avg_tp_loss = tp_losses / batch_count if batch_count > 0 else 0

    return {
        'total_loss': avg_valid_loss,
        'rt_loss': avg_rt_loss,
        'tp_loss': avg_tp_loss,
        'rt_count': rt_count,
        'tp_count': tp_count,
        'rt_mae': rt_mae,
        'rt_rmse': rt_rmse,
        'tp_mae': tp_mae,
        'tp_rmse': tp_rmse,
        'both_count': both_count
    }


def train_rt_task_model(model, train_loader, valid_loader, criterion,
                        optimizer, scheduler, num_epochs, device,
                        early_stopping_patience=10, rt_mean=0, rt_std=1, eval_epochs=10):
    """训练RT单任务模型"""
    logger.info("开始训练RT单任务模型")

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
            loss = criterion(rt_pred.squeeze(), rt_targets)

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

            valid_metrics = evaluate_rt_task_model(
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


def evaluate_rt_task_model(model, valid_loader, criterion, device, rt_mean, rt_std):
    """评估RT单任务模型性能"""
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
            loss = criterion(rt_pred.squeeze(), rt_targets)
            valid_loss += loss.item()
            batch_count += 1

            # 收集预测和目标
            rt_preds.append(rt_pred.squeeze())
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


def train_tp_task_model(model, train_loader, valid_loader, criterion,
                        optimizer, scheduler, num_epochs, device,
                        early_stopping_patience=10, tp_mean=0, tp_std=1, eval_epochs=10):
    """训练TP单任务模型"""
    logger.info("开始训练TP单任务模型")

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
            loss = criterion(tp_pred.squeeze(), tp_targets)

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

            valid_metrics = evaluate_tp_task_model(
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


def evaluate_tp_task_model(model, valid_loader, criterion, device, tp_mean, tp_std):
    """评估TP单任务模型性能"""
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
            loss = criterion(tp_pred.squeeze(), tp_targets)
            valid_loss += loss.item()
            batch_count += 1

            # 收集预测和目标
            tp_preds.append(tp_pred.squeeze())
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
    parser = argparse.ArgumentParser(description='AFES模型用于QoS预测')

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
    parser.add_argument('--dropout_rate', type=float, default=0.3)

    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_task_experts', type=int, default=4)
    parser.add_argument('--num_shared_experts', type=int, default=4)
    parser.add_argument('--expert_hidden_dims', type=str, default='512,256')
    parser.add_argument('--gate_hidden_dims', type=str, default='256,128')
    parser.add_argument('--prediction_hidden_dims', type=str, default='128,64')

    # 任务模式
    parser.add_argument('--task', type=str, default='multi', choices=['multi', 'rt_only', 'tp_only'])
    parser.add_argument('--use_attention', type=bool, default=True)
    parser.add_argument('--use_experts', type=bool, default=True)
    parser.add_argument('--use_single_attribute_samples', type=bool, default=True)

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 解析维度参数
    expert_hidden_dims = [int(x) for x in args.expert_hidden_dims.split(',')]
    prediction_hidden_dims = [int(x) for x in args.prediction_hidden_dims.split(',')]
    gate_hidden_dims = [int(x) for x in args.gate_hidden_dims.split(',')]

    # 检查GPU可用性
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
    if args.task == 'multi':
        # 构建多任务数据集 - 传递使用单属性样本的参数
        qos_dataset.build_multi_task_dataset(args.use_single_attribute_samples)

        # 创建PyTorch数据集
        train_torch_dataset = QoSMultiTorchDataset(qos_dataset.train_dataset, user_feature_dim, ws_feature_dim)
        valid_torch_dataset = QoSMultiTorchDataset(qos_dataset.valid_dataset, user_feature_dim, ws_feature_dim)

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

        # 输出模型配置信息
        logger.info("===== 多任务AFES模型配置 =====")
        logger.info(f"嵌入维度: {args.embedding_dim}")
        logger.info(f"注意力头数: {args.num_heads}")
        logger.info(f"任务专家数量: {args.num_task_experts}")
        logger.info(f"共享专家数量: {args.num_shared_experts}")
        logger.info(f"专家隐藏层维度: {expert_hidden_dims}")
        logger.info(f"门控隐藏层维度: {gate_hidden_dims}")
        logger.info(f"预测塔隐藏层维度: {prediction_hidden_dims}")
        logger.info(f"使用自注意力机制: {'是' if args.use_attention else '否'}")
        logger.info(f"使用专家网络: {'是' if args.use_experts else '否'}")
        logger.info(f"使用单属性样本: {'是' if args.use_single_attribute_samples else '否'}")
        logger.info("===================")

        # 创建多任务AFES模型
        model = AFESMultiModel(
            user_feature_info=user_feature_info,
            ws_feature_info=ws_feature_info,
            embedding_dim=args.embedding_dim,
            num_task_experts=args.num_task_experts,
            num_shared_experts=args.num_shared_experts,
            expert_hidden_dims=expert_hidden_dims,
            prediction_hidden_dims=prediction_hidden_dims,
            gate_hidden_dims=gate_hidden_dims,
            dropout_rate=args.dropout_rate,
            num_heads=args.num_heads,
            use_attention=args.use_attention,
            use_experts=args.use_experts
        ).to(device)

        # 统计模型参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"可训练参数数量: {trainable_params}")

        # 定义损失函数
        rt_criterion = nn.SmoothL1Loss()
        tp_criterion = nn.SmoothL1Loss()

        # 创建优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=1e-5
        )

        # 训练多任务模型
        best_metrics = train_multi_task_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            rt_criterion=rt_criterion,
            tp_criterion=tp_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.num_epochs,
            eval_epochs=args.eval_epochs,
            early_stopping_patience=args.early_stopping_patience,
            rt_mean=qos_dataset.rt_mean,
            rt_std=qos_dataset.rt_std,
            tp_mean=qos_dataset.tp_mean,
            tp_std=qos_dataset.tp_std
        )

    elif args.task == 'rt_only':
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

        # 输出模型配置信息
        logger.info("===== RT单任务模型配置 =====")
        logger.info(f"嵌入维度: {args.embedding_dim}")
        logger.info(f"注意力头数: {args.num_heads}")
        logger.info(f"专家数量: {args.num_task_experts}")
        logger.info(f"专家隐藏层维度: {expert_hidden_dims}")
        logger.info(f"门控隐藏层维度: {gate_hidden_dims}")
        logger.info(f"预测塔隐藏层维度: {prediction_hidden_dims}")
        logger.info(f"使用门控网络加权专家输出")
        logger.info("===================")

        # 创建RT单任务模型
        model = RtTaskModel(
            user_feature_info=user_feature_info,
            ws_feature_info=ws_feature_info,
            embedding_dim=args.embedding_dim,
            num_experts=args.num_task_experts,
            expert_hidden_dims=expert_hidden_dims,
            prediction_hidden_dims=prediction_hidden_dims,
            gate_hidden_dims=gate_hidden_dims,
            dropout_rate=args.dropout_rate,
            num_heads=args.num_heads
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

        # 训练RT单任务模型
        best_metrics = train_rt_task_model(
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

        # 输出模型配置信息
        logger.info("===== TP单任务模型配置 =====")
        logger.info(f"嵌入维度: {args.embedding_dim}")
        logger.info(f"注意力头数: {args.num_heads}")
        logger.info(f"专家数量: {args.num_task_experts}")
        logger.info(f"专家隐藏层维度: {expert_hidden_dims}")
        logger.info(f"门控隐藏层维度: {gate_hidden_dims}")
        logger.info(f"预测塔隐藏层维度: {prediction_hidden_dims}")
        logger.info(f"使用门控网络加权专家输出")
        logger.info("===================")

        # 创建TP单任务模型
        model = TpTaskModel(
            user_feature_info=user_feature_info,
            ws_feature_info=ws_feature_info,
            embedding_dim=args.embedding_dim,
            num_experts=args.num_task_experts,
            expert_hidden_dims=expert_hidden_dims,
            prediction_hidden_dims=prediction_hidden_dims,
            gate_hidden_dims=gate_hidden_dims,
            dropout_rate=args.dropout_rate,
            num_heads=args.num_heads
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

        # 训练TP单任务模型
        best_metrics = train_tp_task_model(
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

    return best_metrics


if __name__ == '__main__':
    main()
