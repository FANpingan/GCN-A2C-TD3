"""
图卷积网络(GCN)实现
用于提取边缘网络拓扑特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCN(nn.Module):
    """
    基础GCN网络
    用于处理图结构数据并提取节点特征
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN卷积层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Batch Normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
        
        Returns:
            节点嵌入向量 [num_nodes, output_dim]
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层不使用激活函数和dropout
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x


class EdgeNetworkGCN(nn.Module):
    """
    边缘网络专用GCN
    包含图级别和节点级别的特征提取
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(EdgeNetworkGCN, self).__init__()
        
        # 节点级别GCN
        self.gcn = GCN(input_dim, hidden_dim, output_dim, num_layers, dropout)
        
        # 图级别聚合
        self.graph_fc = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None, edge_weight=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes] (用于图级别池化)
            edge_weight: 边权重 [num_edges] (可选)
        
        Returns:
            node_embeddings: 节点嵌入 [num_nodes, output_dim]
            graph_embedding: 图嵌入 [batch_size, output_dim] (如果提供batch)
        """
        # 节点级别特征
        node_embeddings = self.gcn(x, edge_index, edge_weight)
        
        # 图级别特征（如果提供batch）
        if batch is not None:
            # 使用mean和max pooling聚合
            mean_pool = global_mean_pool(node_embeddings, batch)
            max_pool = global_max_pool(node_embeddings, batch)
            graph_embedding = torch.cat([mean_pool, max_pool], dim=-1)
            graph_embedding = self.graph_fc(graph_embedding)
            return node_embeddings, graph_embedding
        
        return node_embeddings, None


class AttentionGCN(nn.Module):
    """
    带注意力机制的GCN
    用于动态关注重要的边缘服务器节点
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1, num_heads=4):
        super(AttentionGCN, self).__init__()
        
        self.gcn = GCN(input_dim, hidden_dim, output_dim, num_layers, dropout)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_fc = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
        
        Returns:
            增强的节点嵌入 [num_nodes, output_dim]
        """
        # GCN特征提取
        node_features = self.gcn(x, edge_index, edge_weight)
        
        # 自注意力 (需要转换为[batch_size, seq_len, embed_dim]格式)
        node_features_attn = node_features.unsqueeze(0)  # [1, num_nodes, output_dim]
        attn_output, _ = self.attention(
            node_features_attn,
            node_features_attn,
            node_features_attn
        )
        attn_output = attn_output.squeeze(0)  # [num_nodes, output_dim]
        
        # 残差连接
        output = node_features + attn_output
        output = self.output_fc(output)
        
        return output


def build_graph_from_edge_network(edge_servers, network_links):
    """
    从EdgeSimPy的边缘网络构建PyTorch Geometric图
    
    Args:
        edge_servers: EdgeSimPy的边缘服务器列表
        network_links: 网络连接列表
    
    Returns:
        x: 节点特征矩阵 [num_nodes, feature_dim]
        edge_index: 边索引 [2, num_edges]
        edge_weight: 边权重 [num_edges]
    """
    num_servers = len(edge_servers)
    
    # 构建节点特征 (8维特征)
    node_features = []
    for server in edge_servers:
        features = [
            server.cpu / server.cpu_capacity,           # CPU利用率
            server.ram / server.ram_capacity,           # 内存利用率
            server.disk / server.disk_capacity,         # 磁盘利用率
            server.power_consumption / 500,             # 归一化功耗
            len(server.services) / 10,                  # 服务数量(归一化)
            server.x / 1000,                            # x坐标(归一化)
            server.y / 1000,                            # y坐标(归一化)
            1.0 if server.active else 0.0               # 是否激活
        ]
        node_features.append(features)
    
    x = torch.FloatTensor(node_features)
    
    # 构建边索引和边权重
    edge_list = []
    edge_weights = []
    
    for link in network_links:
        # 假设link包含source, target, bandwidth, delay等信息
        source_id = link.source_id
        target_id = link.target_id
        
        # 添加双向边
        edge_list.append([source_id, target_id])
        edge_list.append([target_id, source_id])
        
        # 边权重基于带宽和延迟
        weight = link.bandwidth / (link.delay + 1e-6)
        edge_weights.append(weight)
        edge_weights.append(weight)
    
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    edge_weight = torch.FloatTensor(edge_weights)
    
    return x, edge_index, edge_weight


if __name__ == "__main__":
    # 测试GCN网络
    print("Testing GCN models...")
    
    # 创建简单测试数据
    num_nodes = 10
    input_dim = 8
    hidden_dim = 128
    output_dim = 64
    
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.LongTensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    
    # 测试基础GCN
    gcn = GCN(input_dim, hidden_dim, output_dim)
    output = gcn(x, edge_index)
    print(f"GCN output shape: {output.shape}")
    
    # 测试EdgeNetworkGCN
    edge_gcn = EdgeNetworkGCN(input_dim, hidden_dim, output_dim)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    node_emb, graph_emb = edge_gcn(x, edge_index, batch)
    print(f"Node embeddings shape: {node_emb.shape}")
    print(f"Graph embedding shape: {graph_emb.shape}")
    
    # 测试AttentionGCN
    attn_gcn = AttentionGCN(input_dim, hidden_dim, output_dim)
    output = attn_gcn(x, edge_index)
    print(f"AttentionGCN output shape: {output.shape}")
    
    print("All tests passed!")
