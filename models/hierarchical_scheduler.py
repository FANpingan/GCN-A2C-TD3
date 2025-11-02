"""
分层任务卸载调度器 (Hierarchical Scheduler)
协调GCN + A2C (高层) 和 GCN + TD3 (低层) 的决策
"""
import torch
import numpy as np
from models.gcn import EdgeNetworkGCN, build_graph_from_edge_network
from models.a2c_agent import A2CAgent
from models.td3_agent import TD3Agent
from config import config


class HierarchicalScheduler:
    """
    分层调度器
    
    架构:
        1. GCN提取网络拓扑特征
        2. 高层A2C做粗粒度决策（选择集群）
        3. 低层TD3做细粒度决策（资源分配）
    """
    def __init__(
        self,
        num_edge_servers,
        num_clusters,
        gcn_input_dim=8,
        gcn_hidden_dim=128,
        gcn_output_dim=64
    ):
        """
        初始化分层调度器
        
        Args:
            num_edge_servers: 边缘服务器数量
            num_clusters: 边缘集群数量
            gcn_input_dim: GCN输入维度（节点特征维度）
            gcn_hidden_dim: GCN隐藏层维度
            gcn_output_dim: GCN输出维度
        """
        self.device = config.DEVICE
        self.num_edge_servers = num_edge_servers
        self.num_clusters = num_clusters
        
        # ============ GCN网络 ============
        self.gcn = EdgeNetworkGCN(
            input_dim=gcn_input_dim,
            hidden_dim=gcn_hidden_dim,
            output_dim=gcn_output_dim,
            num_layers=config.GCN_NUM_LAYERS,
            dropout=config.GCN_DROPOUT
        ).to(self.device)
        
        # ============ 高层A2C决策 ============
        # 状态维度 = GCN图嵌入 + 额外全局状态
        high_level_state_dim = gcn_output_dim + 20  # 20维额外状态（负载、任务队列等）
        self.high_level_agent = A2CAgent(
            state_dim=high_level_state_dim,
            action_dim=num_clusters,
            gcn_output_dim=gcn_output_dim,
            lr=config.A2C_LR,
            gamma=config.A2C_GAMMA,
            value_coef=config.A2C_VALUE_COEF,
            entropy_coef=config.A2C_ENTROPY_COEF,
            max_grad_norm=config.A2C_MAX_GRAD_NORM,
            n_steps=config.A2C_N_STEPS
        )
        
        # ============ 低层TD3决策 ============
        # 状态维度 = GCN节点嵌入均值 + 集群内服务器状态
        low_level_state_dim = gcn_output_dim + num_edge_servers * 4  # 4维状态/服务器
        # 动作维度 = 集群内各服务器的资源分配比例
        low_level_action_dim = num_edge_servers
        
        self.low_level_agent = TD3Agent(
            state_dim=low_level_state_dim,
            action_dim=low_level_action_dim,
            max_action=1.0,
            lr_actor=config.TD3_LR_ACTOR,
            lr_critic=config.TD3_LR_CRITIC,
            gamma=config.TD3_GAMMA,
            tau=config.TD3_TAU,
            policy_noise=config.TD3_POLICY_NOISE,
            noise_clip=config.TD3_NOISE_CLIP,
            policy_delay=config.TD3_POLICY_DELAY,
            buffer_size=config.TD3_BUFFER_SIZE,
            batch_size=config.TD3_BATCH_SIZE
        )
        
        # ============ 决策计数器 ============
        self.step_count = 0
        self.current_cluster = 0  # 当前选定的集群
        
        # ============ 缓存 ============
        self.cached_graph_embedding = None
        self.cached_node_embeddings = None
        
    def extract_graph_features(self, edge_servers, network_links):
        """
        使用GCN提取图特征
        
        Args:
            edge_servers: 边缘服务器列表
            network_links: 网络连接列表
        
        Returns:
            node_embeddings: 节点嵌入 [num_nodes, gcn_output_dim]
            graph_embedding: 图嵌入 [gcn_output_dim]
        """
        # 构建图
        x, edge_index, edge_weight = build_graph_from_edge_network(
            edge_servers, network_links
        )
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        
        # 创建批次索引（单个图）
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        
        # GCN前向传播
        with torch.no_grad():
            node_embeddings, graph_embedding = self.gcn(
                x, edge_index, batch, edge_weight
            )
        
        return node_embeddings, graph_embedding
    
    def build_high_level_state(self, graph_embedding, global_state):
        """
        构建高层决策状态
        
        Args:
            graph_embedding: GCN图嵌入
            global_state: 全局状态（系统负载、任务队列等）
        
        Returns:
            high_level_state: 高层状态向量
        """
        graph_emb = graph_embedding.cpu().numpy().flatten()
        state = np.concatenate([graph_emb, global_state])
        return state
    
    def build_low_level_state(self, node_embeddings, cluster_id, server_states):
        """
        构建低层决策状态
        
        Args:
            node_embeddings: GCN节点嵌入
            cluster_id: 选定的集群ID
            server_states: 服务器状态列表
        
        Returns:
            low_level_state: 低层状态向量
        """
        # 获取集群内节点的平均嵌入
        cluster_node_emb = node_embeddings.mean(dim=0).cpu().numpy()
        
        # 展平服务器状态
        server_features = np.array(server_states).flatten()
        
        state = np.concatenate([cluster_node_emb, server_features])
        return state
    
    def make_high_level_decision(self, edge_servers, network_links, global_state):
        """
        高层决策：选择卸载到哪个集群
        
        Args:
            edge_servers: 边缘服务器列表
            network_links: 网络连接列表
            global_state: 全局状态
        
        Returns:
            cluster_id: 选定的集群ID
            log_prob: 对数概率
            value: 状态价值
        """
        # 提取图特征（缓存以供低层使用）
        self.cached_node_embeddings, self.cached_graph_embedding = \
            self.extract_graph_features(edge_servers, network_links)
        
        # 构建高层状态
        high_level_state = self.build_high_level_state(
            self.cached_graph_embedding,
            global_state
        )
        
        # A2C选择动作
        cluster_id, log_prob, value = self.high_level_agent.select_action(
            high_level_state
        )
        
        self.current_cluster = cluster_id
        
        return cluster_id, log_prob, value
    
    def make_low_level_decision(self, cluster_id, server_states, exploration_noise=0.1):
        """
        低层决策：在选定集群内分配资源
        
        Args:
            cluster_id: 集群ID
            server_states: 服务器状态列表
            exploration_noise: 探索噪声
        
        Returns:
            resource_allocation: 资源分配向量（归一化到和为1）
        """
        # 使用缓存的节点嵌入
        low_level_state = self.build_low_level_state(
            self.cached_node_embeddings,
            cluster_id,
            server_states
        )
        
        # TD3选择动作
        raw_action = self.low_level_agent.select_action(
            low_level_state,
            noise=exploration_noise
        )
        
        # 归一化资源分配（使用softmax确保和为1）
        resource_allocation = self._normalize_allocation(raw_action)
        
        return resource_allocation
    
    def _normalize_allocation(self, raw_action):
        """
        归一化资源分配
        
        使用softmax确保:
        1. 所有值非负
        2. 和为1
        """
        # 转换为概率分布
        exp_actions = np.exp(raw_action - np.max(raw_action))
        allocation = exp_actions / exp_actions.sum()
        return allocation
    
    def schedule_task(self, task, edge_servers, network_links, global_state):
        """
        完整的任务调度流程
        
        Args:
            task: 任务对象
            edge_servers: 边缘服务器列表
            network_links: 网络连接列表
            global_state: 全局状态
        
        Returns:
            offloading_decision: 卸载决策字典
                {
                    'cluster_id': 集群ID,
                    'server_allocations': {server_id: allocation_ratio},
                    'high_level_info': {...},
                    'low_level_info': {...}
                }
        """
        self.step_count += 1
        
        # 是否进行高层决策
#        if self.step_count % config.HIGH_LEVEL_DECISION_INTERVAL == 0:
        if self.step_count == 1 or self.step_count % config.HIGH_LEVEL_DECISION_INTERVAL == 0:
            cluster_id, log_prob, value = self.make_high_level_decision(
                edge_servers, network_links, global_state
            )
            high_level_info = {
                'cluster_id': cluster_id,
                'log_prob': log_prob,
                'value': value
            }
        else:
            # 使用上次的高层决策
            cluster_id = self.current_cluster
            high_level_info = None
        
        # 低层决策（每步都执行）
        server_states = self._extract_server_states(edge_servers, cluster_id)
        resource_allocation = self.make_low_level_decision(
            cluster_id,
            server_states
        )
        
        # 构建服务器分配字典
        cluster_servers = self._get_cluster_servers(edge_servers, cluster_id)
        server_allocations = {
            server.id: allocation
            for server, allocation in zip(cluster_servers, resource_allocation)
        }
        
        offloading_decision = {
            'cluster_id': cluster_id,
            'server_allocations': server_allocations,
            'high_level_info': high_level_info,
            'low_level_info': {
                'resource_allocation': resource_allocation
            }
        }
        
        return offloading_decision
    
    def _extract_server_states(self, edge_servers, cluster_id):
        """提取服务器状态"""
        cluster_servers = self._get_cluster_servers(edge_servers, cluster_id)
        
        server_states = []
        for server in cluster_servers:
            state = [
                server.cpu / server.cpu_capacity,
                server.ram / server.ram_capacity,
                server.power_consumption / 500,
                len(server.services) / 10
            ]
            server_states.append(state)
        
        # 填充到固定长度
        while len(server_states) < self.num_edge_servers:
            server_states.append([0, 0, 0, 0])
        
        return server_states[:self.num_edge_servers]
    
    def _get_cluster_servers(self, edge_servers, cluster_id):
        """获取集群内的服务器"""
        # 简单划分：按server id划分集群
        servers_per_cluster = len(edge_servers) // self.num_clusters
        start_idx = cluster_id * servers_per_cluster
        end_idx = start_idx + servers_per_cluster
        
        if cluster_id == self.num_clusters - 1:
            # 最后一个集群包含剩余所有服务器
            return edge_servers[start_idx:]
        
        return edge_servers[start_idx:end_idx]
    
    def update_high_level(self, trajectory):
        """
        更新高层A2C
        
        Args:
            trajectory: 轨迹数据
                {
                    'states': [...],
                    'actions': [...],
                    'log_probs': [...],
                    'rewards': [...],
                    'dones': [...],
                    'next_state': ...
                }
        """
        return self.high_level_agent.update(
            states=trajectory['states'],
            actions=trajectory['actions'],
            log_probs=trajectory['log_probs'],
            rewards=trajectory['rewards'],
            dones=trajectory['dones'],
            next_state=trajectory['next_state']
        )
    
    def update_low_level(self, state, action, next_state, reward, done):
        """
        更新低层TD3
        
        Args:
            state: 当前状态
            action: 动作
            next_state: 下一状态
            reward: 奖励
            done: 是否终止
        """
        # 添加到回放池
        self.low_level_agent.replay_buffer.add(
            state, action, next_state, reward, done
        )
        
        # 更新网络
        return self.low_level_agent.update()
    
    def save(self, path_prefix):
        """保存所有模型"""
        self.high_level_agent.save(f"{path_prefix}_high_level.pth")
        self.low_level_agent.save(f"{path_prefix}_low_level.pth")
        torch.save(self.gcn.state_dict(), f"{path_prefix}_gcn.pth")
        print(f"Hierarchical scheduler saved with prefix: {path_prefix}")
    
    def load(self, path_prefix):
        """加载所有模型"""
        self.high_level_agent.load(f"{path_prefix}_high_level.pth")
        self.low_level_agent.load(f"{path_prefix}_low_level.pth")
        self.gcn.load_state_dict(torch.load(
            f"{path_prefix}_gcn.pth",
            map_location=self.device
        ))
        print(f"Hierarchical scheduler loaded with prefix: {path_prefix}")


if __name__ == "__main__":
    # 测试分层调度器
    print("Testing Hierarchical Scheduler...")
    
    num_edge_servers = 10
    num_clusters = 3
    
    scheduler = HierarchicalScheduler(
        num_edge_servers=num_edge_servers,
        num_clusters=num_clusters
    )
    
    # 模拟边缘服务器（简化版本）
    class DummyServer:
        def __init__(self, id):
            self.id = id
            self.cpu = np.random.randint(100, 5000)
            self.ram = np.random.randint(100, 4000)
            self.disk = np.random.randint(1000, 50000)
            self.cpu_capacity = 10000
            self.ram_capacity = 8192
            self.disk_capacity = 100000
            self.power_consumption = np.random.uniform(50, 300)
            self.services = []
            self.x = np.random.uniform(0, 1000)
            self.y = np.random.uniform(0, 1000)
            self.active = True
    
#    edge_servers = [DummyServer(i) for i in range(num_edge_servers)]
#    network_links = []  # 简化测试，不构建网络
    edge_servers = [DummyServer(i) for i in range(num_edge_servers)]
    print(f"✓ Created {num_edge_servers} edge servers")


    # 修复：创建实际的网络连接
    class DummyLink:
        def __init__(self, source, target):
            self.source_id = source
            self.target_id = target
            self.bandwidth = 100
            self.delay = 10 + np.random.uniform(0, 5)


    network_links = []
    for i in range(num_edge_servers):
        for j in range(i + 1, min(i + 3, num_edge_servers)):
            network_links.append(DummyLink(i, j))

    print(f"✓ Created {len(network_links)} network links")
    global_state = np.random.randn(20)
    
    # # 测试调度
    # class DummyTask:
    #     pass
    #
    # task = DummyTask()
    # decision = scheduler.schedule_task(
    #     task, edge_servers, network_links, global_state
    # )
    #
    # print(f"\nOffloading Decision:")
    # print(f"  Cluster ID: {decision['cluster_id']}")
    # print(f"  Server Allocations: {decision['server_allocations']}")
    #
    # print("\nAll tests passed!")
    # 测试调度（多次以验证高层和低层决策）
    class DummyTask:
        def __init__(self, id):
            self.id = id


    print("\nTesting task scheduling...")
    print("-" * 60)

    for i in range(15):  # 测试15步，会触发2次高层决策
        task = DummyTask(i)
        decision = scheduler.schedule_task(
            task, edge_servers, network_links, global_state
        )

        if decision['high_level_info'] is not None:
            print(f"\n✓ Step {i + 1}: HIGH-LEVEL DECISION")
            print(f"  ├─ Cluster ID: {decision['cluster_id']}")
            print(f"  ├─ Log Prob: {decision['high_level_info']['log_prob']:.4f}")
            print(f"  └─ Value: {decision['high_level_info']['value']:.4f}")
        else:
            print(f"  Step {i + 1}: Low-level (cluster {decision['cluster_id']})")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
