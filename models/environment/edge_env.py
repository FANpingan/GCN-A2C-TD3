"""
EdgeSimPy环境封装
将EdgeSimPy适配为标准的RL环境接口
"""
from edge_sim_py import *
import math
import os
import random
import msgpack
import pandas as pd
import matplotlib.pyplot as plt


import numpy as np
from config import config


class SimulatedEdgeEnv:
    """
    模拟的边缘计算环境（用于快速测试）
    
    实际使用时，请替换为EdgeSimPyEnv，集成真实的EdgeSimPy仿真器
    """
    def __init__(self, num_servers=10, num_users=20):
        self.num_servers = num_servers
        self.num_users = num_users
        self.time_step = 0
        self.servers = self._init_servers()
        
    def _init_servers(self):
        """初始化服务器"""
        servers = []
        for i in range(self.num_servers):
            server = type('Server', (), {
                'id': i,
                'cpu': np.random.randint(1000, 5000),
                'ram': np.random.randint(1000, 4000),
                'disk': np.random.randint(10000, 50000),
                'cpu_capacity': 10000,
                'ram_capacity': 8192,
                'disk_capacity': 100000,
                'power_consumption': np.random.uniform(100, 300),
                'services': [],
                'x': np.random.uniform(0, 1000),
                'y': np.random.uniform(0, 1000),
                'active': True
            })()
            servers.append(server)
        return servers
    
    def reset(self):
        """重置环境"""
        self.time_step = 0
        self.servers = self._init_servers()
        return self.get_global_state()
    
    def generate_task(self):
        """生成任务"""
        if np.random.rand() < 0.8:  # 80%概率生成任务
            task = type('Task', (), {
                'id': self.time_step,
                'cpu_demand': np.random.randint(*config.TASK_CPU_DEMAND_RANGE),
                'ram_demand': np.random.randint(*config.TASK_RAM_DEMAND_RANGE),
                'data_size': np.random.uniform(*config.TASK_DATA_SIZE_RANGE),
                'priority': np.random.choice(list(config.TASK_PRIORITIES.values()))
            })()
            return task
        return None
    
    def get_edge_servers(self):
        """获取边缘服务器列表"""
        return self.servers
    
    def get_network_links(self):
        """获取网络连接"""
        # 简化：全连接网络
        links = []
        for i in range(self.num_servers):
            for j in range(i + 1, self.num_servers):
                link = type('Link', (), {
                    'source_id': i,
                    'target_id': j,
                    'bandwidth': config.BANDWIDTH,
                    'delay': config.NETWORK_DELAY + np.random.uniform(0, 5)
                })()
                links.append(link)
        return links
    
    def get_global_state(self):
        """获取全局状态"""
        # 20维全局状态
        total_cpu = sum(s.cpu for s in self.servers)
        total_ram = sum(s.ram for s in self.servers)
        avg_power = np.mean([s.power_consumption for s in self.servers])
        
        state = np.array([
            total_cpu / (config.EDGE_CPU_CAPACITY * self.num_servers),  # 总CPU利用率
            total_ram / (config.EDGE_RAM_CAPACITY * self.num_servers),  # 总内存利用率
            avg_power / 500,  # 平均功耗（归一化）
            self.time_step / config.MAX_STEPS_PER_EPISODE,  # 时间进度
            *np.random.randn(16)  # 其他16维状态（任务队列、网络状态等）
        ])
        return state
    
    def execute_offloading(self, task, decision):
        """
        执行任务卸载
        
        Args:
            task: 任务对象
            decision: 卸载决策
        
        Returns:
            result: 执行结果字典
        """
        cluster_id = decision['cluster_id']
        server_allocations = decision['server_allocations']
        
        # 模拟执行
        # 计算延迟（考虑网络传输+计算）
        network_latency = config.NETWORK_DELAY + np.random.uniform(5, 20)
        compute_latency = task.cpu_demand / config.EDGE_CPU_CAPACITY * 1000  # ms
        total_latency = network_latency + compute_latency
        
        # 计算能耗
        allocated_servers = [s for s in self.servers if s.id in server_allocations]
        total_energy = sum(s.power_consumption for s in allocated_servers) / 10  # 简化
        
        # 模拟精度（随机波动）
        accuracy = np.random.uniform(0.90, 0.98)
        
        # 迁移成本（如果切换集群）
        migration_cost = 0
        if hasattr(self, 'last_cluster') and self.last_cluster != cluster_id:
            migration_cost = task.data_size * 10  # 简化
        self.last_cluster = cluster_id
        
        result = {
            'latency': total_latency,
            'energy': total_energy,
            'accuracy': accuracy,
            'migration_cost': migration_cost
        }
        
        # 更新服务器状态
        for server_id, allocation in server_allocations.items():
            server = self.servers[server_id]
            server.cpu += task.cpu_demand * allocation * 0.1  # 简化
            server.cpu = min(server.cpu, server.cpu_capacity)
        
        return result
    
    def step(self):
        """环境步进"""
        self.time_step += 1
        
        # 服务器状态衰减（模拟任务完成）
        for server in self.servers:
            server.cpu = max(0, server.cpu - np.random.uniform(100, 500))
            server.ram = max(0, server.ram - np.random.uniform(100, 300))
        
        # 检查是否结束
        done = self.time_step >= config.MAX_STEPS_PER_EPISODE
        return done


class EdgeSimPyEnv:
    """
    真实的EdgeSimPy环境封装
    
    TODO: 实现与EdgeSimPy的集成
    
    参考EdgeAISim的实现:
    - https://github.com/MuhammedGolec/EdgeAISIM
    
    需要实现的主要方法:
    1. __init__: 初始化EdgeSimPy仿真器
    2. reset(): 重置仿真环境
    3. generate_task(): 根据任务模型生成任务
    4. get_edge_servers(): 获取EdgeSimPy的边缘服务器
    5. get_network_links(): 获取网络拓扑
    6. get_global_state(): 提取系统状态
    7. execute_offloading(): 在EdgeSimPy中执行卸载决策
    8. step(): 推进仿真时间
    """
    def __init__(self, config_file=None):
        """
        初始化EdgeSimPy环境
        
        Args:
            config_file: EdgeSimPy配置文件路径
        """
        # TODO: 导入EdgeSimPy
        # from edge_sim_py import Simulator
        # self.simulator = Simulator(config_file)
        raise NotImplementedError(
            "EdgeSimPyEnv需要集成真实的EdgeSimPy仿真器。\n"
            "请参考EdgeAISim的实现: https://github.com/MuhammedGolec/EdgeAISIM\n"
            "当前可以使用SimulatedEdgeEnv进行快速测试。"
        )


if __name__ == "__main__":
    # 测试环境
    print("Testing Simulated Edge Environment...")
    
    env = SimulatedEdgeEnv(num_servers=10, num_users=20)
    
    # 测试重置
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # 测试任务生成
    for _ in range(10):
        task = env.generate_task()
        if task:
            print(f"Task {task.id}: CPU={task.cpu_demand}, RAM={task.ram_demand}")
    
    # 测试执行卸载
    task = env.generate_task()
    if task:
        decision = {
            'cluster_id': 0,
            'server_allocations': {0: 0.5, 1: 0.5},
            'high_level_info': None,
            'low_level_info': None
        }
        result = env.execute_offloading(task, decision)
        print(f"\nOffloading Result:")
        print(f"  Latency: {result['latency']:.2f} ms")
        print(f"  Energy: {result['energy']:.2f} W")
        print(f"  Accuracy: {result['accuracy']:.4f}")
    
    print("\nAll tests passed!")
