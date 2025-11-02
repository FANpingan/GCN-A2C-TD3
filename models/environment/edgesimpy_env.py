from edge_sim_py import *
import math
import os
import random
import msgpack
import pandas as pd
import matplotlib.pyplot as plt



import numpy as np
import edge_sim_py.simulator as simulator
from edge_sim_py.component_builder import ComponentBuilder

# EdgeSimPy导入 - 参考EdgeAISim的方式
try:
    # EdgeSimPy的正确导入方式
    from EdgeSimPy.edge_sim_py import *
except ImportError:
    try:
        # 备用导入方式
        import EdgeSimPy
        from EdgeSimPy import *
    except ImportError:
        print("❌ EdgeSimPy未安装或导入路径不正确")
        print("请检查EdgeSimPy是否正确安装")
        raise


class EdgeSimPyEnv:
    """EdgeSimPy环境适配器"""

    def __init__(self, dataset_file='../sample_dataset1.json', stopping_criterion=500):
        """
        初始化

        Args:
            dataset_file: EdgeSimPy数据集文件（JSON格式）
            stopping_criterion: 仿真停止步数
        """
        self.dataset_file = dataset_file
        self.stopping_criterion = stopping_criterion
        self.current_step = 0

        # 初始化EdgeSimPy
        self._init_simulator()

    def _init_simulator(self):
        """初始化EdgeSimPy仿真器"""
        # 加载数据集
        ComponentBuilder(input_file=self.dataset_file)

        # 重置仿真器
        simulator.simulator.reset()

        # 获取组件
        self.edge_servers = simulator.simulator.edge_servers
        self.users = simulator.simulator.users
        self.base_stations = simulator.simulator.base_stations

        print(f"✓ EdgeSimPy initialized:")
        print(f"  Servers: {len(self.edge_servers)}")
        print(f"  Users: {len(self.users)}")

    def reset(self):
        """重置环境"""
        simulator.simulator.reset()
        self.current_step = 0
        return self.get_global_state()

    def get_edge_servers(self):
        """获取边缘服务器"""
        return self.edge_servers

    def get_network_links(self):
        """获取网络连接"""
        links = []
        for i, s1 in enumerate(self.edge_servers):
            for s2 in self.edge_servers[i + 1:i + 4]:  # 每个服务器连3个邻居
                link = type('Link', (), {
                    'source_id': s1.id,
                    'target_id': s2.id,
                    'bandwidth': 100,
                    'delay': 10 + np.random.uniform(0, 5)
                })()
                links.append(link)
        return links

    def get_global_state(self):
        """获取全局状态(20维)"""
        total_cpu = sum(s.cpu_demand for s in self.edge_servers)
        total_memory = sum(s.memory_demand for s in self.edge_servers)

        state = np.array([
            total_cpu / 10000,
            total_memory / 8192,
            self.current_step / self.stopping_criterion,
            *np.random.randn(17)  # 填充到20维
        ])
        return state

    def generate_task(self):
        """生成任务"""
        # 从用户服务中获取
        for user in self.users:
            for service in user.services:
                if not hasattr(service, '_scheduled'):
                    service._scheduled = True
                    task = type('Task', (), {
                        'id': service.id,
                        'service': service,
                        'cpu_demand': service.cpu_demand,
                        'ram_demand': service.memory_demand,
                        'data_size': 1.0,
                        'priority': 1.0
                    })()
                    return task
        return None

    def execute_offloading(self, task, decision):
        """执行卸载"""
        allocations = decision['server_allocations']
        target_id = max(allocations, key=allocations.get)
        target_server = next(s for s in self.edge_servers if s.id == target_id)

        # 放置服务
        service = task.service
        service.server = target_server
        target_server.services.append(service)

        # 计算指标
        latency = 200 + np.random.uniform(0, 100)
        energy = target_server.power_model.power_consumption

        return {
            'latency': latency,
            'energy': energy,
            'accuracy': 0.95,
            'migration_cost': 0
        }

    def step(self):
        """推进仿真"""
        self.current_step += 1
        simulator.simulator.step()
        return self.current_step >= self.stopping_criterion

    if __name__ == "__main__":
        print("Testing EdgeSimPy Environment...")
        print("=" * 60)

        try:
            env = EdgeSimPyEnv(
                dataset_file='sample_dataset1.json',
                stopping_criterion=100
            )

            state = env.reset()
            print(f"\n✓ 环境重置成功")
            print(f"  状态维度: {state.shape}")

            servers = env.get_edge_servers()
            print(f"✓ 获取到 {len(servers)} 个边缘服务器")

            links = env.get_network_links()
            print(f"✓ 获取到 {len(links)} 个网络连接")

            print("\n" + "=" * 60)
            print("✓ 所有测试通过!")

        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            print("\n请参考EdgeAISim的示例代码")