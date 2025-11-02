# GCN-Based Hierarchical Task Offloading System (GCN-HOS)

基于图卷积网络和分层强化学习的智能安防云边协同任务卸载系统

## 🎯 项目概述

本项目实现了一个两层分层决策架构：
- **高层决策**: GCN + A2C (离散决策，选择卸载目标集群)
- **低层决策**: GCN + TD3 (连续决策，资源分配优化)

## 📁 项目结构

```
GCN_Hierarchical_Offloading/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖包列表
├── config.py                          # 配置文件
├── models/                            # 模型定义
│   ├── __init__.py
│   ├── gcn.py                         # GCN图卷积网络
│   ├── a2c_agent.py                   # A2C高层决策
│   └── td3_agent.py                   # TD3低层决策
├── environment/                       # 仿真环境
│   ├── __init__.py
│   ├── edge_env.py                    # EdgeSimPy环境封装
│   └── task_generator.py             # 任务生成器
├── utils/                             # 工具函数
│   ├── __init__.py
│   ├── replay_buffer.py               # 经验回放池
│   └── logger.py                      # 日志记录
├── hierarchical_scheduler.py         # 分层调度器(核心)
├── train.py                          # 训练脚本
├── evaluate.py                       # 评估脚本
├── datasets/                         # 数据集目录
│   └── edge_network_topology.json    # 网络拓扑配置
└── results/                          # 实验结果
    ├── models/                       # 保存的模型
    ├── logs/                         # 训练日志
    └── plots/                        # 可视化图表
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py --epochs 1000 --save_interval 100
```

### 3. 评估模型

```bash
python evaluate.py --model_path results/models/best_model.pth
```

## 📊 核心功能

### 高层决策 (GCN + A2C)
- 使用GCN提取边缘网络拓扑特征
- A2C进行离散动作选择（卸载到哪个边缘集群）
- 熵正则化增强探索

### 低层决策 (GCN + TD3)
- 使用GCN提取选定集群内的服务器状态
- TD3进行连续资源分配优化
- 双Q网络减少过估计

### 分层协调
- 高层每T步做一次粗粒度决策
- 低层持续进行细粒度资源分配
- 异步更新机制

## 📈 性能指标

- **平均任务延迟**: 端到端推理时间
- **系统能耗**: 边缘服务器总功耗
- **任务完成率**: 成功完成任务的比例
- **资源利用率**: CPU/内存/带宽使用率

## 🔬 算法对比

实验对比以下baseline:
- Random Offloading
- All Cloud
- All Edge
- DDPG-based (单层)
- GCN-A2C-TD3 (本方法)

## 📝 引用

如果使用本项目，请引用：
```bibtex
@mastersthesis{fan2025gcn,
  title={基于混合智能体强化学习的云边协同智能安防任务卸载模型},
  author={范平安},
  school={中国地质大学(武汉)},
  year={2025}
}
```

## 🤝 贡献

欢迎提issue和PR！

## 📧 联系方式

- 作者: 范平安
- 学校: 中国地质大学(武汉)
- 导师: 吴湘宁
