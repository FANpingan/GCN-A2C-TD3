# 🚀 快速开始指南

## 项目完整实现 - GCN-A2C-TD3 分层任务卸载系统

恭喜！这是一个完整的、可运行的GCN + A2C（高层）+ GCN + TD3（低层）分层任务卸载框架实现。

---

## 📋 已完成的组件

### ✅ 核心模型
- **models/gcn.py**: 图卷积网络（3种实现：基础GCN、EdgeNetworkGCN、AttentionGCN）
- **models/a2c_agent.py**: A2C高层决策代理（离散动作，选择集群）
- **models/td3_agent.py**: TD3低层决策代理（连续动作，资源分配）

### ✅ 核心框架
- **hierarchical_scheduler.py**: 分层调度器（协调高低层决策）
- **config.py**: 完整的配置系统
- **train.py**: 训练脚本
- **environment/edge_env.py**: 环境封装（含模拟环境）

---

## 🔧 安装步骤

### 1. 安装依赖

```bash
cd GCN_Hierarchical_Offloading

# 安装Python依赖
pip install -r requirements.txt

# 如果torch-geometric安装有问题，使用：
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2. 验证安装

```bash
# 测试GCN网络
python models/gcn.py

# 测试A2C代理
python models/a2c_agent.py

# 测试TD3代理
python models/td3_agent.py

# 测试分层调度器
python hierarchical_scheduler.py
```

---

## 🎯 运行训练

### 基础训练（使用模拟环境）

```bash
# 快速测试（10个episodes）
python train.py --epochs 10 --num_clusters 3

# 完整训练（1000个episodes）
python train.py --epochs 1000 --num_clusters 3 --save_interval 100

# 自定义训练
python train.py \
    --epochs 500 \
    --num_clusters 5 \
    --log_interval 10 \
    --save_interval 50 \
    --seed 42
```

### 训练参数说明

- `--epochs`: 训练的episode数量（默认1000）
- `--num_clusters`: 边缘集群数量（默认3）
- `--log_interval`: 日志输出间隔（默认10）
- `--save_interval`: 模型保存间隔（默认100）
- `--seed`: 随机种子（默认42）

---

## 📊 查看结果

训练后会生成以下文件：

```
results/
├── models/
│   ├── best_model_high_level.pth    # 最佳A2C模型
│   ├── best_model_low_level.pth     # 最佳TD3模型
│   ├── best_model_gcn.pth           # 最佳GCN模型
│   └── final_model_*.pth            # 最终模型
├── logs/
│   └── training.log                 # 训练日志
└── plots/
    └── training_curves.png          # 训练曲线
```

---

## 🔗 集成EdgeSimPy（重要！）

当前代码使用**模拟环境**进行快速测试。要使用真实的EdgeSimPy仿真器，需要：

### Step 1: 实现EdgeSimPyEnv

编辑 `environment/edge_env.py`，实现 `EdgeSimPyEnv` 类：

```python
class EdgeSimPyEnv:
    def __init__(self, config_file):
        # 初始化EdgeSimPy
        from edge_sim_py import Simulator
        self.simulator = Simulator(config_file)
        
    def reset(self):
        # 重置仿真环境
        self.simulator.reset()
        return self.get_global_state()
    
    # 实现其他方法...
```

### Step 2: 参考EdgeAISim

EdgeAISim已经实现了EdgeSimPy的集成，可以直接参考：

```bash
# 克隆EdgeAISim
git clone https://github.com/MuhammedGolec/EdgeAISIM.git

# 查看他们如何集成EdgeSimPy
# 特别关注以下文件：
# - Qlearning_migration.py
# - GCN_Q_learning.py
```

### Step 3: 修改train.py

将train.py中的环境创建部分改为：

```python
# 替换
from environment.edge_env import SimulatedEdgeEnv
env = SimulatedEdgeEnv(...)

# 改为
from environment.edge_env import EdgeSimPyEnv
env = EdgeSimPyEnv(config_file="datasets/edge_network_topology.json")
```

---

## 🎓 核心算法流程

### 高层决策（GCN + A2C）

```
1. GCN提取网络拓扑特征
   ↓
2. 构建高层状态（图嵌入 + 全局状态）
   ↓
3. A2C选择集群（离散动作）
   ↓
4. 每N步更新一次（粗粒度决策）
```

### 低层决策（GCN + TD3）

```
1. 使用高层选定的集群
   ↓
2. GCN提取集群内节点特征
   ↓
3. 构建低层状态（节点嵌入 + 服务器状态）
   ↓
4. TD3输出资源分配比例（连续动作）
   ↓
5. 每步更新（细粒度决策）
```

### 训练流程

```
For each episode:
    For each step:
        1. 生成任务
        2. 高层决策（每N步）→ 选择集群
        3. 低层决策（每步）→ 资源分配
        4. 执行卸载 → 获取奖励
        5. 更新TD3（每步）
    6. 更新A2C（episode结束时）
```

---

## 🔬 实验对比

### Baseline算法

1. **Random Offloading**: 随机选择服务器
2. **All Cloud**: 全部卸载到云端
3. **All Edge**: 全部在边缘处理
4. **DDPG-based**: 单层DDPG（用于对比TD3）
5. **GCN-A2C-TD3**: 本方法

### 评估指标

- 平均任务延迟（ms）
- 系统总能耗（W）
- 任务完成率（%）
- 资源利用率（%）

---

## 📝 修改配置

所有超参数都在 `config.py` 中，可以轻松修改：

```python
# 网络架构
GCN_HIDDEN_DIM = 128        # GCN隐藏层维度
A2C_LR = 3e-4              # A2C学习率
TD3_LR_ACTOR = 3e-4        # TD3学习率

# 训练参数
NUM_EPISODES = 1000        # 训练episodes
HIGH_LEVEL_DECISION_INTERVAL = 10  # 高层决策间隔

# 奖励权重
WEIGHT_LATENCY = 0.4       # 延迟权重
WEIGHT_ENERGY = 0.3        # 能耗权重
WEIGHT_ACCURACY = 0.2      # 精度权重
```

---

## 🐛 常见问题

### Q1: torch-geometric安装失败？

```bash
# 根据你的PyTorch版本安装
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Q2: EdgeSimPy在哪里？

EdgeSimPy需要单独安装：
```bash
pip install git+https://github.com/EdgeSimPy/EdgeSimPy.git@v1.1.0
```

### Q3: 如何可视化网络拓扑？

```python
import matplotlib.pyplot as plt
import networkx as nx

# 在训练脚本中添加
G = nx.Graph()
for link in network_links:
    G.add_edge(link.source_id, link.target_id)
nx.draw(G, with_labels=True)
plt.savefig('topology.png')
```

---

## 📚 进阶功能

### 1. 添加Attention机制

在 `hierarchical_scheduler.py` 中将GCN替换为AttentionGCN：

```python
from models.gcn import AttentionGCN
self.gcn = AttentionGCN(...)
```

### 2. 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir=results/logs

# 在浏览器打开
http://localhost:6006
```

### 3. 添加WandB日志

在 `config.py` 中设置：
```python
USE_WANDB = True
WANDB_PROJECT = "gcn-hos"
WANDB_ENTITY = "your-username"
```

---

## 🎉 完成检查清单

- [x] 安装所有依赖
- [x] 运行单元测试（各模型的main函数）
- [x] 运行模拟训练（10 episodes）
- [ ] 集成EdgeSimPy
- [ ] 运行完整训练（1000 episodes）
- [ ] 评估模型性能
- [ ] 对比baseline算法
- [ ] 撰写论文实验部分

---

## 📧 需要帮助？

如果遇到任何问题：

1. 检查 `config.py` 中的路径设置
2. 确认所有依赖已正确安装
3. 查看 `results/logs/` 中的错误日志
4. 运行各模块的单元测试

---

## 🎊 祝贺！

你现在拥有了一个完整的、可运行的GCN-A2C-TD3分层任务卸载框架！

**下一步建议：**
1. 先用模拟环境跑通整个流程（10-100 episodes）
2. 理解代码结构和数据流
3. 集成EdgeSimPy
4. 开始正式实验

**Good Luck! 🚀**
