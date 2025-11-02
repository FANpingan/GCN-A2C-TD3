"""
配置文件：包含所有超参数和系统配置
"""
import torch
import os

class Config:
    """全局配置类"""
    
    # ============ 基础设置 ============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    # ============ 路径设置 ============
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets")
    RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
    MODELS_PATH = os.path.join(RESULTS_PATH, "models")
    LOGS_PATH = os.path.join(RESULTS_PATH, "logs")
    PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")
    
    # ============ EdgeSimPy环境设置 ============
    # 边缘服务器配置
    NUM_EDGE_SERVERS = 10
    EDGE_CPU_CAPACITY = 10000  # MIPS
    EDGE_RAM_CAPACITY = 8192   # MB
    EDGE_DISK_CAPACITY = 100000  # MB
    
    # 基站配置
    NUM_BASE_STATIONS = 5
    
    # 用户配置
    NUM_USERS = 20
    
    # 网络配置
    BANDWIDTH = 100  # Mbps
    NETWORK_DELAY = 10  # ms
    
    # ============ 任务配置 ============
    # YOLOv5/v8任务特征
    TASK_CPU_DEMAND_RANGE = (500, 5000)  # MIPS
    TASK_RAM_DEMAND_RANGE = (256, 2048)  # MB
    TASK_DATA_SIZE_RANGE = (1, 10)  # MB
    TASK_ARRIVAL_RATE = 5  # tasks per time step
    
    # 任务优先级
    TASK_PRIORITIES = {
        'critical': 1.0,   # 安防关键任务
        'normal': 0.5,     # 普通监控
        'background': 0.1  # 后台分析
    }
    
    # ============ GCN网络配置 ============
    # GCN架构
    GCN_INPUT_DIM = 8  # 节点特征维度
    GCN_HIDDEN_DIM = 128
    GCN_OUTPUT_DIM = 64
    GCN_NUM_LAYERS = 3
    GCN_DROPOUT = 0.1
    
    # ============ A2C高层决策配置 ============
    # 网络架构
    A2C_ACTOR_HIDDEN_DIMS = [256, 128]
    A2C_CRITIC_HIDDEN_DIMS = [256, 128]
    
    # 训练参数
    A2C_LR = 3e-4
    A2C_GAMMA = 0.99
    A2C_VALUE_COEF = 0.5
    A2C_ENTROPY_COEF = 0.01  # 熵正则化系数
    A2C_MAX_GRAD_NORM = 0.5
    A2C_N_STEPS = 5  # n-step returns
    
    # 高层决策频率
    HIGH_LEVEL_DECISION_INTERVAL = 10  # 每10步做一次高层决策
    
    # ============ TD3低层决策配置 ============
    # 网络架构
    TD3_ACTOR_HIDDEN_DIMS = [256, 256]
    TD3_CRITIC_HIDDEN_DIMS = [256, 256]
    
    # 训练参数
    TD3_LR_ACTOR = 3e-4
    TD3_LR_CRITIC = 3e-4
    TD3_GAMMA = 0.99
    TD3_TAU = 0.005  # 软更新系数
    TD3_POLICY_NOISE = 0.2  # 目标策略平滑噪声
    TD3_NOISE_CLIP = 0.5
    TD3_POLICY_DELAY = 2  # 延迟策略更新
    TD3_BUFFER_SIZE = 1000000
    TD3_BATCH_SIZE = 256
    TD3_WARMUP_STEPS = 1000  # 预热步数
    
    # ============ 奖励函数权重 ============
    # 多目标优化权重
    WEIGHT_LATENCY = 0.4      # 延迟权重
    WEIGHT_ENERGY = 0.3       # 能耗权重
    WEIGHT_ACCURACY = 0.2     # 精度权重
    WEIGHT_MIGRATION = 0.1    # 迁移成本权重
    
    # 归一化参数
    MAX_LATENCY = 1000  # ms
    MAX_ENERGY = 500    # W
    MAX_MIGRATION_COST = 100
    
    # ============ 训练配置 ============
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 500
    EVAL_INTERVAL = 50  # 每50个episode评估一次
    SAVE_INTERVAL = 100  # 每100个episode保存一次
    
    # 早停配置
    PATIENCE = 100  # 早停耐心值
    MIN_IMPROVEMENT = 0.001  # 最小改进阈值
    
    # ============ 评估配置 ============
    NUM_EVAL_EPISODES = 10
    EVAL_RENDER = False
    
    # ============ 日志配置 ============
    USE_TENSORBOARD = True
    USE_WANDB = False  # 如果使用wandb，需要配置API key
    WANDB_PROJECT = "gcn-hos"
    WANDB_ENTITY = None  # 你的wandb用户名
    
    LOG_INTERVAL = 10  # 每10步记录一次
    VERBOSE = True
    
    # ============ 可视化配置 ============
    PLOT_TRAINING_CURVES = True
    PLOT_NETWORK_TOPOLOGY = True
    PLOT_RESOURCE_UTILIZATION = True
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.DATASET_PATH, exist_ok=True)
        os.makedirs(cls.MODELS_PATH, exist_ok=True)
        os.makedirs(cls.LOGS_PATH, exist_ok=True)
        os.makedirs(cls.PLOTS_PATH, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("Configuration Settings")
        print("=" * 60)
        print(f"Device: {cls.DEVICE}")
        print(f"Seed: {cls.SEED}")
        print(f"\nEdge Servers: {cls.NUM_EDGE_SERVERS}")
        print(f"Base Stations: {cls.NUM_BASE_STATIONS}")
        print(f"Users: {cls.NUM_USERS}")
        print(f"\nGCN Hidden Dim: {cls.GCN_HIDDEN_DIM}")
        print(f"A2C Learning Rate: {cls.A2C_LR}")
        print(f"TD3 Learning Rate: {cls.TD3_LR_ACTOR}")
        print(f"\nTraining Episodes: {cls.NUM_EPISODES}")
        print(f"Max Steps per Episode: {cls.MAX_STEPS_PER_EPISODE}")
        print("=" * 60)


# 创建全局配置实例
config = Config()

# 在导入时创建目录
Config.create_dirs()
