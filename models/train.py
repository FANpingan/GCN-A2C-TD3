"""
训练脚本：训练GCN-A2C-TD3分层任务卸载模型
"""
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from config import config
from hierarchical_scheduler import HierarchicalScheduler


def compute_reward(offloading_result):
    """
    计算奖励函数
    
    综合考虑：延迟、能耗、精度、迁移成本
    
    Args:
        offloading_result: 卸载执行结果字典
            {
                'latency': 任务延迟(ms),
                'energy': 能耗(W),
                'accuracy': 推理精度,
                'migration_cost': 迁移成本
            }
    
    Returns:
        reward: 标量奖励值
    """
    # 归一化各指标
    latency_norm = offloading_result['latency'] / config.MAX_LATENCY
    energy_norm = offloading_result['energy'] / config.MAX_ENERGY
    migration_norm = offloading_result['migration_cost'] / config.MAX_MIGRATION_COST
    accuracy = offloading_result.get('accuracy', 0.95)
    
    # 加权求和（惩罚延迟和能耗，奖励精度）
    reward = (
        - config.WEIGHT_LATENCY * latency_norm
        - config.WEIGHT_ENERGY * energy_norm
        + config.WEIGHT_ACCURACY * accuracy
        - config.WEIGHT_MIGRATION * migration_norm
    )
    
    return reward


def train_one_episode(scheduler, env, episode_num):
    """
    训练一个episode
    
    Args:
        scheduler: 分层调度器
        env: EdgeSimPy环境
        episode_num: 当前episode编号
    
    Returns:
        episode_metrics: episode统计指标
    """
    # 重置环境
    state = env.reset()
    
    # 轨迹数据（用于A2C更新）
    high_level_trajectory = {
        'states': [],
        'actions': [],
        'log_probs': [],
        'rewards': [],
        'dones': []
    }
    
    episode_reward = 0
    episode_latency = 0
    episode_energy = 0
    step_count = 0
    
    for step in range(config.MAX_STEPS_PER_EPISODE):
        # 模拟任务到达
        task = env.generate_task()
        if task is None:
            continue
        
        # 分层调度决策
        edge_servers = env.get_edge_servers()
        network_links = env.get_network_links()
        global_state = env.get_global_state()
        
        decision = scheduler.schedule_task(
            task, edge_servers, network_links, global_state
        )
        
        # 执行卸载
        result = env.execute_offloading(task, decision)
        
        # 计算奖励
        reward = compute_reward(result)
        episode_reward += reward
        episode_latency += result['latency']
        episode_energy += result['energy']
        
        # # 保存高层轨迹（如果有高层决策）
        # if decision['high_level_info'] is not None:
        #     high_level_info = decision['high_level_info']
        #     high_level_trajectory['states'].append(global_state)
        #     high_level_trajectory['actions'].append(high_level_info['cluster_id'])
        #     high_level_trajectory['log_probs'].append(high_level_info['log_prob'])
        #     high_level_trajectory['rewards'].append(reward)
        #     high_level_trajectory['dones'].append(False)

        # 保存高层轨迹（如果有高层决策）
        if decision['high_level_info'] is not None:
            high_level_info = decision['high_level_info']
            # 构建完整的高层状态
            high_level_state = scheduler.build_high_level_state(
                scheduler.cached_graph_embedding, global_state
            )
            high_level_trajectory['states'].append(high_level_state)  # ✅ 完整84维
            high_level_trajectory['actions'].append(high_level_info['cluster_id'])
            high_level_trajectory['log_probs'].append(high_level_info['log_prob'])
            high_level_trajectory['rewards'].append(reward)
            high_level_trajectory['dones'].append(False)

        # 更新低层TD3（每步更新）
        # 构建低层状态
        low_state = scheduler.build_low_level_state(
            scheduler.cached_node_embeddings,
            decision['cluster_id'],
            scheduler._extract_server_states(edge_servers, decision['cluster_id'])
        )
        
        # 获取下一状态
        next_state = env.get_global_state()
        next_low_state = scheduler.build_low_level_state(
            scheduler.cached_node_embeddings,
            decision['cluster_id'],
            scheduler._extract_server_states(edge_servers, decision['cluster_id'])
        )
        
        # 更新TD3
        scheduler.update_low_level(
            low_state,
            decision['low_level_info']['resource_allocation'],
            next_low_state,
            reward,
            False
        )
        
        step_count += 1
        
        # 环境步进
        done = env.step()
        if done:
            break
    
    # # Episode结束，更新高层A2C
    # if len(high_level_trajectory['states']) > 0:
    #     high_level_trajectory['next_state'] = env.get_global_state()
    #     high_level_trajectory['dones'][-1] = True
    #     scheduler.update_high_level(high_level_trajectory)
    # Episode结束，更新高层A2C
    if len(high_level_trajectory['states']) > 0:
        # 构建完整的高层next_state（需要包含GCN图嵌入）
        final_global_state = env.get_global_state()
        edge_servers = env.get_edge_servers()
        network_links = env.get_network_links()

        # 提取最后的图特征
        _, final_graph_embedding = scheduler.extract_graph_features(
            edge_servers, network_links
        )

        # 构建完整的高层状态
        final_high_level_state = scheduler.build_high_level_state(
            final_graph_embedding, final_global_state
        )

        high_level_trajectory['next_state'] = final_high_level_state
        high_level_trajectory['dones'][-1] = True
        scheduler.update_high_level(high_level_trajectory)

    # 返回统计指标
    avg_latency = episode_latency / max(step_count, 1)
    avg_energy = episode_energy / max(step_count, 1)
    
    return {
        'episode_reward': episode_reward,
        'avg_latency': avg_latency,
        'avg_energy': avg_energy,
        'steps': step_count
    }


def train(args):
    """主训练函数"""
    print("=" * 60)
    print("Starting Training: GCN-A2C-TD3 Hierarchical Task Offloading")
    print("=" * 60)
    
    # 打印配置
    config.print_config()
    
    # 创建分层调度器
    scheduler = HierarchicalScheduler(
        num_edge_servers=config.NUM_EDGE_SERVERS,
        num_clusters=args.num_clusters,
        gcn_input_dim=config.GCN_INPUT_DIM,
        gcn_hidden_dim=config.GCN_HIDDEN_DIM,
        gcn_output_dim=config.GCN_OUTPUT_DIM
    )
    
    # 创建环境（这里需要实现EdgeSimPy环境封装）
    # from environment.edge_env import EdgeSimPyEnv
    # env = EdgeSimPyEnv(...)
    
    # # 临时：使用模拟环境
    # print("\n⚠️  Warning: Using simulated environment for demonstration")
    # print("   Replace with actual EdgeSimPy environment in environment/edge_env.py\n")
    #
    # from environment.edge_env import SimulatedEdgeEnv
    # env = SimulatedEdgeEnv(
    #     num_servers=config.NUM_EDGE_SERVERS,
    #     num_users=config.NUM_USERS
    # )
    if args.use_edgesimpy:
        print("\n✓ Using EdgeSimPy environment")
        from environment.edgesimpy_env import EdgeSimPyEnv
        env = EdgeSimPyEnv(
            dataset_file=args.dataset_file,
            stopping_criterion=config.MAX_STEPS_PER_EPISODE
        )
    else:
        print("\n⚠️  Using simulated environment")
        from environment.edge_env import SimulatedEdgeEnv
        env = SimulatedEdgeEnv(
            num_servers=config.NUM_EDGE_SERVERS,
            num_users=config.NUM_USERS
        )
    # 训练循环
    best_reward = -float('inf')
    rewards_history = []
    latency_history = []
    energy_history = []
    
    for episode in tqdm(range(args.epochs), desc="Training"):
        # 训练一个episode
        metrics = train_one_episode(scheduler, env, episode)
        
        # 记录指标
        rewards_history.append(metrics['episode_reward'])
        latency_history.append(metrics['avg_latency'])
        energy_history.append(metrics['avg_energy'])
        
        # 日志输出
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(rewards_history[-args.log_interval:])
            avg_latency = np.mean(latency_history[-args.log_interval:])
            avg_energy = np.mean(energy_history[-args.log_interval:])
            
            print(f"\nEpisode {episode + 1}/{args.epochs}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Latency: {avg_latency:.2f} ms")
            print(f"  Avg Energy: {avg_energy:.2f} W")
        
        # 保存最佳模型
        if metrics['episode_reward'] > best_reward:
            best_reward = metrics['episode_reward']
            save_path = os.path.join(config.MODELS_PATH, "best_model")
            scheduler.save(save_path)
            print(f"✓ Best model saved! Reward: {best_reward:.4f}")
        
        # 定期保存
        if (episode + 1) % args.save_interval == 0:
            save_path = os.path.join(config.MODELS_PATH, f"model_ep{episode+1}")
            scheduler.save(save_path)
    
    # 保存最终模型
    final_path = os.path.join(config.MODELS_PATH, "final_model")
    scheduler.save(final_path)
    
    # 绘制训练曲线
    plot_training_curves(rewards_history, latency_history, energy_history)
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print(f"Best Reward: {best_reward:.4f}")
    print(f"Models saved to: {config.MODELS_PATH}")
    print("=" * 60)


def plot_training_curves(rewards, latencies, energies):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 奖励曲线
    axes[0].plot(rewards, alpha=0.6, label='Episode Reward')
    axes[0].plot(smooth_curve(rewards, 50), linewidth=2, label='Smoothed (50)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 延迟曲线
    axes[1].plot(latencies, alpha=0.6, label='Avg Latency')
    axes[1].plot(smooth_curve(latencies, 50), linewidth=2, label='Smoothed (50)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('Average Task Latency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 能耗曲线
    axes[2].plot(energies, alpha=0.6, label='Avg Energy')
    axes[2].plot(smooth_curve(energies, 50), linewidth=2, label='Smoothed (50)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Energy (W)')
    axes[2].set_title('Average Energy Consumption')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(config.PLOTS_PATH, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    plt.close()


def smooth_curve(values, window=50):
    """平滑曲线"""
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    return smoothed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GCN-A2C-TD3 Model')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_clusters', type=int, default=3)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    # 添加EdgeSimPy支持
    parser.add_argument('--use_edgesimpy', action='store_true',
                        help='Use EdgeSimPy instead of simulated environment')
    parser.add_argument('--dataset_file', type=str, default='../sample_dataset1.json',
                        help='EdgeSimPy dataset file')

    args = parser.parse_args()
    # parser = argparse.ArgumentParser(description='Train GCN-A2C-TD3 Model')
    # parser.add_argument('--epochs', type=int, default=1000, help='Number of training episodes')
    # parser.add_argument('--num_clusters', type=int, default=3, help='Number of edge clusters')
    # parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    # parser.add_argument('--save_interval', type=int, default=100, help='Save interval')
    # parser.add_argument('--seed', type=int, default=42, help='Random seed')
    #
    # args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 开始训练
    train(args)
