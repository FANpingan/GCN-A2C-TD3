"""
评估脚本：评估训练好的模型性能
"""
import os
import numpy as np
import torch
import argparse
from tqdm import tqdm

from config import config
from hierarchical_scheduler import HierarchicalScheduler
from environment.edge_env import SimulatedEdgeEnv
from train import compute_reward


def evaluate(args):
    """评估模型性能"""
    print("=" * 60)
    print("Evaluating GCN-A2C-TD3 Model")
    print("=" * 60)
    
    # 加载模型
    scheduler = HierarchicalScheduler(
        num_edge_servers=config.NUM_EDGE_SERVERS,
        num_clusters=args.num_clusters,
        gcn_input_dim=config.GCN_INPUT_DIM,
        gcn_hidden_dim=config.GCN_HIDDEN_DIM,
        gcn_output_dim=config.GCN_OUTPUT_DIM
    )
    
    if args.model_path:
        scheduler.load(args.model_path)
        print(f"Model loaded from: {args.model_path}")
    else:
        print("Warning: No model path provided, using untrained model")
    
    # 创建环境
    env = SimulatedEdgeEnv(
        num_servers=config.NUM_EDGE_SERVERS,
        num_users=config.NUM_USERS
    )
    
    # 评估循环
    all_rewards = []
    all_latencies = []
    all_energies = []
    all_accuracies = []
    
    print(f"\nRunning {args.num_episodes} evaluation episodes...")
    
    for episode in tqdm(range(args.num_episodes)):
        state = env.reset()
        episode_reward = 0
        episode_latency = 0
        episode_energy = 0
        episode_accuracy = 0
        step_count = 0
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            task = env.generate_task()
            if task is None:
                continue
            
            # 使用确定性策略（无探索）
            edge_servers = env.get_edge_servers()
            network_links = env.get_network_links()
            global_state = env.get_global_state()
            
            decision = scheduler.schedule_task(
                task, edge_servers, network_links, global_state
            )
            
            result = env.execute_offloading(task, decision)
            reward = compute_reward(result)
            
            episode_reward += reward
            episode_latency += result['latency']
            episode_energy += result['energy']
            episode_accuracy += result['accuracy']
            step_count += 1
            
            done = env.step()
            if done:
                break
        
        # 计算平均值
        if step_count > 0:
            all_rewards.append(episode_reward)
            all_latencies.append(episode_latency / step_count)
            all_energies.append(episode_energy / step_count)
            all_accuracies.append(episode_accuracy / step_count)
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes: {args.num_episodes}")
    print(f"\nPerformance Metrics:")
    print(f"  Average Reward:    {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"  Average Latency:   {np.mean(all_latencies):.2f} ± {np.std(all_latencies):.2f} ms")
    print(f"  Average Energy:    {np.mean(all_energies):.2f} ± {np.std(all_energies):.2f} W")
    print(f"  Average Accuracy:  {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    
    # 最佳和最差情况
    print(f"\nBest Episode:")
    best_idx = np.argmax(all_rewards)
    print(f"  Reward:   {all_rewards[best_idx]:.4f}")
    print(f"  Latency:  {all_latencies[best_idx]:.2f} ms")
    print(f"  Energy:   {all_energies[best_idx]:.2f} W")
    
    print(f"\nWorst Episode:")
    worst_idx = np.argmin(all_rewards)
    print(f"  Reward:   {all_rewards[worst_idx]:.4f}")
    print(f"  Latency:  {all_latencies[worst_idx]:.2f} ms")
    print(f"  Energy:   {all_energies[worst_idx]:.2f} W")
    print("=" * 60)
    
    # 保存结果
    if args.save_results:
        import json
        results = {
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'mean_latency': float(np.mean(all_latencies)),
            'std_latency': float(np.std(all_latencies)),
            'mean_energy': float(np.mean(all_energies)),
            'std_energy': float(np.std(all_energies)),
            'mean_accuracy': float(np.mean(all_accuracies)),
            'std_accuracy': float(np.std(all_accuracies)),
            'all_rewards': [float(r) for r in all_rewards],
            'all_latencies': [float(l) for l in all_latencies],
            'all_energies': [float(e) for e in all_energies]
        }
        
        save_path = os.path.join(config.RESULTS_PATH, 'evaluation_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GCN-A2C-TD3 Model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (without extension)')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--num_clusters', type=int, default=3,
                        help='Number of edge clusters')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 开始评估
    evaluate(args)
