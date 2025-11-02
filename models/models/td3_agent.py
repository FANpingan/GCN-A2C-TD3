"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 低层决策代理
负责连续决策：资源分配优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from config import config


class TD3Actor(nn.Module):
    """TD3 Actor网络：输出连续动作"""
    def __init__(self, state_dim, action_dim, max_action, hidden_dims=[256, 256]):
        super(TD3Actor, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # 输出范围[-1, 1]
        
        self.network = nn.Sequential(*layers)
        self.max_action = max_action
        
    def forward(self, state):
        """前向传播"""
        action = self.network(state)
        return self.max_action * action


class TD3Critic(nn.Module):
    """TD3 Critic网络：双Q网络"""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(TD3Critic, self).__init__()
        
        # Q1网络
        q1_layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            q1_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        q1_layers.append(nn.Linear(input_dim, 1))
        self.q1_network = nn.Sequential(*q1_layers)
        
        # Q2网络
        q2_layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            q2_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        q2_layers.append(nn.Linear(input_dim, 1))
        self.q2_network = nn.Sequential(*q2_layers)
        
    def forward(self, state, action):
        """
        前向传播
        
        Returns:
            q1, q2: 两个Q值
        """
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_network(sa)
        q2 = self.q2_network(sa)
        return q1, q2
    
    def q1(self, state, action):
        """返回Q1值"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1_network(sa)


class ReplayBuffer:
    """经验回放池"""
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.next_states = np.zeros((max_size, state_dim))
        self.rewards = np.zeros((max_size, 1))
        self.dones = np.zeros((max_size, 1))
        
    def add(self, state, action, next_state, reward, done):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        """采样批次"""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.dones[ind])
        )


class TD3Agent:
    """
    TD3代理 - 低层决策
    处理连续动作空间的资源分配
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        buffer_size=1000000,
        batch_size=256
    ):
        """
        初始化TD3代理
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度（连续）
            max_action: 最大动作值
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            tau: 软更新系数
            policy_noise: 目标策略平滑噪声
            noise_clip: 噪声裁剪
            policy_delay: 延迟策略更新
            buffer_size: 经验回放池大小
            batch_size: 批次大小
        """
        self.device = config.DEVICE
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        
        # 创建网络
        self.actor = TD3Actor(
            state_dim,
            action_dim,
            max_action,
            config.TD3_ACTOR_HIDDEN_DIMS
        ).to(self.device)
        
        self.actor_target = TD3Actor(
            state_dim,
            action_dim,
            max_action,
            config.TD3_ACTOR_HIDDEN_DIMS
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = TD3Critic(
            state_dim,
            action_dim,
            config.TD3_CRITIC_HIDDEN_DIMS
        ).to(self.device)
        
        self.critic_target = TD3Critic(
            state_dim,
            action_dim,
            config.TD3_CRITIC_HIDDEN_DIMS
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验回放池
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        
        # 训练计数器
        self.total_it = 0
        
        # 训练统计
        self.training_stats = {
            'critic_loss': [],
            'actor_loss': []
        }
        
    def select_action(self, state, noise=0.1):
        """
        选择动作
        
        Args:
            state: 状态
            noise: 探索噪声标准差
        
        Returns:
            action: 动作数组
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # 添加探索噪声
        if noise > 0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def update(self):
        """更新网络"""
        if self.replay_buffer.size < self.batch_size:
            return None
        
        self.total_it += 1
        
        # 从回放池采样
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        with torch.no_grad():
            # 目标策略平滑：添加裁剪噪声
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = (
                self.actor_target(next_states) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # 计算目标Q值：取两个Q网络的最小值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 更新Critic
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟更新Actor
        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            # 计算Actor损失
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        # 记录统计
        self.training_stats['critic_loss'].append(critic_loss.item())
        if actor_loss is not None:
            self.training_stats['actor_loss'].append(actor_loss.item())
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None
        }
    
    def _soft_update(self, source, target):
        """软更新目标网络"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
            'training_stats': self.training_stats
        }, path)
        print(f"TD3 model saved to {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint['total_it']
        self.training_stats = checkpoint['training_stats']
        print(f"TD3 model loaded from {path}")


if __name__ == "__main__":
    # 测试TD3代理
    print("Testing TD3 Agent...")
    
    state_dim = 128
    action_dim = 10  # 10个边缘服务器的资源分配比例
    max_action = 1.0
    
    agent = TD3Agent(state_dim, action_dim, max_action)
    
    # 测试动作选择
    dummy_state = np.random.randn(state_dim)
    action = agent.select_action(dummy_state, noise=0.1)
    print(f"Selected action shape: {action.shape}")
    print(f"Action range: [{action.min():.4f}, {action.max():.4f}]")
    
    # 添加经验到回放池
    for _ in range(1000):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        next_state = np.random.randn(state_dim)
        reward = np.random.randn()
        done = np.random.rand() < 0.1
        agent.replay_buffer.add(state, action, next_state, reward, done)
    
    # 测试更新
    stats = agent.update()
    print(f"\nTraining stats:")
    if stats:
        for key, value in stats.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
    
    print("\nAll tests passed!")
