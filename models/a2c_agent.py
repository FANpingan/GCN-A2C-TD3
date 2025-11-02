"""
A2C (Advantage Actor-Critic) 高层决策代理
负责离散决策：选择任务卸载到哪个边缘集群
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from config import config


class ActorNetwork(nn.Module):
    """Actor网络：输出动作概率分布"""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128]):
        super(ActorNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # 输出层：动作概率logits
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
        
        Returns:
            action_logits: 动作logits [batch_size, action_dim]
        """
        return self.network(state)
    
    def get_action_probs(self, state):
        """获取动作概率分布"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_action(self, state, deterministic=False):
        """
        采样动作
        
        Args:
            state: 状态
            deterministic: 是否确定性选择（测试时使用）
        
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            entropy: 熵值
        """
        probs = self.get_action_probs(state)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        # 计算log概率
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        
        # 计算熵（用于熵正则化）
        entropy = dist.entropy()
        
        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    """Critic网络：估计状态价值V(s)"""
    def __init__(self, state_dim, hidden_dims=[256, 128]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # 输出层：状态价值
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
        
        Returns:
            value: 状态价值 [batch_size, 1]
        """
        return self.network(state)


class A2CAgent:
    """
    A2C代理 - 高层决策
    使用优势函数和熵正则化
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        gcn_output_dim=64,
        lr=3e-4,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_steps=5
    ):
        """
        初始化A2C代理
        
        Args:
            state_dim: 状态维度（GCN输出 + 其他状态）
            action_dim: 动作维度（边缘集群数量）
            gcn_output_dim: GCN输出维度
            lr: 学习率
            gamma: 折扣因子
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
            n_steps: n-step returns
        """
        self.device = config.DEVICE
        self.action_dim = action_dim
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        # 创建Actor和Critic网络
        self.actor = ActorNetwork(
            state_dim,
            action_dim,
            config.A2C_ACTOR_HIDDEN_DIMS
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim,
            config.A2C_CRITIC_HIDDEN_DIMS
        ).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 训练统计
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_loss': []
        }
        
    def select_action(self, state, deterministic=False):
        """
        选择动作
        
        Args:
            state: 状态张量
            deterministic: 是否确定性选择
        
        Returns:
            action: 动作索引
            log_prob: 对数概率
            value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_returns(self, rewards, dones, next_value):
        """
        计算n-step returns
        
        Args:
            rewards: 奖励列表
            dones: 终止标志列表
            next_value: 下一状态的价值
        
        Returns:
            returns: 回报列表
        """
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def update(self, states, actions, log_probs, rewards, dones, next_state):
        """
        更新网络
        
        Args:
            states: 状态列表
            actions: 动作列表
            log_probs: 对数概率列表
            rewards: 奖励列表
            dones: 终止标志列表
            next_state: 下一状态
        """
        # 转换为张量
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        # 计算returns
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.critic(next_state_tensor).item()
        
        returns = self.compute_returns(rewards, dones, next_value)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 计算当前值
        values = self.critic(states_tensor).squeeze()
        
        # 计算优势函数
        advantages = returns_tensor - values.detach()
        
        # Actor损失
        _, new_log_probs, entropy = self.actor.get_action(states_tensor)
        actor_loss = -(new_log_probs * advantages).mean()
        
        # Critic损失
        critic_loss = F.mse_loss(values, returns_tensor)
        
        # 熵损失（鼓励探索）
        entropy_loss = -entropy.mean()
        
        # 总损失
        total_loss = (
            actor_loss + 
            self.value_coef * critic_loss + 
            self.entropy_coef * entropy_loss
        )
        
        # 更新网络
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm
        )
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # 记录统计信息
        self.training_stats['actor_loss'].append(actor_loss.item())
        self.training_stats['critic_loss'].append(critic_loss.item())
        self.training_stats['entropy'].append(entropy.mean().item())
        self.training_stats['total_loss'].append(total_loss.item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
        print(f"A2C model saved to {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_stats = checkpoint['training_stats']
        print(f"A2C model loaded from {path}")


if __name__ == "__main__":
    # 测试A2C代理
    print("Testing A2C Agent...")
    
    state_dim = 128
    action_dim = 5  # 5个边缘集群
    
    agent = A2CAgent(state_dim, action_dim)
    
    # 测试动作选择
    dummy_state = np.random.randn(state_dim)
    action, log_prob, value = agent.select_action(dummy_state)
    print(f"Selected action: {action}")
    print(f"Log probability: {log_prob:.4f}")
    print(f"State value: {value:.4f}")
    
    # 测试更新
    states = [np.random.randn(state_dim) for _ in range(10)]
    actions = [np.random.randint(0, action_dim) for _ in range(10)]
    log_probs = [np.random.randn() for _ in range(10)]
    rewards = [np.random.randn() for _ in range(10)]
    dones = [False] * 9 + [True]
    next_state = np.random.randn(state_dim)
    
    stats = agent.update(states, actions, log_probs, rewards, dones, next_state)
    print(f"\nTraining stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nAll tests passed!")
