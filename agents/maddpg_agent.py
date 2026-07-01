"""MADDPG 智能体模块。

集中式训练 + 分布式执行（CTDE）：
- 集中式 Critic 使用所有智能体的观测和动作
- 每个智能体独立 Actor 根据局部观测决策

集成 TD3 三项稳定性增强：
1. 截断双 Q 网络（Twin Critic，取 min）
2. 延迟策略更新（Critic 每 2 步，Actor 更新 1 次）
3. 目标策略平滑（target actor 输出加截断噪声）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List
import copy

from .base_agent import BaseAgent
from networks.actor import Actor
from networks.critic import TwinCritic
from utils.replay_buffer import ReplayBufferTensor
from utils.ou_noise import OUNoise


class MADDPGAgent(BaseAgent):
    """MADDPG 智能体管理器（包含 N 个 Agent 的 Actor 和 1 个集中式 TwinCritic）。"""

    def __init__(self, num_agents: int, obs_dim: int, act_dim: int = 5,
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 gamma: float = 0.95, tau: float = 0.01,
                 policy_delay: int = 2,
                 target_noise_std: float = 0.2,
                 target_noise_clip: float = 0.5,
                 buffer_capacity: int = 1000000,
                 device: Optional[torch.device] = None):
        """
        初始化 MADDPG。

        Args:
            num_agents: 智能体数量
            obs_dim: 每个智能体的观测维度
            act_dim: 每个智能体的动作维度
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子
            tau: 软更新系数
            policy_delay: Actor 延迟更新步数（TD3）
            target_noise_std: 目标策略平滑噪声标准差（TD3）
            target_noise_clip: 目标策略平滑噪声截断值（TD3）
            buffer_capacity: 回放池容量
            device: 计算设备
        """
        super().__init__(name='MADDPG')
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Agent names
        self.agent_names = [f'agent_{i}' for i in range(num_agents)]

        # 每个智能体的 Actor + Target Actor
        self.actors = nn.ModuleDict()
        self.target_actors = nn.ModuleDict()
        self.actor_optimizers = {}
        for name in self.agent_names:
            self.actors[name] = Actor(obs_dim, act_dim).to(self.device)
            self.target_actors[name] = copy.deepcopy(self.actors[name])
            self.actor_optimizers[name] = optim.Adam(
                self.actors[name].parameters(), lr=actor_lr
            )

        # 集中式双 Critic + Target Critic
        all_obs_dim = obs_dim * num_agents
        all_act_dim = act_dim * num_agents
        self.critic = TwinCritic(all_obs_dim, all_act_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # 经验回放池（所有智能体共享）
        self.buffer = ReplayBufferTensor(capacity=buffer_capacity)

        # OU 噪声（每个智能体独立）
        self.noises = {name: OUNoise(act_dim) for name in self.agent_names}
        self.noise_scale = 1.0

        # 训练计数器（用于延迟策略更新）
        self._update_step = 0

    def set_noise_scale(self, scale: float):
        """设置探索噪声缩放系数，用于训练后期逐步降低随机扰动。"""
        self.noise_scale = float(np.clip(scale, 0.0, 1.0))

    def act(self, agent_name: str, obs: np.ndarray,
            add_noise: bool = True) -> np.ndarray:
        """
        单个智能体根据局部观测选择动作。

        Args:
            agent_name: 智能体名称
            obs: (obs_dim,) 局部观测
            add_noise: 是否添加探索噪声

        Returns:
            (act_dim,) 动作
        """
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        self.actors[agent_name].eval()
        with torch.no_grad():
            action = self.actors[agent_name](
                obs_tensor.unsqueeze(0)
            ).squeeze(0).cpu().numpy()

        if add_noise:
            noise = self.noises[agent_name].sample() * self.noise_scale
            action = action + noise
            action = np.clip(action, 0.0, 1.0)

        return action.astype(np.float32)

    def act_all(self, observations: Dict[str, np.ndarray],
                add_noise: bool = True) -> Dict[str, np.ndarray]:
        """所有智能体一次性决策。"""
        return {
            name: self.act(name, observations[name], add_noise)
            for name in self.agent_names
        }

    def reset_noise(self):
        """重置所有智能体的 OU 噪声。"""
        for noise in self.noises.values():
            noise.reset()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行一次 MADDPG 更新（集中式 Critic + TD3 增强）。

        Args:
            batch: {'obs': {name: (B, obs_dim)}, 'actions': {name: (B, act_dim)},
                    'rewards': {name: (B, 1)}, 'next_obs': {name: (B, obs_dim)},
                    'dones': {name: (B, 1)}}

        Returns:
            {'critic_loss': float, 'actor_loss': float (mean)}
        """
        # 拼接所有智能体的观测和动作
        all_obs = torch.cat(
            [batch['obs'][name] for name in self.agent_names], dim=1
        )
        all_actions = torch.cat(
            [batch['actions'][name] for name in self.agent_names], dim=1
        )
        all_next_obs = torch.cat(
            [batch['next_obs'][name] for name in self.agent_names], dim=1
        )
        # 奖励取所有智能体的均值
        all_rewards = torch.cat(
            [batch['rewards'][name] for name in self.agent_names], dim=1
        ).mean(dim=1, keepdim=True)
        # Dones: 任一智能体 done 即认为 episode 结束
        all_dones = torch.cat(
            [batch['dones'][name] for name in self.agent_names], dim=1
        ).max(dim=1, keepdim=True)[0]

        # --- 目标策略平滑 + 计算 TD Target ---
        with torch.no_grad():
            next_actions_list = []
            for name in self.agent_names:
                next_act = self.target_actors[name](batch['next_obs'][name])
                # TD3: 目标策略平滑
                noise = torch.randn_like(next_act) * self.target_noise_std
                noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
                next_act = next_act + noise
                next_act = torch.clamp(next_act, 0.0, 1.0)
                next_actions_list.append(next_act)
            all_next_actions = torch.cat(next_actions_list, dim=1)

            # 双 Q 取 min
            target_q = self.target_critic.q_min(all_next_obs, all_next_actions)
            target_q = all_rewards + self.gamma * (1 - all_dones) * target_q

        # --- 更新 Critic（两个 Critic 同时更新）---
        qa, qb = self.critic.q_values(all_obs, all_actions)
        critic_loss = nn.MSELoss()(qa, target_q.detach()) + \
                      nn.MSELoss()(qb, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # --- 延迟策略更新：每 policy_delay 步更新一次 Actor ---
        actor_loss_mean = 0.0
        if self._update_step % self.policy_delay == 0:
            actor_losses = []
            for name in self.agent_names:
                pred_actions = self.actors[name](batch['obs'][name])

                # 构造 all_actions：其他 agent 用真实动作，当前 agent 用预测动作
                all_actions_pred = []
                for n in self.agent_names:
                    if n == name:
                        all_actions_pred.append(pred_actions)
                    else:
                        all_actions_pred.append(batch['actions'][n])
                all_actions_pred = torch.cat(all_actions_pred, dim=1)

                actor_loss = -self.critic.critic_A(all_obs, all_actions_pred).mean()
                actor_losses.append(actor_loss)

                self.actor_optimizers[name].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[name].step()

            actor_loss_mean = torch.stack(actor_losses).mean().item()

            # --- 软更新 Target 网络 ---
            self._soft_update_all()

        self._update_step += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss_mean,
        }

    def _soft_update_all(self):
        """软更新所有 target 网络。"""
        # Target Critics
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        # Target Actors
        for name in self.agent_names:
            for tp, sp in zip(
                self.target_actors[name].parameters(),
                self.actors[name].parameters()
            ):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path: str):
        """保存所有模型。"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            'actors': {name: self.actors[name].state_dict()
                       for name in self.agent_names},
            'target_actors': {name: self.target_actors[name].state_dict()
                             for name in self.agent_names},
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'update_step': self._update_step,
        }
        torch.save(ckpt, path)

    def load(self, path: str):
        """加载所有模型。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        for name in self.agent_names:
            self.actors[name].load_state_dict(ckpt['actors'][name])
            self.target_actors[name].load_state_dict(ckpt['target_actors'][name])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        self._update_step = ckpt.get('update_step', 0)
