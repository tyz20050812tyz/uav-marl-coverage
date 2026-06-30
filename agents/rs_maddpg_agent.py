"""RS-MADDPG（Reward-Shaped MADDPG）智能体。

在 MADDPG 框架基础上集成四项奖励塑形机制：
1. 目标分配引导奖励（Assignment Guidance）
2. 冗余覆盖惩罚（Redundancy Penalty）
3. 安全距离约束（Safety Distance Constraint）
4. 动态奖励权重机制（Dynamic Weight Scheduling）

RS-MADDPG 继承 MADDPGAgent，扩展奖励计算逻辑。
"""

import numpy as np
import torch
from typing import Optional, Dict

from agents.maddpg_agent import MADDPGAgent
from rewards.assignment import AssignmentReward
from rewards.redundancy import RedundancyPenalty
from rewards.safety import SafetyPenalty
from rewards.weight_scheduler import WeightScheduler


class RSMADDPGAgent(MADDPGAgent):
    """奖励塑形 MADDPG 智能体管理器。"""

    def __init__(self, num_agents: int, obs_dim: int, act_dim: int = 5,
                 # MADDPG 基类参数
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 gamma: float = 0.95, tau: float = 0.01,
                 policy_delay: int = 2,
                 target_noise_std: float = 0.2,
                 target_noise_clip: float = 0.5,
                 buffer_capacity: int = 1000000,
                 # RS-MADDPG 额外参数
                 lambda_a: float = 0.5,
                 hysteresis_epsilon: float = 0.05,
                 lock_steps: int = 5,
                 coverage_radius: float = 0.24,
                 lambda_r_max: float = 0.3,
                 safe_distance: float = 0.2,
                 lambda_s_max: float = 0.5,
                 lambda_c_max: float = 0.5,
                 # 奖励模块开关（消融实验用）
                 use_assignment: bool = True,
                 use_redundancy: bool = True,
                 use_safety: bool = True,
                 use_collision: bool = True,
                 collision_threshold: float = 0.15,
                 # 动态权重调度
                 use_weight_scheduling: bool = False,
                 total_episodes: int = 20000,
                 device: Optional[torch.device] = None):
        """
        初始化 RS-MADDPG。

        Args:
            num_agents: 智能体数量
            obs_dim: 每个智能体的观测维度
            act_dim: 动作维度
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子
            tau: 软更新系数
            policy_delay: Actor 延迟更新步数
            target_noise_std: 目标策略噪声标准差
            target_noise_clip: 目标策略噪声截断值
            buffer_capacity: 回放池容量
            lambda_a: 分配奖励权重
            hysteresis_epsilon: 迟滞阈值
            lock_steps: 前 K 步锁定
            coverage_radius: 覆盖半径
            lambda_r_max: 冗余惩罚最大权重
            safe_distance: 安全距离阈值
            lambda_s_max: 安全距离惩罚最大权重
            lambda_c_max: 碰撞惩罚最大权重
            use_weight_scheduling: 是否启用动态权重
            total_episodes: 总训练 episode 数（用于权重调度）
            device: 计算设备
        """
        super().__init__(
            num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
            actor_lr=actor_lr, critic_lr=critic_lr,
            gamma=gamma, tau=tau,
            policy_delay=policy_delay,
            target_noise_std=target_noise_std,
            target_noise_clip=target_noise_clip,
            buffer_capacity=buffer_capacity,
            device=device,
        )

        # RS-MADDPG 奖励塑形模块（按开关条件创建）
        self.use_assignment = use_assignment
        self.use_redundancy = use_redundancy
        self.use_safety = use_safety
        self.use_collision = use_collision
        self.collision_threshold = collision_threshold

        if use_assignment:
            self.assignment_reward = AssignmentReward(
                lambda_a=lambda_a,
                hysteresis_epsilon=hysteresis_epsilon,
                lock_steps=lock_steps,
            )
        else:
            self.assignment_reward = None

        if use_redundancy:
            self.redundancy_penalty = RedundancyPenalty(
                coverage_radius=coverage_radius,
                lambda_r_max=lambda_r_max,
            )
        else:
            self.redundancy_penalty = None

        if use_safety:
            self.safety_penalty = SafetyPenalty(
                safe_distance=safe_distance,
                lambda_s_max=lambda_s_max,
            )
        else:
            self.safety_penalty = None

        if not use_collision:
            self.lambda_c_max = 0.0
            self._current_lambda_c = 0.0
        else:
            self.lambda_c_max = lambda_c_max
            self._current_lambda_c = lambda_c_max

        # 动态权重调度器
        self.use_weight_scheduling = use_weight_scheduling
        if use_weight_scheduling:
            self.weight_scheduler = WeightScheduler(total_episodes=total_episodes)
        else:
            self.weight_scheduler = None

        # 当前 episode 计数器（用于权重调度）
        self._current_episode = 0

        # 保存配置
        self._rs_config = {
            'lambda_a': lambda_a,
            'lambda_r_max': lambda_r_max,
            'lambda_s_max': lambda_s_max,
            'lambda_c_max': lambda_c_max,
        }

    def reset_episode(self):
        """每个 episode 开始时重置奖励模块状态。"""
        if self.assignment_reward is not None:
            self.assignment_reward.reset()

    def set_episode(self, episode: int):
        """设置当前 episode 编号（用于权重调度）。"""
        self._current_episode = episode
        if self.use_weight_scheduling and self.weight_scheduler is not None:
            weights = self.weight_scheduler.get_weights(
                episode,
                self._rs_config['lambda_s_max'],
                self._rs_config['lambda_r_max'],
                self._rs_config['lambda_a'],
                self._rs_config['lambda_c_max'],
            )
            self.redundancy_penalty.set_lambda(weights['lambda_r'])
            self.safety_penalty.set_lambda(weights['lambda_s'])
            self._current_lambda_c = weights['lambda_c']

    def compute_shaped_reward(self, env_rewards: Dict[str, float],
                              agent_positions: np.ndarray,
                              landmark_positions: np.ndarray) -> Dict[str, float]:
        """
        计算综合奖励：原始环境奖励 + RS-MADDPG 改进项。

        公式: R_i = R_env + R_assign + R_redund + R_safe + β·R_collision

        Args:
            env_rewards: 原始环境返回的奖励 {agent_name: reward}
            agent_positions: (N, 2) 智能体位置
            landmark_positions: (M, 2) 目标点位置

        Returns:
            {agent_name: total_reward}
        """
        # 在循环外计算各塑形项（只算一次，避免 N 倍冗余）
        assign_rewards = {}
        redund_penalties = {}
        safety_penalties = {}
        collision_penalties = {}

        if self.use_assignment and self.assignment_reward is not None:
            assign_rewards, _ = self.assignment_reward.compute(
                agent_positions, landmark_positions
            )

        if self.use_redundancy and self.redundancy_penalty is not None:
            redund_penalties, _ = self.redundancy_penalty.compute(
                agent_positions, landmark_positions
            )

        if self.use_safety and self.safety_penalty is not None:
            safety_penalties, _ = self.safety_penalty.compute(agent_positions)

        if self.use_collision:
            collision_penalties = self._compute_collision_penalty(
                agent_positions, self.collision_threshold
            )

        # 综合奖励: R_total = R_env + R_assign + R_redund + R_safe + β·R_collision
        total_rewards = {}
        for name in self.agent_names:
            r_total = env_rewards.get(name, 0.0)
            r_total += assign_rewards.get(name, 0.0)
            r_total += redund_penalties.get(name, 0.0)
            r_total += safety_penalties.get(name, 0.0)
            r_total += self._current_lambda_c * collision_penalties.get(name, 0.0)
            total_rewards[name] = r_total

        return total_rewards

    def _compute_collision_penalty(self, agent_positions: np.ndarray,
                                    collision_threshold: float = None) -> Dict[str, float]:
        """
        计算碰撞惩罚（基于 agent 间距离）。

        对每对距离低于阈值的 agent，按接近程度施加惩罚。

        Args:
            agent_positions: (N, 2) 智能体位置
            collision_threshold: 碰撞距离阈值（None 则使用实例默认值）

        Returns:
            {agent_name: penalty}（负值表示惩罚）
        """
        if collision_threshold is None:
            collision_threshold = self.collision_threshold
        N = agent_positions.shape[0]
        penalties = {f'agent_{i}': 0.0 for i in range(N)}

        for i in range(N):
            for k in range(i + 1, N):
                dist = np.linalg.norm(agent_positions[i] - agent_positions[k])
                if dist < collision_threshold:
                    # 越近惩罚越大，按距离反比
                    penalty = -(collision_threshold - dist) / collision_threshold
                    penalties[f'agent_{i}'] += penalty
                    penalties[f'agent_{k}'] += penalty

        return penalties

    def get_rs_info(self) -> Dict:
        """获取 RS-MADDPG 当前配置信息。"""
        return {
            'use_weight_scheduling': self.use_weight_scheduling,
            'use_assignment': self.use_assignment,
            'use_redundancy': self.use_redundancy,
            'use_safety': self.use_safety,
            'use_collision': self.use_collision,
            'episode': self._current_episode,
            'lambda_r': self.redundancy_penalty._current_lambda if self.redundancy_penalty is not None else 0.0,
            'lambda_s': self.safety_penalty._current_lambda if self.safety_penalty is not None else 0.0,
            **self._rs_config,
        }

    def save(self, path: str):
        """保存模型（含 RS-MADDPG 状态和模块开关配置）。"""
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
            'current_episode': self._current_episode,
            'rs_config': self._rs_config,
            'ablation_config': {
                'use_assignment': self.use_assignment,
                'use_redundancy': self.use_redundancy,
                'use_safety': self.use_safety,
                'use_collision': self.use_collision,
            },
        }
        torch.save(ckpt, path)

    def load(self, path: str):
        """加载模型。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        for name in self.agent_names:
            self.actors[name].load_state_dict(ckpt['actors'][name])
            self.target_actors[name].load_state_dict(ckpt['target_actors'][name])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        self._update_step = ckpt.get('update_step', 0)
        self._current_episode = ckpt.get('current_episode', 0)
