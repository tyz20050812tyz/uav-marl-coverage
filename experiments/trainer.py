"""通用训练器模块。

提供统一的训练循环，支持所有算法类型（Random/IDDPG/MADDPG/RS-MADDPG）。
自动记录训练日志、保存 checkpoint、执行定期评估。
支持 Ctrl+C 中断恢复：中断时自动保存 checkpoint 和日志，下次可从断点继续。
"""

import json
import logging
import os
import signal
import time
import numpy as np
import torch
from typing import Dict, Optional, Callable
from env.simple_spread_wrapper import SimpleSpreadWrapper
from utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """通用训练器，支持 MADDPG 和 RS-MADDPG 算法。"""

    # 类级别中断标记（用于信号处理）
    _interrupted = False

    @classmethod
    def _signal_handler(cls, signum, frame):
        """信号处理器：捕获 SIGINT/SIGTERM，设置中断标记。"""
        logger.info("\n[Trainer] 收到中断信号 (Ctrl+C)，将在当前 episode 结束后保存并退出...")
        cls._interrupted = True

    def __init__(self, env_wrapper: SimpleSpreadWrapper,
                 agent,  # MADDPGAgent or RSMADDPGAgent
                 agent_type: str = 'maddpg',
                 eval_interval: int = 500,
                 eval_episodes: int = 10,
                 log_dir: str = 'outputs/logs',
                 model_dir: str = 'outputs/models',
                 checkpoint_interval: int = 1000,
                 coverage_radius: Optional[float] = None,
                 coverage_radius_ratio: float = 0.12,
                 use_wandb: bool = False,
                 wandb_project: str = 'uav-marl',
                 seed: int = 42,
                 resume_from: Optional[str] = None):
        """
        初始化训练器。

        Args:
            env_wrapper: 环境封装
            agent: 智能体（MADDPGAgent 或 RSMADDPGAgent）
            agent_type: 算法类型标识（用于日志）
            eval_interval: 评估间隔（episodes）
            eval_episodes: 每次评估的 episode 数
            log_dir: 日志输出目录
            model_dir: 模型保存目录
            checkpoint_interval: checkpoint 保存间隔
            coverage_radius: 覆盖率半径（None 时使用 ratio 自动计算）
            coverage_radius_ratio: 覆盖率半径占环境边长的比例（默认 0.12）
            use_wandb: 是否启用 wandb 日志
            wandb_project: wandb 项目名
            seed: 随机种子
            resume_from: 恢复训练的 checkpoint 路径（None 表示从头训练）
        """
        self.env = env_wrapper
        self.agent = agent
        self.agent_type = agent_type
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.checkpoint_interval = checkpoint_interval
        self.seed = seed

        # 恢复训练状态
        self._resume_episode = 0
        if resume_from is not None and os.path.exists(resume_from):
            self._load_checkpoint(resume_from)
            logger.info("[Trainer] 从 checkpoint 恢复: %s (start_episode=%d)",
                        resume_from, self._resume_episode)
        elif resume_from is not None:
            logger.warning("[Trainer] 指定的 checkpoint 不存在: %s，将从头开始训练", resume_from)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # W&B 可选集成
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    config={
                        'agent_type': agent_type,
                        'num_agents': env_wrapper.num_agents,
                        'obs_dim': env_wrapper.obs_dim,
                        'act_dim': env_wrapper.act_dim,
                        'seed': seed,
                    },
                )
            except Exception:
                self.use_wandb = False

        # 训练日志
        self.logs = []

        # 实时可视化状态（供 Streamlit 前端轮询读取）
        self.latest_world_state = None       # 最新世界状态（用于实时 2D 渲染）
        self.latest_eval_episode = 0         # 最新评估 episode 编号
        self.latest_episode_history = []     # 最新评估 episode 的轨迹历史

        # 覆盖率计算参数（优先使用显式值，否则按比例自动计算）
        if coverage_radius is not None:
            self.coverage_radius = coverage_radius
        else:
            self.coverage_radius = env_wrapper.world_size * coverage_radius_ratio

    def train(self, num_episodes: int,
              batch_size: int = 1024,
              buffer_warmup: int = 1024,
              progress_callback: Optional[Callable] = None) -> Dict:
        """
        执行完整训练流程。

        支持 Ctrl+C 中断恢复：中断时自动保存 checkpoint 和日志，
        下次可通过 resume_from 参数继续训练。

        Args:
            num_episodes: 训练 episode 总数
            batch_size: 每次更新的批次大小
            buffer_warmup: 回放池预热大小（需先积累足够数据再开始更新）
            progress_callback: 进度回调 (episode, metrics, world_state) -> None

        Returns:
            训练日志列表
        """
        # 注册中断信号处理器（仅主线程支持；Streamlit 后台线程忽略）
        signals_registered = False
        try:
            old_sigint = signal.signal(signal.SIGINT, Trainer._signal_handler)
            old_sigterm = signal.signal(signal.SIGTERM, Trainer._signal_handler)
            signals_registered = True
        except ValueError:
            old_sigint = signal.SIG_DFL
            old_sigterm = signal.SIG_DFL
        Trainer._interrupted = False

        try:
            return self._train_loop(num_episodes, batch_size, buffer_warmup,
                                    progress_callback)
        finally:
            # 只在主线程注册成功时恢复。Streamlit 后台线程不能调用 signal.signal。
            if signals_registered:
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)
            Trainer._interrupted = False

    def _train_loop(self, num_episodes: int,
                    batch_size: int = 1024,
                    buffer_warmup: int = 1024,
                    progress_callback: Optional[Callable] = None) -> Dict:
        """内部训练循环。"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        start_episode = self._resume_episode
        logger.info("[Trainer] 开始训练: %s, episodes=%d→%d, seed=%d",
                    self.agent_type, start_episode, num_episodes, self.seed)
        t_start = time.time()

        if start_episode >= num_episodes:
            logger.info("[Trainer] 所有 episode 已完成 (resume_episode=%d >= %d)，跳过训练",
                        start_episode, num_episodes)
            return self.logs

        total_steps = 0

        for episode in range(start_episode, num_episodes):
            # 检查中断信号
            if Trainer._interrupted:
                logger.info("[Trainer] 中断信号已触发，在 episode %d 后保存并退出", episode)
                self._save_interrupted_state(episode)
                break

            # RS-MADDPG: 设置当前 episode（用于权重调度）
            if hasattr(self.agent, 'set_episode'):
                self.agent.set_episode(episode)
            if hasattr(self.agent, 'reset_episode'):
                self.agent.reset_episode()

            # 重置环境
            obs, _ = self.env.reset()
            if hasattr(self.agent, 'reset_noise'):
                self.agent.reset_noise()

            ep_reward = 0.0
            ep_steps = 0
            agent_positions_history = []

            while True:
                # 选择动作
                actions = {}
                for name in obs.keys():
                    # 对于 MADDPG/RS-MADDPG 使用 act()，Random/IDDPG 直接调用
                    if hasattr(self.agent, 'act'):
                        actions[name] = self.agent.act(name, obs[name], add_noise=True)
                    else:
                        actions[name] = obs[name]  # fallback (should not happen)

                # 环境步进
                next_obs, env_rewards, terms, truncs, _ = self.env.step(actions)

                # RS-MADDPG: 计算塑形奖励
                if hasattr(self.agent, 'compute_shaped_reward'):
                    world_state = self.env.get_world_state()
                    rewards = self.agent.compute_shaped_reward(
                        env_rewards,
                        world_state['agent_positions'],
                        world_state['landmark_positions'],
                    )
                else:
                    rewards = env_rewards

                # 记录位置历史
                if hasattr(self.env, 'get_world_state'):
                    state = self.env.get_world_state()
                    agent_positions_history.append(state['agent_positions'].copy())

                # 存储 transition
                dones = {name: bool(terms[name]) or bool(truncs[name])
                        for name in obs.keys()}
                if hasattr(self.agent, 'buffer'):
                    self.agent.buffer.push(obs, actions, rewards, next_obs, dones)

                ep_reward += sum(rewards.values())
                ep_steps += 1
                total_steps += 1

                obs = next_obs
                done = all(dones.values())
                if done:
                    break

            # 捕获最新世界状态（供前端实时可视化）
            if hasattr(self.env, 'get_world_state'):
                self.latest_world_state = self.env.get_world_state()

            # 训练更新（buffer 足够时）
            if hasattr(self.agent, 'buffer') and \
               hasattr(self.agent, 'update') and \
               self.agent.buffer.is_ready(batch_size):
                for _ in range(ep_steps):  # 每步做一次更新
                    batch = self.agent.buffer.sample(batch_size, self.agent.device)
                    update_info = self.agent.update(batch)

            # 定期评估
            if (episode + 1) % self.eval_interval == 0 or episode == 0:
                eval_metrics = self.evaluate()
                log_entry = {
                    'episode': episode + 1,
                    'total_steps': total_steps,
                    'ep_reward': ep_reward,
                    **eval_metrics,
                }
                self.logs.append(log_entry)

                logger.info("  Episode %d/%d | Reward: %.2f | Coverage: %.1%% | Collisions: %.1f",
                            episode+1, num_episodes, ep_reward,
                            eval_metrics['coverage_rate'] * 100,
                            eval_metrics['collision_count'])

                # W&B 日志
                if self.use_wandb and self.wandb_run:
                    self.wandb_run.log(log_entry)

                # 进度回调（含世界状态快照供前端可视化）
                if progress_callback:
                    eval_state = self.latest_world_state if hasattr(self, 'latest_world_state') else None
                    progress_callback(episode + 1, log_entry, eval_state)

            # 保存 checkpoint
            if (episode + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(episode + 1)

            # 保存每隔 N 步的日志
            if (episode + 1) % 100 == 0:
                self._save_logs()

        # 训练完成
        elapsed = time.time() - t_start
        logger.info("[Trainer] 训练完成, 耗时: %.1fs", elapsed)

        # 最终评估
        final_metrics = self.evaluate()
        logger.info("[Trainer] 最终覆盖率: %.1%%, 碰撞: %.1f",
                    final_metrics['coverage_rate'] * 100,
                    final_metrics['collision_count'])

        # 保存最终模型
        self.save_checkpoint(num_episodes, final=True)

        # 保存日志
        self._save_logs()

        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

        return self.logs

    def evaluate(self) -> Dict[str, float]:
        """
        评估当前策略（无噪声）。

        Returns:
            指标 dict
        """
        metrics, trajectory = Trainer.evaluate_agent(
            self.env, self.agent, self.eval_episodes, self.coverage_radius,
            capture_trajectory=True,
        )
        if trajectory:
            self.latest_episode_history = trajectory
        return metrics

    @staticmethod
    def evaluate_agent(env, agent, eval_episodes: int = 10,
                       coverage_radius: float = 0.24,
                       capture_trajectory: bool = False) -> tuple:
        """
        静态评估方法：评估任意 agent 在指定环境中的表现（无噪声）。

        与 Trainer.evaluate() 共享逻辑，但接受显式参数，
        方便外部脚本（如 exp1）直接调用。

        Args:
            env: 环境封装
            agent: 智能体实例
            eval_episodes: 评估 episode 数
            coverage_radius: 覆盖率计算半径
            capture_trajectory: 是否捕获最后一个 episode 的轨迹数据

        Returns:
            (指标 dict, 轨迹数据 list or None)
            轨迹数据格式: [(N, 2) agent_positions, ...] 每个元素是一帧的智能体位置
        """
        total_reward = 0.0
        total_coverage = 0.0
        total_collisions = 0.0
        total_redundancy = 0.0
        total_min_dist = 0.0
        total_completion_steps = 0.0
        captured_trajectory = None

        for ep_idx in range(eval_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            steps = 0
            position_history = []
            is_last_ep = capture_trajectory and (ep_idx == eval_episodes - 1)

            while True:
                actions = {}
                for name in obs.keys():
                    if hasattr(agent, 'act'):
                        actions[name] = agent.act(name, obs[name], add_noise=False)
                    else:
                        actions[name] = np.ones(5) * 0.5  # fallback

                next_obs, env_rewards, terms, truncs, _ = env.step(actions)

                if hasattr(agent, 'compute_shaped_reward'):
                    state = env.get_world_state()
                    rewards = agent.compute_shaped_reward(
                        env_rewards,
                        state['agent_positions'],
                        state['landmark_positions'],
                    )
                    position_history.append(state['agent_positions'].copy())
                else:
                    rewards = env_rewards
                    if hasattr(env, 'get_world_state'):
                        state = env.get_world_state()
                        position_history.append(state['agent_positions'].copy())

                ep_reward += sum(rewards.values())
                steps += 1

                dones = {name: bool(terms[name]) or bool(truncs[name])
                        for name in obs.keys()}
                obs = next_obs
                if all(dones.values()):
                    break

            total_reward += ep_reward

            # 捕获最后一个评估 episode 的轨迹
            if is_last_ep and position_history:
                captured_trajectory = position_history

            # 计算最终步的指标
            if position_history:
                final_state = env.get_world_state()
                metrics = compute_metrics(
                    final_state['agent_positions'],
                    final_state['landmark_positions'],
                    coverage_radius,
                )
                total_coverage += metrics['coverage_rate']
                total_collisions += metrics['collision_count']
                total_redundancy += metrics['redundancy_rate']
                total_min_dist += metrics['avg_min_distance']

                # 完成步数
                from utils.metrics import compute_completion_steps
                completion = compute_completion_steps(
                    position_history,
                    final_state['landmark_positions'],
                    coverage_radius,
                )
                total_completion_steps += completion

        n = eval_episodes
        metrics = {
            'avg_reward': total_reward / n,
            'coverage_rate': total_coverage / n,
            'collision_count': total_collisions / n,
            'avg_min_distance': total_min_dist / n,
            'redundancy_rate': total_redundancy / n,
            'completion_steps': total_completion_steps / n,
        }
        return metrics, captured_trajectory

    def save_checkpoint(self, episode: int, final: bool = False):
        """保存模型 checkpoint。"""
        if not hasattr(self.agent, 'save'):
            return
        suffix = 'final' if final else f'ep{episode}'
        path = os.path.join(
            self.model_dir,
            f'{self.agent_type}_seed{self.seed}_{suffix}.pt',
        )
        self.agent.save(path)
        if final:
            logger.info("[Trainer] 最终模型已保存: %s", path)

    def _save_logs(self):
        """保存训练日志到 JSON 文件。"""
        path = os.path.join(
            self.log_dir,
            f'{self.agent_type}_seed{self.seed}_logs.json',
        )
        with open(path, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def _save_interrupted_state(self, episode: int):
        """中断时保存 checkpoint、日志和恢复状态。"""
        logger.info("[Trainer] 正在保存中断状态 (episode=%d)...", episode)
        if hasattr(self.agent, 'save'):
            path = os.path.join(
                self.model_dir,
                f'{self.agent_type}_seed{self.seed}_interrupted.pt',
            )
            # 将当前 episode 注入 agent 状态以便恢复
            if hasattr(self.agent, '_current_episode'):
                saved_ep = self.agent._current_episode
                self.agent._current_episode = episode
            try:
                self.agent.save(path)
                logger.info("[Trainer] 中断 checkpoint 已保存: %s", path)
            finally:
                if hasattr(self.agent, '_current_episode'):
                    self.agent._current_episode = saved_ep
        self._save_logs()
        # 保存恢复元信息
        resume_info_path = os.path.join(
            self.log_dir,
            f'{self.agent_type}_seed{self.seed}_resume_info.json',
        )
        with open(resume_info_path, 'w') as f:
            json.dump({'last_episode': episode, 'agent_type': self.agent_type,
                        'seed': self.seed}, f, indent=2)

    def _load_checkpoint(self, path: str):
        """从 checkpoint 加载模型和恢复训练状态。

        支持标准 checkpoint 和中断 checkpoint 两种格式。
        """
        if not hasattr(self.agent, 'load'):
            logger.warning("[Trainer] agent 不支持 load，无法恢复")
            return
        self.agent.load(path)
        # 尝试读取恢复元信息
        resume_info_path = os.path.join(
            self.log_dir,
            f'{self.agent_type}_seed{self.seed}_resume_info.json',
        )
        if os.path.exists(resume_info_path):
            with open(resume_info_path, 'r') as f:
                info = json.load(f)
            self._resume_episode = info.get('last_episode', 0)
            logger.info("[Trainer] 从 resume_info 恢复: last_episode=%d", self._resume_episode)
        # 恢复已有日志
        logs_path = os.path.join(
            self.log_dir,
            f'{self.agent_type}_seed{self.seed}_logs.json',
        )
        if os.path.exists(logs_path):
            with open(logs_path, 'r') as f:
                self.logs = json.load(f)
            logger.info("[Trainer] 已加载 %d 条历史日志", len(self.logs))
