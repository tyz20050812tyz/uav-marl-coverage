"""实验四：不同智能体数量下的算法泛化性。

测试 MADDPG 和 RS-MADDPG 在 N=3/4/5 不同规模任务中的表现。
每种规模单独从头训练，对比覆盖率、碰撞次数的变化趋势。

Usage:
    python experiments/exp4_generalization.py                          # 快速验证: 5000 episodes, 1 seed, N=3/4/5
    python experiments/exp4_generalization.py --episodes 10000         # 自定义
    python experiments/exp4_generalization.py --scales 3,4             # 仅 N=3,4
    python experiments/exp4_generalization.py --algorithms maddpg      # 仅 MADDPG
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

from env.simple_spread_wrapper import SimpleSpreadWrapper
from agents.maddpg_agent import MADDPGAgent
from agents.rs_maddpg_agent import RSMADDPGAgent
from experiments.trainer import Trainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_agent(algo: str, num_agents: int, obs_dim: int, act_dim: int,
                 coverage_radius: float, safe_distance: float,
                 total_episodes: int, device=None):
    """创建指定类型的智能体。"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if algo == 'maddpg':
        return MADDPGAgent(
            num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
            actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01,
            policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5,
            buffer_capacity=1000000, device=device,
        )
    elif algo == 'rs_maddpg':
        return RSMADDPGAgent(
            num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
            actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01,
            policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5,
            buffer_capacity=1000000,
            coverage_radius=coverage_radius,
            safe_distance=safe_distance,
            lambda_a=0.5, lambda_r_max=0.3, lambda_s_max=0.5,
            total_episodes=total_episodes,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def run_scale(algo: str, n: int, total_episodes: int, seed: int,
              log_dir: str, model_dir: str) -> list:
    """训练指定规模和算法。"""
    # max_cycles 随 N 线性增长
    max_cycles = 50 + (n - 3) * 20  # N=3→50, N=4→70, N=5→90

    logger.info(f"\n{'='*60}")
    logger.info(f"  {algo.upper()} | N={n} | max_cycles={max_cycles} | episodes={total_episodes}")
    logger.info(f"{'='*60}")

    env = SimpleSpreadWrapper(num_agents=n, max_cycles=max_cycles)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    coverage_radius = env.world_size * 0.12
    safe_distance = env.world_size * 0.1

    agent = create_agent(
        algo=algo, num_agents=n,
        obs_dim=obs_dim, act_dim=act_dim,
        coverage_radius=coverage_radius,
        safe_distance=safe_distance,
        total_episodes=total_episodes,
    )

    # 日志和模型命名包含 N
    agent_type = f'{algo}_n{n}'

    trainer = Trainer(
        env_wrapper=env, agent=agent, agent_type=agent_type,
        eval_interval=max(100, total_episodes // 20),
        eval_episodes=5,
        log_dir=log_dir, model_dir=model_dir,
        checkpoint_interval=max(1000, total_episodes // 5),
        seed=seed,
    )

    logs = trainer.train(
        num_episodes=total_episodes,
        batch_size=1024,
        buffer_warmup=1024,
    )

    env.close()
    return logs


def main():
    parser = argparse.ArgumentParser(description='实验四：泛化实验')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='训练 episode 数（默认 5000 快速验证）')
    parser.add_argument('--seeds', type=str, default='42',
                       help='随机种子，逗号分隔')
    parser.add_argument('--scales', type=str, default='3,4,5',
                       help='智能体规模，逗号分隔')
    parser.add_argument('--algorithms', type=str, default='maddpg,rs_maddpg',
                       help='算法，逗号分隔')
    args = parser.parse_args()

    scales = [int(s.strip()) for s in args.scales.split(',')]
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    algorithms = [a.strip() for a in args.algorithms.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    logger.info(f"=== 实验四：泛化实验 ===")
    logger.info(f"  算法: {algorithms}")
    logger.info(f"  规模: N={scales}")
    logger.info(f"  种子: {seeds}")
    logger.info(f"  Episodes: {args.episodes}")

    log_dir = os.path.join(PROJECT_ROOT, 'outputs', 'logs')
    model_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    for algo in algorithms:
        for n in scales:
            for seed in seeds:
                try:
                    run_scale(
                        algo=algo, n=n,
                        total_episodes=args.episodes,
                        seed=seed,
                        log_dir=log_dir, model_dir=model_dir,
                    )
                except Exception as e:
                    logger.warning(f"  [ERROR] {algo} N={n} seed={seed} 失败: {e}")
                    import traceback
                    traceback.print_exc()

    logger.info("\n=== 实验四完成 ===")
    logger.info(f"日志目录: {log_dir}")


if __name__ == '__main__':
    main()
