"""实验三：消融实验 — 各改进模块贡献度分析。

验证 RS-MADDPG 中每个改进模块的独立贡献。
五组实验：MADDPG → +Assignment → +Redundancy → +Safety → RS-MADDPG(full)

Usage:
    python experiments/exp3_ablation.py                          # 快速验证: 5000 episodes, 1 seed
    python experiments/exp3_ablation.py --episodes 10000 --seeds 42,123  # 自定义
    python experiments/exp3_ablation.py --groups maddpg,assign,full  # 仅训练指定组
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


class AblationMADDPGAgent(RSMADDPGAgent):
    """消融实验专用智能体：继承 RSMADDPGAgent，通过 use_* 开关控制模块组合。

    消除了与 RSMADDPGAgent 的代码重复，所有奖励塑形逻辑均复用父类实现。
    """
    pass


# 消融实验组定义
ABLATION_GROUPS = {
    'maddpg': {
        'label': 'MADDPG (baseline)',
        'use_assignment': False,
        'use_redundancy': False,
        'use_safety': False,
    },
    'assign': {
        'label': '+ Assignment',
        'use_assignment': True,
        'use_redundancy': False,
        'use_safety': False,
    },
    'ar': {
        'label': '+ Assignment + Redundancy',
        'use_assignment': True,
        'use_redundancy': True,
        'use_safety': False,
    },
    'ars': {
        'label': '+ Assignment + Redundancy + Safety',
        'use_assignment': True,
        'use_redundancy': True,
        'use_safety': True,
    },
    'full': {
        'label': 'RS-MADDPG (full) = ARS + Collision',
        'use_assignment': True,
        'use_redundancy': True,
        'use_safety': True,
        'use_collision': True,
    },
}


def run_ablation_group(group_key: str, group_config: dict,
                       num_agents: int, obs_dim: int, act_dim: int,
                       total_episodes: int, seed: int,
                       coverage_radius: float, safe_distance: float,
                       log_dir: str, model_dir: str) -> list:
    """运行一组消融实验。"""
    label = group_config['label']
    logger.info(f"\n{'='*60}")
    logger.info(f"  消融组: {label} (seed={seed}, episodes={total_episodes})")
    logger.info(f"{'='*60}")

    env = SimpleSpreadWrapper(num_agents=num_agents)

    agent = AblationMADDPGAgent(
        num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
        actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01,
        policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5,
        buffer_capacity=1000000,
        use_assignment=group_config['use_assignment'],
        use_redundancy=group_config['use_redundancy'],
        use_safety=group_config['use_safety'],
        use_collision=group_config.get('use_collision', False),
        coverage_radius=coverage_radius,
        safe_distance=safe_distance,
        lambda_a=0.5, lambda_r_max=0.3, lambda_s_max=0.5,
        lambda_c_max=0.5, collision_threshold=0.15,
    )

    # 日志前缀
    agent_type = f'ablation_{group_key}'

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
    parser = argparse.ArgumentParser(description='实验三：消融实验')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='训练 episode 数（默认 5000 快速验证）')
    parser.add_argument('--seeds', type=str, default='42',
                       help='随机种子，逗号分隔')
    parser.add_argument('--groups', type=str,
                       default='maddpg,assign,ar,ars,full',
                       help='消融组，逗号分隔: maddpg,assign,ar,ars,full')
    parser.add_argument('--num_agents', type=int, default=3, help='智能体数量')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    group_keys = [g.strip() for g in args.groups.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # 验证 group_keys（避免遍历中修改列表）
    valid_keys = [gk for gk in group_keys if gk in ABLATION_GROUPS]
    for gk in group_keys:
        if gk not in ABLATION_GROUPS:
            logger.warning(f"警告: 未知消融组 '{gk}'，跳过")
    group_keys = valid_keys

    logger.info(f"=== 实验三：消融实验 (N={args.num_agents}) ===")
    logger.info(f"  消融组: {[ABLATION_GROUPS[gk]['label'] for gk in group_keys]}")
    logger.info(f"  种子: {seeds}")
    logger.info(f"  Episodes: {args.episodes}")

    log_dir = os.path.join(PROJECT_ROOT, 'outputs', 'logs')
    model_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 环境参数
    env = SimpleSpreadWrapper(num_agents=args.num_agents)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    coverage_radius = env.world_size * 0.12
    safe_distance = env.world_size * 0.1
    env.close()

    for gk in group_keys:
        config = ABLATION_GROUPS[gk]
        for seed in seeds:
            try:
                run_ablation_group(
                    group_key=gk, group_config=config,
                    num_agents=args.num_agents,
                    obs_dim=obs_dim, act_dim=act_dim,
                    total_episodes=args.episodes,
                    seed=seed,
                    coverage_radius=coverage_radius,
                    safe_distance=safe_distance,
                    log_dir=log_dir, model_dir=model_dir,
                )
            except Exception as e:
                logger.warning(f"  [ERROR] {gk} seed={seed} 失败: {e}")
                import traceback
                traceback.print_exc()

    logger.info("\n=== 实验三完成 ===")
    logger.info(f"日志目录: {log_dir}")


if __name__ == '__main__':
    main()
