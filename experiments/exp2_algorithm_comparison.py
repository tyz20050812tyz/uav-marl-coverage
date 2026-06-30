"""实验二：四算法性能对比。

分别训练 Random、IDDPG、MADDPG、RS-MADDPG 四种算法，
对比训练曲线（平均奖励、覆盖率、碰撞次数、完成步数）。

Usage:
    python experiments/exp2_algorithm_comparison.py                        # 快速验证: 5000 episodes, 1 seed
    python experiments/exp2_algorithm_comparison.py --full                 # 完整实验: 20000 episodes, 3 seeds
    python experiments/exp2_algorithm_comparison.py --episodes 10000       # 自定义 episode 数
    python experiments/exp2_algorithm_comparison.py --algorithms maddpg,rs_maddpg  # 仅训练指定算法
    python experiments/exp2_algorithm_comparison.py --seed 123             # 指定种子
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
from experiments.adapters import IDDPGManager, RandomManager
from experiments.trainer import Trainer
from utils.config_loader import load_config, get_training_config, get_network_config, get_rs_maddpg_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_agent(algo: str, num_agents: int, obs_dim: int, act_dim: int,
                 coverage_radius: float, safe_distance: float,
                 total_episodes: int, seed: int, device=None,
                 config: dict = None):
    """创建指定类型的智能体。
    
    优先使用 YAML 配置文件中的默认值，fallback 到硬编码默认值。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 从配置中提取网络和 RS-MADDPG 参数
    net_cfg = get_network_config(config) if config else {}
    rs_cfg = get_rs_maddpg_config(config) if config else {}
    
    actor_lr = net_cfg.get('actor_lr', 1e-3)
    critic_lr = net_cfg.get('critic_lr', 1e-3)
    gamma = net_cfg.get('gamma', 0.95)
    tau = net_cfg.get('tau', 0.01)
    lambda_a = rs_cfg.get('assignment', {}).get('lambda_a', 0.5)
    lambda_r_max = rs_cfg.get('redundancy', {}).get('lambda_r_max', 0.3)
    lambda_s_max = rs_cfg.get('safety', {}).get('lambda_s_max', 0.5)

    if algo == 'random':
        return RandomManager(num_agents=num_agents, act_dim=act_dim)

    elif algo == 'iddpg':
        return IDDPGManager(
            num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
            actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau,
            buffer_capacity=1000000, device=device,
        )

    elif algo == 'maddpg':
        return MADDPGAgent(
            num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
            actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau,
            policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5,
            buffer_capacity=1000000, device=device,
        )

    elif algo == 'rs_maddpg':
        return RSMADDPGAgent(
            num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
            actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau,
            policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5,
            buffer_capacity=1000000,
            coverage_radius=coverage_radius,
            safe_distance=safe_distance,
            lambda_a=lambda_a, lambda_r_max=lambda_r_max, lambda_s_max=lambda_s_max,
            total_episodes=total_episodes,
            device=device,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def run_single(algo: str, num_agents: int, obs_dim: int, act_dim: int,
               total_episodes: int, seed: int, coverage_radius: float,
               safe_distance: float, log_dir: str, model_dir: str,
               config: dict = None) -> list:
    """运行单个算法的训练，返回日志列表。"""
    logger.info(f"\n{'='*60}")
    logger.info(f"  训练 {algo.upper()} (seed={seed}, episodes={total_episodes})")
    logger.info(f"{'='*60}")

    # 创建环境
    env = SimpleSpreadWrapper(num_agents=num_agents)

    # 创建智能体
    agent = create_agent(
        algo=algo, num_agents=num_agents,
        obs_dim=obs_dim, act_dim=act_dim,
        coverage_radius=coverage_radius,
        safe_distance=safe_distance,
        total_episodes=total_episodes,
        seed=seed,
        config=config,
    )

    # 对于 Random，只需评估，无需训练
    if algo == 'random':
        logger.info("  Random agent 无需训练，直接评估...")
        trainer = Trainer(
            env_wrapper=env, agent=agent, agent_type=algo,
            eval_interval=1, eval_episodes=10,
            log_dir=log_dir, model_dir=model_dir,
            checkpoint_interval=total_episodes + 1,
            seed=seed,
        )
        # 进行一次评估并记录
        eval_result = trainer.evaluate()
        log_entry = {
            'episode': 1,
            'total_steps': 0,
            'ep_reward': eval_result['avg_reward'],
            **eval_result,
        }
        trainer.logs = [log_entry]
        trainer._save_logs()
        env.close()
        return trainer.logs

    # 创建训练器
    trainer = Trainer(
        env_wrapper=env, agent=agent, agent_type=algo,
        eval_interval=max(100, total_episodes // 20),  # 约 20 个评估点
        eval_episodes=5,
        log_dir=log_dir, model_dir=model_dir,
        checkpoint_interval=max(1000, total_episodes // 5),
        seed=seed,
    )

    # 训练
    logs = trainer.train(
        num_episodes=total_episodes,
        batch_size=1024,
        buffer_warmup=1024,
    )

    env.close()
    return logs


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='实验二：四算法性能对比')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='训练 episode 数（默认 5000 快速验证）')
    parser.add_argument('--full', action='store_true',
                       help='完整实验：20000 episodes × 3 seeds')
    parser.add_argument('--algorithms', type=str, default='random,iddpg,maddpg,rs_maddpg',
                       help='要训练的算法，逗号分隔')
    parser.add_argument('--seeds', type=str, default='42',
                       help='随机种子，逗号分隔')
    parser.add_argument('--num_agents', type=int, default=3, help='智能体数量')
    parser.add_argument('--config', type=str, default=None,
                       help='YAML 配置文件路径（默认自动查找 configs/default.yaml）')
    args = parser.parse_args()

    # 加载 YAML 配置
    try:
        cfg = load_config(args.config)
        logger.info("[config] 已加载配置文件: %s", args.config or 'configs/default.yaml')
    except FileNotFoundError:
        logger.warning("[config] 配置文件未找到，使用硬编码默认值")
        cfg = None

    # 完整实验模式
    if args.full:
        args.episodes = 20000
        args.seeds = '42,123,456'

    algorithms = [a.strip() for a in args.algorithms.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    total_episodes = args.episodes

    logger.info(f"=== 实验二：四算法性能对比 (N={args.num_agents}) ===")
    logger.info(f"  算法: {algorithms}")
    logger.info(f"  种子: {seeds}")
    logger.info(f"  Episodes: {total_episodes}")

    # 输出目录
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

    # 逐算法、逐种子训练
    for algo in algorithms:
        for seed in seeds:
            try:
                run_single(
                    algo=algo, num_agents=args.num_agents,
                    obs_dim=obs_dim, act_dim=act_dim,
                    total_episodes=total_episodes,
                    seed=seed,
                    coverage_radius=coverage_radius,
                    safe_distance=safe_distance,
                    log_dir=log_dir, model_dir=model_dir,
                    config=cfg,
                )
            except Exception as e:
                logger.warning(f"  [ERROR] {algo} seed={seed} 失败: {e}")
                import traceback
                traceback.print_exc()

    logger.info("\n=== 实验二完成 ===")
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"模型目录: {model_dir}")


if __name__ == '__main__':
    main()
