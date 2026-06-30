"""实验一：随机策略 vs 训练后策略对比。

评估 Random 策略、MADDPG、RS-MADDPG 三种方法在覆盖率、碰撞次数、
完成步数上的差异。若预训练模型不存在则自动进行快速训练。

Usage:
    python experiments/exp1_random_vs_trained.py                    # 默认: N=3, 评估 100 episodes
    python experiments/exp1_random_vs_trained.py --num_agents 4     # N=4
    python experiments/exp1_random_vs_trained.py --quick_train 2000 # 快速训练 2000 episodes 再评估
"""

import argparse
import json
import logging
import os
import sys
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

from env.simple_spread_wrapper import SimpleSpreadWrapper
from agents.maddpg_agent import MADDPGAgent
from agents.rs_maddpg_agent import RSMADDPGAgent
from experiments.adapters import RandomManager
from experiments.trainer import Trainer
from utils.metrics import compute_metrics
from frontend.charts import plot_bar_comparison

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate_agent(env, agent, num_episodes: int = 100,
                   coverage_radius: float = 0.24) -> dict:
    """评估智能体（无噪声），返回各项平均指标。

    委托给 Trainer.evaluate_agent 统一实现，消除代码重复。
    """
    metrics, _ = Trainer.evaluate_agent(env, agent, num_episodes, coverage_radius)
    return metrics


def quick_train(agent_type: str, num_episodes: int, seed: int,
                env: SimpleSpreadWrapper, obs_dim: int, act_dim: int,
                num_agents: int) -> tuple:
    """快速训练并返回 agent 和日志。"""
    log_dir = os.path.join(PROJECT_ROOT, 'outputs', 'logs')
    model_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if agent_type == 'maddpg':
        agent = MADDPGAgent(
            num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
            actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01,
            policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5,
        )
    elif agent_type == 'rs_maddpg':
        agent = RSMADDPGAgent(
            num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
            actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01,
            policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5,
            coverage_radius=env.world_size * 0.12,
            safe_distance=env.world_size * 0.1,
            lambda_a=0.5, lambda_r_max=0.3, lambda_s_max=0.5,
            total_episodes=num_episodes,
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    trainer = Trainer(
        env_wrapper=env, agent=agent, agent_type=agent_type,
        eval_interval=max(100, num_episodes // 10),
        eval_episodes=5,
        log_dir=log_dir, model_dir=model_dir,
        checkpoint_interval=num_episodes + 1,  # 仅在最终保存
        seed=seed,
    )
    logs = trainer.train(num_episodes=num_episodes, batch_size=1024, buffer_warmup=1024)
    return agent, logs


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='实验一：随机 vs 训练后策略对比')
    parser.add_argument('--num_agents', type=int, default=3, help='智能体数量')
    parser.add_argument('--eval_episodes', type=int, default=100, help='评估 episode 数')
    parser.add_argument('--quick_train', type=int, default=0,
                       help='快速训练 episode 数（0=跳过训练，仅加载已有模型）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    logger.info(f"=== 实验一：随机策略 vs 训练后策略对比 (N={args.num_agents}) ===")

    # 创建评估环境
    env = SimpleSpreadWrapper(num_agents=args.num_agents)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    coverage_radius = env.world_size * 0.12

    results = {}

    # 1. 评估 Random 策略
    logger.info("\n--- 评估 Random 策略 ---")
    random_agent = RandomManager(num_agents=args.num_agents, act_dim=act_dim)
    results['Random'] = evaluate_agent(env, random_agent, args.eval_episodes, coverage_radius)
    logger.info(f"  Random: 覆盖率={results['Random']['coverage_rate']:.1%}, "
                f"碰撞={results['Random']['collision_count']:.1f}, "
                f"完成步数={results['Random']['completion_steps']:.0f}")

    # 2. MADDPG
    logger.info("\n--- MADDPG ---")
    model_path = os.path.join(PROJECT_ROOT, 'outputs', 'models',
                             f'maddpg_seed{args.seed}_final.pt')
    if os.path.exists(model_path) and args.quick_train == 0:
        logger.info(f"  加载已有模型: {model_path}")
        maddpg_agent = MADDPGAgent(
            num_agents=args.num_agents, obs_dim=obs_dim, act_dim=act_dim,
        )
        maddpg_agent.load(model_path)
    else:
        episodes = args.quick_train if args.quick_train > 0 else 2000
        logger.info(f"  训练 {episodes} episodes...")
        train_env = SimpleSpreadWrapper(num_agents=args.num_agents)
        maddpg_agent, _ = quick_train(
            'maddpg', episodes, args.seed, train_env,
            obs_dim, act_dim, args.num_agents,
        )

    results['MADDPG'] = evaluate_agent(env, maddpg_agent, args.eval_episodes, coverage_radius)
    logger.info(f"  MADDPG: 覆盖率={results['MADDPG']['coverage_rate']:.1%}, "
                f"碰撞={results['MADDPG']['collision_count']:.1f}, "
                f"完成步数={results['MADDPG']['completion_steps']:.0f}")

    # 3. RS-MADDPG
    logger.info("\n--- RS-MADDPG ---")
    model_path = os.path.join(PROJECT_ROOT, 'outputs', 'models',
                             f'rs_maddpg_seed{args.seed}_final.pt')
    if os.path.exists(model_path) and args.quick_train == 0:
        logger.info(f"  加载已有模型: {model_path}")
        rs_agent = RSMADDPGAgent(
            num_agents=args.num_agents, obs_dim=obs_dim, act_dim=act_dim,
            coverage_radius=coverage_radius,
            safe_distance=env.world_size * 0.1,
        )
        rs_agent.load(model_path)
    else:
        episodes = args.quick_train if args.quick_train > 0 else 2000
        logger.info(f"  训练 {episodes} episodes...")
        train_env = SimpleSpreadWrapper(num_agents=args.num_agents)
        rs_agent, _ = quick_train(
            'rs_maddpg', episodes, args.seed, train_env,
            obs_dim, act_dim, args.num_agents,
        )

    results['RS-MADDPG'] = evaluate_agent(env, rs_agent, args.eval_episodes, coverage_radius)
    logger.info(f"  RS-MADDPG: 覆盖率={results['RS-MADDPG']['coverage_rate']:.1%}, "
                f"碰撞={results['RS-MADDPG']['collision_count']:.1f}, "
                f"完成步数={results['RS-MADDPG']['completion_steps']:.0f}")

    # 4. 打印汇总表格
    logger.info("\n" + "=" * 60)
    logger.info("对比汇总:")
    logger.info(f"{'算法':<12} {'覆盖率':>8} {'碰撞':>8} {'完成步数':>8} {'平均奖励':>10}")
    logger.info("-" * 60)
    for algo in ['Random', 'MADDPG', 'RS-MADDPG']:
        r = results[algo]
        logger.info(f"{algo:<12} {r['coverage_rate']:>7.1%} {r['collision_count']:>8.1f} "
                    f"{r['completion_steps']:>8.0f} {r['avg_reward']:>10.1f}")

    # 5. 保存结果 JSON
    output_path = os.path.join(PROJECT_ROOT, 'outputs', 'logs', 'exp1_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 转换 numpy 类型为 Python 原生类型
    serializable = {}
    for algo, metrics in results.items():
        serializable[algo] = {k: float(v) for k, v in metrics.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"\n结果已保存: {output_path}")

    # 6. 生成柱状图
    try:
        chart_data = {}
        for algo, metrics in results.items():
            chart_data[algo] = {
                '覆盖率': metrics['coverage_rate'],
                '完成步数': metrics['completion_steps'] / env.max_cycles,
            }
        fig = plot_bar_comparison(chart_data, '覆盖率', '实验一：随机 vs 训练后策略')
        chart_path = os.path.join(PROJECT_ROOT, 'outputs', 'exports', 'exp1_bar.png')
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        fig.write_image(chart_path)
        logger.info(f"图表已保存: {chart_path}")
    except Exception as e:
        logger.warning(f"图表生成失败: {e}")

    env.close()


if __name__ == '__main__':
    main()
