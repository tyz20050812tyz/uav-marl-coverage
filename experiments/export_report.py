"""汇报素材生成脚本。

从训练日志和模型生成：
1. 代表性 episode 的 GIF 动画
2. 实验对比图表的高清 PNG
3. 核心数据表格 CSV

Usage:
    python experiments/export_report.py                          # 导出所有素材
    python experiments/export_report.py --algo maddpg            # 仅导出指定算法 GIF
    python experiments/export_report.py --no_gif                 # 仅导出图表和数据
"""

import argparse
import json
import logging
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

from env.simple_spread_wrapper import SimpleSpreadWrapper
from agents.maddpg_agent import MADDPGAgent
from agents.rs_maddpg_agent import RSMADDPGAgent

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_gif(algo: str, model_path: str, output_path: str,
               num_agents: int = 3, max_frames: int = 50,
               fps: int = 5, seed: int = 42):
    """导出一个 episode 的 GIF 动画。

    Args:
        algo: 算法名称 ('maddpg' 或 'rs_maddpg')
        model_path: 模型文件路径
        output_path: GIF 输出路径
        num_agents: 智能体数量
        max_frames: 最大帧数
        fps: 帧率
        seed: 随机种子
    """
    try:
        import imageio
    except ImportError:
        logger.warning("  [WARN] imageio 未安装，跳过 GIF 导出。安装: pip install imageio")
        return False

    np.random.seed(seed)

    env = SimpleSpreadWrapper(num_agents=num_agents)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    coverage_radius = env.world_size * 0.12
    safe_distance = env.world_size * 0.1

    # 加载模型
    if algo == 'maddpg':
        agent = MADDPGAgent(num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim)
    elif algo == 'rs_maddpg':
        agent = RSMADDPGAgent(
            num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
            coverage_radius=coverage_radius, safe_distance=safe_distance,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    if not os.path.exists(model_path):
        logger.info(f"  [SKIP] 模型不存在: {model_path}")
        env.close()
        return False

    agent.load(model_path)
    logger.info(f"  加载模型: {model_path}")

    # 录制一个 episode
    frames = []
    obs, _ = env.reset()
    if hasattr(agent, 'reset_noise'):
        agent.reset_noise()

    step = 0
    while step < max_frames:
        # 渲染当前帧
        frame = env.render()

        # 叠加覆盖圈和智能体标签
        try:
            import cv2
            state = env.get_world_state()
            agent_pos = state['agent_positions']
            landmark_pos = state['landmark_positions']

            h, w = frame.shape[:2]
            scale_x = w / env.world_size
            scale_y = h / env.world_size
            cx, cy = w // 2, h // 2

            # 绘制覆盖圈（每个智能体周围）
            r_c = int(coverage_radius * scale_x)
            for i, pos in enumerate(agent_pos):
                px = int(cx + pos[0] * scale_x)
                py = int(cy - pos[1] * scale_y)
                cv2.circle(frame, (px, py), r_c, (0, 255, 100), 1)
                cv2.putText(frame, f'A{i}', (px + 5, py - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 210, 255), 1)

            # 绘制目标点标签
            for j, pos in enumerate(landmark_pos):
                px = int(cx + pos[0] * scale_x)
                py = int(cy - pos[1] * scale_y)
                cv2.putText(frame, f'L{j}', (px + 5, py - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

            # 添加标题
            cv2.putText(frame, f'{algo.upper()} | Step {step}',
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)
        except Exception:
            pass  # OpenCV 叠加失败不阻塞

        frames.append(frame)

        # 选择动作（无噪声）
        actions = {}
        for name in obs.keys():
            actions[name] = agent.act(name, obs[name], add_noise=False)

        next_obs, _, terms, truncs, _ = env.step(actions)
        dones = {name: bool(terms[name]) or bool(truncs[name])
                for name in obs.keys()}
        obs = next_obs
        step += 1

        if all(dones.values()):
            break

    # 保存 GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    logger.info(f"  GIF 已保存: {output_path} ({len(frames)} 帧, {fps} fps)")

    env.close()
    return True


def export_chart_png(log_dir: str, export_dir: str):
    """从日志导出高清图表 PNG。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  [WARN] matplotlib 未安装，跳过图表导出")
        return

    os.makedirs(export_dir, exist_ok=True)
    logs_found = 0

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('_logs.json'):
            continue

        log_path = os.path.join(log_dir, fname)
        with open(log_path) as f:
            logs = json.load(f)

        if not logs:
            continue

        algo_name = fname.replace('_logs.json', '').replace('_seed42', '')

        # 提取指标
        episodes = [l['episode'] for l in logs]
        metrics_names = ['avg_reward', 'coverage_rate', 'collision_count', 'completion_steps']
        metric_labels = ['平均奖励', '目标覆盖率', '碰撞次数', '完成步数']

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{algo_name} 训练曲线', fontsize=14)

        for idx, (metric, label) in enumerate(zip(metrics_names, metric_labels)):
            ax = axes[idx // 2][idx % 2]
            values = [l.get(metric, 0.0) for l in logs]
            ax.plot(episodes, values, linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(export_dir, f'{algo_name}_curves.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  图表已保存: {output_path}")
        logs_found += 1

    if logs_found == 0:
        logger.warning("  [WARN] 未找到日志文件")


def export_data_csv(log_dir: str, export_dir: str):
    """导出核心数据表格 CSV。"""
    import csv

    os.makedirs(export_dir, exist_ok=True)
    rows = []

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('_logs.json'):
            continue

        log_path = os.path.join(log_dir, fname)
        with open(log_path) as f:
            logs = json.load(f)

        if not logs:
            continue

        algo_name = fname.replace('_logs.json', '').replace('_seed42', '')
        final = logs[-1]

        rows.append({
            '算法': algo_name,
            '最终Episode': final['episode'],
            '平均奖励': f"{final['avg_reward']:.2f}",
            '覆盖率': f"{final['coverage_rate']:.1%}",
            '碰撞次数': f"{final['collision_count']:.2f}",
            '完成步数': f"{final['completion_steps']:.1f}",
            '冗余覆盖率': f"{final.get('redundancy_rate', 0):.1%}",
            '平均最小距离': f"{final.get('avg_min_distance', 0):.3f}",
        })

    if rows:
        output_path = os.path.join(export_dir, 'results_summary.csv')
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  CSV 已保存: {output_path} ({len(rows)} 条记录)")
    else:
        logger.warning("  [WARN] 未找到日志数据")


def main():
    parser = argparse.ArgumentParser(description='汇报素材生成')
    parser.add_argument('--algo', type=str, default='all',
                       help='导出 GIF 的算法 (maddpg/rs_maddpg/all)')
    parser.add_argument('--no_gif', action='store_true', help='跳过 GIF 导出')
    parser.add_argument('--seed', type=int, default=42, help='模型种子')
    parser.add_argument('--num_agents', type=int, default=3, help='智能体数量')
    args = parser.parse_args()

    log_dir = os.path.join(PROJECT_ROOT, 'outputs', 'logs')
    model_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models')
    export_dir = os.path.join(PROJECT_ROOT, 'outputs', 'exports')

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    logger.info("=== 汇报素材生成 ===")

    # 1. GIF 动画
    if not args.no_gif:
        logger.info("\n--- GIF 动画 ---")
        if args.algo in ('maddpg', 'all'):
            model_path = os.path.join(model_dir, f'maddpg_seed{args.seed}_final.pt')
            export_gif('maddpg', model_path,
                      os.path.join(export_dir, 'maddpg_demo.gif'),
                      num_agents=args.num_agents, seed=args.seed)
        if args.algo in ('rs_maddpg', 'all'):
            model_path = os.path.join(model_dir, f'rs_maddpg_seed{args.seed}_final.pt')
            export_gif('rs_maddpg', model_path,
                      os.path.join(export_dir, 'rs_maddpg_demo.gif'),
                      num_agents=args.num_agents, seed=args.seed)

    # 2. PNG 图表
    logger.info("\n--- PNG 图表 ---")
    export_chart_png(log_dir, export_dir)

    # 3. CSV 数据表
    logger.info("\n--- CSV 数据表 ---")
    export_data_csv(log_dir, export_dir)

    logger.info(f"\n=== 素材导出完成: {export_dir} ===")


if __name__ == '__main__':
    main()
