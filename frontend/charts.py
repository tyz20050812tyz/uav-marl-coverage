"""训练曲线和图表绘制模块。"""

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
from frontend.components import get_algo_color


def load_logs(log_path: str) -> List[Dict]:
    """加载 JSON 日志文件。"""
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_training_curves(logs_list: List[tuple], metrics: List[str] = None):
    """
    绘制多算法训练曲线对比。

    Args:
        logs_list: [(label, logs), ...] 每个算法的标签和日志
        metrics: 要绘制的指标列表，默认 ['avg_reward', 'coverage_rate', 'collision_count', 'completion_steps']
    """
    if metrics is None:
        metrics = ['avg_reward', 'coverage_rate', 'collision_count', 'completion_steps']

    metric_labels = {
        'avg_reward': '平均奖励',
        'coverage_rate': '目标覆盖率',
        'collision_count': '碰撞次数',
        'completion_steps': '完成步数',
        'redundancy_rate': '冗余覆盖率',
        'avg_min_distance': '平均最小距离',
    }

    n_metrics = len(metrics)
    cols = 2
    rows = (n_metrics + 1) // 2

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[metric_labels.get(m, m) for m in metrics])

    for label, logs in logs_list:
        if not logs:
            continue
        episodes = [l['episode'] for l in logs]
        color = get_algo_color(label)

        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            values = [l.get(metric, 0.0) for l in logs]

            fig.add_trace(
                go.Scatter(x=episodes, y=values, mode='lines',
                          name=label, line=dict(color=color, width=2),
                          showlegend=(idx == 0)),
                row=row, col=col,
            )

    fig.update_layout(height=300 * rows, hovermode='x unified',
                      margin=dict(l=40, r=40, t=40, b=40))
    fig.update_xaxes(title_text='Episode')
    return fig


def plot_bar_comparison(data: Dict[str, Dict[str, float]], metric: str,
                        title: str = ""):
    """
    绘制分组柱状图。

    Args:
        data: {group_name: {bar_label: value, ...}}
        metric: 指标名称
        title: 图表标题
    """
    groups = list(data.keys())
    bar_labels = list(data[groups[0]].keys()) if groups else []

    fig = go.Figure()
    for label in bar_labels:
        values = [data[g].get(label, 0.0) for g in groups]
        fig.add_trace(go.Bar(name=label, x=groups, y=values,
                             text=[f'{v:.3f}' for v in values],
                             textposition='outside'))

    fig.update_layout(title=title, barmode='group',
                      margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_trajectory(positions_history: List[np.ndarray],
                    landmark_positions: np.ndarray,
                    coverage_radius: float = 0.24,
                    title: str = "Trajectory"):
    """
    绘制智能体轨迹图（深色主题）。

    Args:
        positions_history: list of (N, 2) arrays
        landmark_positions: (M, 2) 目标点位置
        coverage_radius: 覆盖圈半径
        title: 标题
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title, color='#e0e0e0', fontsize=14)
    ax.grid(True, alpha=0.15, color='white')
    ax.tick_params(colors='#a8a8b8')
    for spine in ax.spines.values():
        spine.set_color('#0f3460')

    # 覆盖圈（在目标点周围）
    for j, pos in enumerate(landmark_positions):
        circle = plt.Circle(pos, coverage_radius, fill=False,
                           color='#00c864', alpha=0.5, linewidth=1, linestyle='--')
        ax.add_patch(circle)

    # 目标点
    for j, pos in enumerate(landmark_positions):
        ax.scatter(*pos, c='#ff6b6b', marker='x', s=100, linewidths=2,
                  label='目标点' if j == 0 else "", zorder=6)

    # 轨迹
    if positions_history:
        history = np.array(positions_history)  # (T, N, 2)
        N = history.shape[1]
        from frontend.components import get_agent_color
        for i in range(N):
            ax.plot(history[:, i, 0], history[:, i, 1],
                   color=get_agent_color(i), linewidth=1, alpha=0.6)
            ax.scatter(history[0, i, 0], history[0, i, 1],
                      color=get_agent_color(i), s=40, alpha=0.4)
            ax.scatter(history[-1, i, 0], history[-1, i, 1],
                      color=get_agent_color(i), s=80, zorder=5,
                      edgecolors='white', linewidths=1,
                      label=f'UAV{i+1}' if i == 0 else f'UAV{i+1}')

    ax.legend(loc='upper right', facecolor='#16213e',
             edgecolor='#0f3460', labelcolor='#c0c0d0', fontsize=9)
    fig.tight_layout()
    return fig
