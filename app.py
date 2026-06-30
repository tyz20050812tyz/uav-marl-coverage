"""无人机协同覆盖训练控制台。

主页面负责快速配置、启动训练、展示实时评估快照和最终结果。
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import sys
import time
import threading
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import pygame
pygame.display.init()

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.maddpg_agent import MADDPGAgent
from agents.rs_maddpg_agent import RSMADDPGAgent
from env.simple_spread_wrapper import SimpleSpreadWrapper
from experiments.adapters import IDDPGManager, RandomManager
from experiments.trainer import Trainer
from frontend.charts import plot_training_curves, plot_trajectory
from frontend.components import inject_global_css, section_header
from frontend.render_2d import render_frame


st.set_page_config(page_title="无人机协同覆盖训练控制台", page_icon="UAV", layout="wide")
inject_global_css()


def init_state():
    defaults = {
        'stage': 'config',
        'training_logs': [],
        'trainer': None,
        'env': None,
        'agent': None,
        'latest_eval_state': None,
        'latest_metrics': None,
        'reward_history': [],
        'coverage_history': [],
        '_eval_queue': [],
        'training_thread': None,
        'training_error': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_runtime():
    for key in [
        'trainer', 'env', 'agent', 'latest_eval_state', 'latest_metrics',
        'reward_history', 'coverage_history', '_eval_queue',
        'training_thread', 'training_error',
    ]:
        st.session_state[key] = [] if key in {'reward_history', 'coverage_history', '_eval_queue'} else None


def max_cycles_for(num_agents: int) -> int:
    return {3: 50, 4: 70, 5: 90}.get(num_agents, 50)


def build_agent(algo: str, env: SimpleSpreadWrapper):
    common = dict(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        actor_lr=st.session_state.actor_lr,
        critic_lr=st.session_state.critic_lr,
        gamma=st.session_state.gamma,
    )
    if algo == 'RS-MADDPG':
        return RSMADDPGAgent(
            **common,
            coverage_radius=env.world_size * st.session_state.coverage_ratio,
            safe_distance=env.world_size * st.session_state.safe_ratio,
            use_weight_scheduling=st.session_state.use_weight_scheduling,
            total_episodes=st.session_state.num_episodes,
        )
    if algo == 'MADDPG':
        return MADDPGAgent(**common)
    if algo == 'IDDPG':
        return IDDPGManager(**common)
    return RandomManager(num_agents=env.num_agents, act_dim=env.act_dim)


def start_training():
    num_episodes = st.session_state.num_episodes
    batch_size = st.session_state.batch_size
    buffer_warmup = st.session_state.buffer_warmup
    env = SimpleSpreadWrapper(
        num_agents=st.session_state.num_agents,
        max_cycles=max_cycles_for(st.session_state.num_agents),
    )
    agent = build_agent(st.session_state.algo, env)
    trainer = Trainer(
        env,
        agent,
        agent_type=st.session_state.algo,
        eval_interval=st.session_state.eval_interval,
        eval_episodes=st.session_state.eval_episodes,
        seed=st.session_state.seed,
        coverage_radius=env.world_size * st.session_state.coverage_ratio,
        use_wandb=st.session_state.use_wandb,
    )

    st.session_state.env = env
    st.session_state.agent = agent
    st.session_state.trainer = trainer
    queue = st.session_state._eval_queue

    def on_eval(episode, metrics, world_state):
        queue.append((episode, metrics, world_state))

    def run_training():
        try:
            trainer.train(
                num_episodes=num_episodes,
                batch_size=batch_size,
                buffer_warmup=buffer_warmup,
                progress_callback=on_eval,
            )
            trainer._completed = True
        except Exception as exc:
            trainer._failed = repr(exc)

    thread = threading.Thread(target=run_training, daemon=True)
    st.session_state.training_thread = thread
    thread.start()


def sync_training_events():
    queue = st.session_state.get('_eval_queue') or []
    while queue:
        episode, metrics, world_state = queue.pop(0)
        st.session_state.latest_metrics = (episode, metrics)
        st.session_state.reward_history.append((episode, metrics.get('avg_reward', 0.0)))
        st.session_state.coverage_history.append((episode, metrics.get('coverage_rate', 0.0)))
        if world_state is not None:
            st.session_state.latest_eval_state = world_state


def metric_strip(metrics):
    cols = st.columns(5)
    cols[0].metric("覆盖率", f"{metrics.get('coverage_rate', 0):.1%}")
    cols[1].metric("碰撞", f"{metrics.get('collision_count', 0):.1f}")
    cols[2].metric("平均奖励", f"{metrics.get('avg_reward', 0):.2f}")
    cols[3].metric("完成步数", f"{metrics.get('completion_steps', 0):.0f}")
    cols[4].metric("冗余率", f"{metrics.get('redundancy_rate', 0):.1%}")


def plot_live_history():
    fig = go.Figure()
    if st.session_state.reward_history:
        eps, rewards = zip(*st.session_state.reward_history[-80:])
        fig.add_trace(go.Scatter(
            x=eps, y=rewards, mode='lines+markers', name='平均奖励',
            line=dict(color='#f4c430', width=2),
        ))
    if st.session_state.coverage_history:
        eps, coverage = zip(*st.session_state.coverage_history[-80:])
        fig.add_trace(go.Scatter(
            x=eps, y=[v * 100 for v in coverage], mode='lines+markers',
            name='覆盖率(%)', yaxis='y2', line=dict(color='#2ec4b6', width=2),
        ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=30),
        paper_bgcolor='#101827',
        plot_bgcolor='#101827',
        font=dict(color='#c9d4e5'),
        hovermode='x unified',
        legend=dict(orientation='h', y=1.1),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', title='奖励'),
        yaxis2=dict(overlaying='y', side='right', range=[0, 100], title='覆盖率'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Episode'),
    )
    st.plotly_chart(fig, width='stretch')


def preview_frame(num_agents: int):
    env = SimpleSpreadWrapper(num_agents=num_agents, max_cycles=8)
    obs, _ = env.reset()
    for _ in range(4):
        actions = {name: np.random.random(env.act_dim).astype(np.float32) for name in obs}
        obs, _, terms, truncs, _ = env.step(actions)
        if all(bool(terms[n]) or bool(truncs[n]) for n in obs):
            break
    state = env.get_world_state()
    frame = render_frame(
        env,
        state,
        coverage_radius=env.world_size * st.session_state.get('coverage_ratio', 0.12),
        safe_distance=env.world_size * st.session_state.get('safe_ratio', 0.1),
    )
    env.close()
    return frame


init_state()

if st.session_state.stage == 'config':
    st.title("无人机协同覆盖训练控制台")
    st.caption("参数配置、训练监控、结果分析集中在一个轻量页面，适合快速观察算法是否真的在变好。")

    with st.sidebar:
        section_header("实验配置")
        st.session_state.algo = st.selectbox("算法", ['RS-MADDPG', 'MADDPG', 'IDDPG', 'Random'], index=0)
        st.session_state.num_agents = st.selectbox("智能体数量", [3, 4, 5], index=0)
        mode = st.radio("训练模式", ["快速观察", "正式实验"], horizontal=True)
        default_episodes = 3000 if mode == "快速观察" else 20000
        default_eval = 100 if mode == "快速观察" else 500
        st.session_state.num_episodes = st.slider("Episodes", 100, 50000, default_episodes, 100)
        st.session_state.eval_interval = st.slider("评估间隔", 20, 2000, default_eval, 20)
        st.session_state.eval_episodes = st.slider("每次评估局数", 1, 10, 3, 1)

        section_header("训练参数")
        st.session_state.actor_lr = st.number_input("Actor LR", 1e-5, 1e-2, 1e-3, format="%.5f")
        st.session_state.critic_lr = st.number_input("Critic LR", 1e-5, 1e-2, 1e-3, format="%.5f")
        st.session_state.gamma = st.slider("Gamma", 0.80, 0.99, 0.95, 0.01)
        st.session_state.batch_size = st.select_slider("Batch Size", [64, 128, 256, 512, 1024], value=256)
        st.session_state.buffer_warmup = st.select_slider("预热样本", [64, 128, 256, 512, 1024], value=256)
        st.session_state.seed = st.number_input("随机种子", 0, 9999, 42)

        section_header("RS 奖励")
        st.session_state.coverage_ratio = st.slider("覆盖半径比例", 0.06, 0.20, 0.12, 0.01)
        st.session_state.safe_ratio = st.slider("安全距离比例", 0.05, 0.18, 0.10, 0.01)
        st.session_state.use_weight_scheduling = st.toggle("动态奖励权重", value=True)
        st.session_state.use_wandb = st.toggle("W&B 日志", value=False)

        if st.button("开始训练", type="primary", width='stretch'):
            reset_runtime()
            st.session_state.stage = 'training'
            st.rerun()

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("任务预览")
        st.image(preview_frame(st.session_state.num_agents), width='stretch')
    with right:
        st.subheader("方案核对")
        st.markdown(
            """
            - 环境：PettingZoo `simple_spread_v3`
            - 动作：5 维连续动作强度
            - 指标：奖励、覆盖率、碰撞、完成步数、冗余率
            - 改进：目标分配、冗余惩罚、安全距离、动态权重
            """
        )
        st.info("建议先用快速观察模式确认曲线方向，再切到正式实验跑长训练。")

elif st.session_state.stage == 'training':
    if st.session_state.trainer is None:
        start_training()

    trainer = st.session_state.trainer
    sync_training_events()

    if getattr(trainer, '_failed', None):
        st.session_state.training_error = trainer._failed
        st.session_state.stage = 'results'
        st.rerun()

    if getattr(trainer, '_completed', False):
        st.session_state.training_logs = trainer.logs
        st.session_state.stage = 'results'
        st.rerun()

    st.title(f"{st.session_state.algo} 训练中")
    episode = 0
    metrics = {}
    if st.session_state.latest_metrics is not None:
        episode, metrics = st.session_state.latest_metrics
    st.progress(
        min(episode / max(st.session_state.num_episodes, 1), 1.0),
        text=f"Episode {episode} / {st.session_state.num_episodes}",
    )

    top_left, top_right = st.columns([1.15, 1])
    with top_left:
        st.subheader("最新评估态势")
        if st.session_state.latest_eval_state is not None:
            frame = render_frame(
                st.session_state.env,
                st.session_state.latest_eval_state,
                coverage_radius=st.session_state.env.world_size * st.session_state.coverage_ratio,
                safe_distance=st.session_state.env.world_size * st.session_state.safe_ratio,
            )
            st.image(frame, width='stretch')
        else:
            st.info("训练已启动，等待第一次评估结果。")

    with top_right:
        st.subheader("关键指标")
        if metrics:
            metric_strip(metrics)
        else:
            st.info("评估完成后会显示覆盖率、碰撞和奖励。")
        st.subheader("实时趋势")
        plot_live_history()

    time.sleep(0.8)
    st.rerun()

elif st.session_state.stage == 'results':
    st.title("训练结果")

    if st.session_state.training_error:
        st.error(f"训练异常结束：{st.session_state.training_error}")
    else:
        logs = st.session_state.training_logs
        if logs:
            last = logs[-1]
            metric_strip(last)
            st.subheader("训练曲线")
            st.plotly_chart(plot_training_curves([(st.session_state.algo, logs)]), width='stretch')

            st.subheader("策略轨迹")
            if st.button("生成一次评估轨迹", type="primary"):
                env = st.session_state.env
                agent = st.session_state.agent
                obs, _ = env.reset()
                positions_history = []
                while True:
                    actions = {
                        name: agent.act(name, obs[name], add_noise=False)
                        if hasattr(agent, 'act') else np.ones(env.act_dim, dtype=np.float32) * 0.5
                        for name in obs
                    }
                    obs, _, terms, truncs, _ = env.step(actions)
                    positions_history.append(env.get_world_state()['agent_positions'].copy())
                    if all(bool(terms[n]) or bool(truncs[n]) for n in obs):
                        break
                landmarks = env.get_world_state()['landmark_positions']
                fig = plot_trajectory(
                    positions_history,
                    landmarks,
                    coverage_radius=env.world_size * st.session_state.coverage_ratio,
                    title=f"{st.session_state.algo} 协同覆盖轨迹",
                )
                st.pyplot(fig)

            st.success(f"模型与日志已保存到 `{PROJECT_ROOT / 'outputs'}`。")
        else:
            st.warning("还没有可展示的训练日志。")

    if st.button("返回配置", width='stretch'):
        if st.session_state.env is not None:
            st.session_state.env.close()
        reset_runtime()
        st.session_state.training_logs = []
        st.session_state.stage = 'config'
        st.rerun()
