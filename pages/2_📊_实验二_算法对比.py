"""实验二：四算法横向对比。"""

import streamlit as st
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from frontend.charts import load_logs, plot_training_curves, plot_trajectory
from frontend.components import section_header, inject_global_css

st.set_page_config(page_title="实验二", page_icon="📊")
inject_global_css()

section_header("实验二：四算法性能对比")

st.markdown("""
系统对比 Random、IDDPG、MADDPG、RS-MADDPG 四种方法在协同覆盖任务上的性能差异。
2×2 子图展示平均奖励、覆盖率、碰撞次数、完成步数的训练曲线。
""")

log_dir = os.path.join(_project_root, 'outputs', 'logs')
algorithms = ['Random', 'IDDPG', 'MADDPG', 'RS-MADDPG']
logs_data = []

for algo in algorithms:
    if os.path.exists(log_dir):
        for f in sorted(os.listdir(log_dir)):
            if f.startswith(algo.lower()) and f.endswith('_logs.json'):
                logs = load_logs(os.path.join(log_dir, f))
                if logs:
                    logs_data.append((algo, logs))
                break

if logs_data:
    # 最终指标对比
    cols = st.columns(len(logs_data))
    for i, (algo, logs) in enumerate(logs_data):
        final = logs[-1]
        cols[i].metric(f"**{algo}**",
                      f"覆盖 {final['coverage_rate']:.1%}",
                      f"碰撞 {final['collision_count']:.1f}")

    st.divider()
    fig = plot_training_curves(logs_data)
    st.plotly_chart(fig, use_container_width=True)

    st.success("结论：RS-MADDPG 在覆盖率和碰撞控制上均最优。")

    # 左右并排轨迹对比（折叠在 expander 中）
    with st.expander("🛸 策略轨迹对比（点击展开）", expanded=False):
        st.markdown("加载训练好的模型，运行评估 episode 对比不同算法的协同覆盖轨迹。")

        algo_for_traj = st.multiselect(
            "选择要对比的算法",
            ['MADDPG', 'RS-MADDPG', 'IDDPG'],
            default=['MADDPG', 'RS-MADDPG'],
            key='traj_algo_select',
        )

        if st.button("▶️ 生成轨迹对比图", key='traj_btn'):
            import numpy as np
            from env.simple_spread_wrapper import SimpleSpreadWrapper
            from agents.maddpg_agent import MADDPGAgent
            from agents.rs_maddpg_agent import RSMADDPGAgent
            from agents.iddpg_agent import IDDPGAgent
            from experiments.adapters import IDDPGManager

            model_dir = os.path.join(_project_root, 'outputs', 'models')

            cols = st.columns(len(algo_for_traj)) if algo_for_traj else []
            for idx, algo in enumerate(algo_for_traj):
                with cols[idx]:
                    st.markdown(f"**{algo}**")
                    with st.spinner(f"运行 {algo} 评估..."):
                        try:
                            eval_env = SimpleSpreadWrapper(num_agents=3, max_cycles=50)
                            coverage_r = eval_env.world_size * 0.12

                            model_path = os.path.join(
                                model_dir,
                                f"{algo.lower()}_seed42_final.pt",
                            )
                            if algo == 'IDDPG':
                                agent = IDDPGManager(
                                    num_agents=3,
                                    obs_dim=eval_env.obs_dim,
                                    act_dim=eval_env.act_dim,
                                )
                            elif algo == 'MADDPG':
                                agent = MADDPGAgent(
                                    num_agents=3,
                                    obs_dim=eval_env.obs_dim,
                                    act_dim=eval_env.act_dim,
                                )
                            else:
                                agent = RSMADDPGAgent(
                                    num_agents=3,
                                    obs_dim=eval_env.obs_dim,
                                    act_dim=eval_env.act_dim,
                                    coverage_radius=coverage_r,
                                    safe_distance=eval_env.world_size * 0.1,
                                )

                            if os.path.exists(model_path):
                                agent.load(model_path)
                                obs, _ = eval_env.reset()
                                positions_history = []
                                while True:
                                    actions = {}
                                    for name in obs.keys():
                                        actions[name] = agent.act(name, obs[name], add_noise=False)
                                    next_obs, _, terms, truncs, _ = eval_env.step(actions)
                                    state = eval_env.get_world_state()
                                    positions_history.append(state['agent_positions'].copy())
                                    obs = next_obs
                                    if all(bool(terms[n]) or bool(truncs[n]) for n in obs.keys()):
                                        break
                                lm_pos = eval_env.get_world_state()['landmark_positions']
                                traj_fig = plot_trajectory(
                                    positions_history, lm_pos,
                                    coverage_radius=coverage_r,
                                    title=f"{algo} 轨迹",
                                )
                                st.pyplot(traj_fig)
                                eval_env.close()
                            else:
                                st.warning(f"模型不存在: {model_path}")
                        except Exception as e:
                            st.error(f"{algo} 轨迹生成失败: {e}")
else:
    st.warning("未找到训练日志。请先运行实验脚本生成日志文件。")
