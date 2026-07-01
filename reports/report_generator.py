"""Structured experiment summary generation.

The functions here intentionally do not call an LLM. They turn raw episode
metrics into a grounded JSON summary. LLM text generation can then consume
this summary without inventing facts.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


CORE_METRICS = [
    'avg_reward',
    'coverage_rate',
    'collision_count',
    'avg_min_distance',
    'redundancy_rate',
    'completion_steps',
]


def summarize_experiment(config: Dict[str, Any],
                         run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a structured report from one or more finished runs."""
    run_summaries = [summarize_run(run) for run in run_results]
    best = choose_best_run(run_summaries)
    comparisons = compare_runs(run_summaries)
    headline = build_headline(config, run_summaries, best)
    report_text = build_report_text(config, run_summaries, best, comparisons)

    return {
        'experiment_mode': config.get('experiment_mode', 'single'),
        'headline': headline,
        'grade': grade_experiment(best),
        'best_run': best,
        'runs': run_summaries,
        'comparisons': comparisons,
        'diagnosis': build_diagnosis(best, run_summaries),
        'recommendations': build_recommendations(config, best, run_summaries),
        'report_text': report_text,
    }


def summarize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    history = run.get('history') or []
    label = run.get('label') or run.get('algo') or 'Run'
    config = run.get('config') or {}
    best_metrics = run.get('best_metrics') or {}

    if not history:
        return {
            'id': run.get('id'),
            'label': label,
            'algo': run.get('algo'),
            'config': config,
            'episodes': 0,
            'score': 0.0,
            'final': {},
            'last_window': {},
            'stability': {},
            'trend': {},
            'metric_source': 'none',
            'best_metrics': best_metrics,
            'best_checkpoint': run.get('best_checkpoint'),
            'warnings': ['没有可用训练日志'],
        }

    metric_history, metric_source = build_metric_history(history)
    window_size = max(1, len(metric_history) // 10)
    last_window = metric_history[-window_size:]
    final = metric_history[-1]
    last_stats = {metric: stats(last_window, metric) for metric in CORE_METRICS}
    full_stats = {metric: stats(metric_history, metric) for metric in CORE_METRICS}
    trend = build_trend(metric_history)
    final_core = {metric: safe_float(final.get(metric)) for metric in CORE_METRICS}
    score = compute_score(last_stats, final_core)

    return {
        'id': run.get('id'),
        'label': label,
        'algo': run.get('algo'),
        'config': config,
        'episodes': len(history),
        'total_steps': safe_float(final.get('total_steps')),
        'metric_points': len(metric_history),
        'metric_source': metric_source,
        'score': round(score, 2),
        'final': final_core,
        'last_window': last_stats,
        'full_stats': full_stats,
        'stability': build_stability(last_stats),
        'trend': trend,
        'best_metrics': best_metrics,
        'best_checkpoint': run.get('best_checkpoint'),
        'warnings': run_warnings(config, history, last_stats),
    }


def build_metric_history(history: List[Dict[str, Any]]) -> tuple:
    """Prefer deterministic evaluation rows when available."""
    eval_rows = [row for row in history if row.get('eval_coverage_rate') is not None]
    if eval_rows:
        converted = []
        for row in eval_rows:
            converted.append({
                **row,
                'avg_reward': safe_float(row.get('eval_avg_reward')),
                'coverage_rate': safe_float(row.get('eval_coverage_rate')),
                'collision_count': safe_float(row.get('eval_collision_count')),
                'avg_min_distance': safe_float(row.get('eval_avg_min_distance')),
                'redundancy_rate': safe_float(row.get('eval_redundancy_rate')),
                'completion_steps': safe_float(row.get('eval_completion_steps')),
            })
        return converted, 'eval'
    return history, 'train'


def stats(history: List[Dict[str, Any]], metric: str) -> Dict[str, float]:
    values = [safe_float(row.get(metric)) for row in history if row.get(metric) is not None]
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': round(mean(values), 6),
        'std': round(pstdev(values), 6) if len(values) > 1 else 0.0,
        'min': round(min(values), 6),
        'max': round(max(values), 6),
    }


def build_trend(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(history) < 4:
        return {
            'reward_improved': False,
            'coverage_improved': False,
            'stable_coverage': False,
            'note': 'episode 数较少，趋势判断可信度有限',
        }

    half = max(1, len(history) // 2)
    early = history[:half]
    late = history[half:]
    early_reward = stats(early, 'avg_reward')['mean']
    late_reward = stats(late, 'avg_reward')['mean']
    early_cov = stats(early, 'coverage_rate')['mean']
    late_cov = stats(late, 'coverage_rate')['mean']
    late_cov_std = stats(late, 'coverage_rate')['std']

    return {
        'reward_improved': late_reward > early_reward,
        'coverage_improved': late_cov > early_cov,
        'stable_coverage': late_cov_std <= 0.15,
        'reward_delta': round(late_reward - early_reward, 6),
        'coverage_delta': round(late_cov - early_cov, 6),
        'late_coverage_std': late_cov_std,
    }


def build_stability(last_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    coverage_std = last_stats['coverage_rate']['std']
    reward_std = last_stats['avg_reward']['std']
    return {
        'coverage_std': coverage_std,
        'reward_std': reward_std,
        'stable': coverage_std <= 0.15,
    }


def compute_score(last_stats: Dict[str, Dict[str, float]],
                  final: Dict[str, float]) -> float:
    coverage = last_stats['coverage_rate']['mean']
    collision = last_stats['collision_count']['mean']
    redundancy = last_stats['redundancy_rate']['mean']
    min_distance = last_stats['avg_min_distance']['mean']
    completion = last_stats['completion_steps']['mean']

    coverage_score = coverage * 45
    safety_score = max(0.0, 25 - collision * 8)
    redundancy_score = max(0.0, 15 - redundancy * 15)
    distance_score = max(0.0, 10 - min_distance * 6)
    efficiency_score = max(0.0, 5 - max(completion - 50, 0) * 0.08)
    final_bonus = 3 if final.get('coverage_rate', 0.0) >= 0.99 else 0
    return max(0.0, min(100.0, coverage_score + safety_score +
                        redundancy_score + distance_score +
                        efficiency_score + final_bonus))


def choose_best_run(run_summaries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not run_summaries:
        return None
    return max(run_summaries, key=lambda run: run.get('score', 0.0))


def compare_runs(run_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(run_summaries) < 2:
        return {'available': False, 'notes': ['只有一个 run，无法做横向对比']}

    sorted_runs = sorted(run_summaries, key=lambda r: r.get('score', 0.0), reverse=True)
    best = sorted_runs[0]
    second = sorted_runs[1]
    return {
        'available': True,
        'ranking': [
            {
                'label': run['label'],
                'algo': run.get('algo'),
                'score': run.get('score', 0.0),
                'coverage_mean': run['last_window']['coverage_rate']['mean'],
                'collision_mean': run['last_window']['collision_count']['mean'],
                'redundancy_mean': run['last_window']['redundancy_rate']['mean'],
            }
            for run in sorted_runs
        ],
        'score_gap': round(best.get('score', 0.0) - second.get('score', 0.0), 2),
        'best_label': best['label'],
    }


def build_headline(config: Dict[str, Any],
                   runs: List[Dict[str, Any]],
                   best: Optional[Dict[str, Any]]) -> str:
    mode = config.get('experiment_mode', 'single')
    if not best:
        return '实验尚未产生有效结果'
    if mode == 'single':
        return f"{best['label']} 训练完成，综合评分 {best['score']:.1f}/100。"
    return f"{mode_label(mode)}完成，当前最优方案为 {best['label']}，综合评分 {best['score']:.1f}/100。"


def build_diagnosis(best: Optional[Dict[str, Any]],
                    runs: List[Dict[str, Any]]) -> List[str]:
    if not best:
        return ['没有足够数据生成诊断']
    diag = []
    cov = best['last_window']['coverage_rate']['mean']
    collision = best['last_window']['collision_count']['mean']
    redundancy = best['last_window']['redundancy_rate']['mean']
    trend = best.get('trend', {})

    if cov >= 0.8:
        diag.append('目标覆盖率较高，智能体已经具备较明显的覆盖能力。')
    elif cov >= 0.5:
        diag.append('目标覆盖率处于中等水平，已出现有效覆盖但仍不稳定。')
    else:
        diag.append('目标覆盖率偏低，当前策略尚未形成稳定覆盖行为。')

    if collision <= 0.2:
        diag.append('碰撞次数较少，安全性表现较好。')
    else:
        diag.append('仍存在明显碰撞，需要加强安全距离约束或延长训练。')

    if redundancy <= 0.1:
        diag.append('冗余覆盖率较低，搜索资源浪费较少。')
    else:
        diag.append('存在重复覆盖，智能体分工仍需改善。')

    if trend.get('reward_improved'):
        diag.append('后半段奖励相较前半段有所提升，训练方向基本有效。')
    else:
        diag.append('奖励趋势提升不明显，可能需要更多 episodes 或调整奖励权重。')

    return diag


def build_recommendations(config: Dict[str, Any],
                          best: Optional[Dict[str, Any]],
                          runs: List[Dict[str, Any]]) -> List[str]:
    if not best:
        return ['先完成至少一次训练，再生成有效建议。']

    rec = []
    episodes = best.get('episodes', 0)
    cov = best['last_window']['coverage_rate']['mean']
    collision = best['last_window']['collision_count']['mean']
    redundancy = best['last_window']['redundancy_rate']['mean']

    if episodes < 3000:
        rec.append('当前训练轮数偏少，适合趋势观察；正式报告建议提升到 3000+ episodes。')
    if cov < 0.8:
        rec.append('覆盖率仍有提升空间，建议延长训练或比较 RS-MADDPG 与 MADDPG。')
    if collision > 0.2:
        rec.append('碰撞偏多时，可提高安全距离权重或启用动态奖励权重。')
    if redundancy > 0.1:
        rec.append('冗余覆盖偏高时，可重点展示目标分配和冗余惩罚的消融效果。')
    if len(runs) == 1:
        rec.append('建议继续运行算法对比实验，用 Random、IDDPG、MADDPG、RS-MADDPG 横向证明改进效果。')

    return rec or ['当前结果较好，可以进入更长训练和报告整理阶段。']


def build_report_text(config: Dict[str, Any],
                      runs: List[Dict[str, Any]],
                      best: Optional[Dict[str, Any]],
                      comparisons: Dict[str, Any]) -> str:
    if not best:
        return '当前实验尚未产生有效数据，无法生成报告文本。'

    mode = mode_label(config.get('experiment_mode', 'single'))
    metric_note = '无探索噪声评估' if best.get('metric_source') == 'eval' else '训练采样'
    cov = best['last_window']['coverage_rate']['mean']
    collision = best['last_window']['collision_count']['mean']
    redundancy = best['last_window']['redundancy_rate']['mean']
    completion = best['last_window']['completion_steps']['mean']

    text = (
        f"本次{mode}中，基于{metric_note}指标，{best['label']} 的综合表现最好，综合评分为 "
        f"{best['score']:.1f}/100。最后 10% episode 的平均目标覆盖率为 "
        f"{cov:.1%}，平均碰撞次数为 {collision:.2f}，冗余覆盖率为 "
        f"{redundancy:.1%}，平均完成步数为 {completion:.1f}。"
    )
    if comparisons.get('available'):
        text += (
            f" 横向对比结果显示，{comparisons['best_label']} 在覆盖效率、"
            "安全性和冗余控制等指标上更具优势。"
        )
    text += (
        " 这些结果可用于分析多智能体强化学习在无人机协同区域覆盖任务中的协作能力，"
        "并进一步支撑奖励塑形机制对目标分配、安全约束和重复搜索抑制的作用。"
    )
    return text


def run_warnings(config: Dict[str, Any],
                 history: List[Dict[str, Any]],
                 last_stats: Dict[str, Dict[str, float]]) -> List[str]:
    warnings = []
    if len(history) < 3000:
        warnings.append('训练轮数偏少，结论适合趋势观察，不宜作为最终强结论。')
    if last_stats['coverage_rate']['std'] > 0.2:
        warnings.append('覆盖率波动较大，策略稳定性仍需观察。')
    if last_stats['collision_count']['mean'] > 0.5:
        warnings.append('碰撞次数偏高，安全性表现不足。')
    return warnings


def grade_experiment(best: Optional[Dict[str, Any]]) -> str:
    if not best:
        return 'unknown'
    score = best.get('score', 0.0)
    if score >= 80:
        return 'excellent'
    if score >= 65:
        return 'good'
    if score >= 45:
        return 'watch'
    return 'weak'


def mode_label(mode: str) -> str:
    labels = {
        'single': '单次训练',
        'exp1': '实验一：随机策略与训练后策略对比',
        'exp2': '实验二：算法对比实验',
        'exp3': '实验三：消融实验',
        'exp4': '实验四：泛化实验',
    }
    return labels.get(mode, mode)


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
