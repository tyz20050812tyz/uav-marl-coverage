"""Industrial HTML training console for the UAV MARL project.

Run with:
    python web_server.py --port 8600

The server intentionally uses Python's standard library so the project does
not depend on Streamlit, FastAPI, Node, or a build pipeline. Training runs in a
background thread and streams every environment step to the browser through
Server-Sent Events. The browser renders those positions on a Canvas.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

from agents.maddpg_agent import MADDPGAgent
from agents.rs_maddpg_agent import RSMADDPGAgent
from env.simple_spread_wrapper import SimpleSpreadWrapper
from experiments.adapters import IDDPGManager, RandomManager
from reports.report_generator import summarize_experiment
from utils.metrics import compute_completion_steps, compute_metrics


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / 'web_static'
LOG_DIR = ROOT / 'outputs' / 'logs'
MODEL_DIR = ROOT / 'outputs' / 'models'
ARCHIVE_DIR = ROOT / 'outputs' / 'archive'
TASKS_INDEX_PATH = ROOT / 'outputs' / 'tasks_index.json'
DOWNLOAD_ROOTS = (LOG_DIR.resolve(), MODEL_DIR.resolve())
LOCAL_ENV_FILE = ROOT / '.env.local'

LOGGER = logging.getLogger('uav-web')


def _load_local_env(path: Path = LOCAL_ENV_FILE):
    """Load simple KEY=value or export KEY=value entries without overriding shell env."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].strip()
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).round(5).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _max_cycles(num_agents: int) -> int:
    return {3: 50, 4: 70, 5: 90}.get(num_agents, 50)


def _agent_factory(algo: str, env: SimpleSpreadWrapper, cfg: Dict[str, Any]):
    common = dict(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        actor_lr=cfg['actor_lr'],
        critic_lr=cfg['critic_lr'],
        gamma=cfg['gamma'],
    )
    if algo == 'RS-MADDPG':
        return RSMADDPGAgent(
            **common,
            coverage_radius=env.world_size * cfg['coverage_ratio'],
            safe_distance=env.world_size * cfg['safe_ratio'],
            use_weight_scheduling=cfg['use_weight_scheduling'],
            use_assignment=cfg.get('use_assignment', True),
            use_redundancy=cfg.get('use_redundancy', True),
            use_safety=cfg.get('use_safety', True),
            use_collision=cfg.get('use_collision', True),
            total_episodes=cfg['episodes'],
        )
    if algo == 'MADDPG':
        return MADDPGAgent(**common)
    if algo == 'IDDPG':
        return IDDPGManager(**common)
    return RandomManager(num_agents=env.num_agents, act_dim=env.act_dim)


class EventHub:
    """Small fan-out broadcaster for Server-Sent Events clients."""

    def __init__(self):
        self._clients = set()
        self._lock = threading.RLock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._clients.add(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            self._clients.discard(q)

    def publish(self, event: str, data: Dict[str, Any]):
        payload = {'event': event, **_jsonable(data)}
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.put_nowait(dict(payload))
            except queue.Full:
                try:
                    client.get_nowait()
                    client.put_nowait(dict(payload))
                except queue.Empty:
                    pass


class TrainingService:
    """Owns the training thread and exposes a compact application state."""

    DEFAULT_CONFIG = {
        'experiment_mode': 'single',
        'algo': 'RS-MADDPG',
        'episodes': 500,
        'num_agents': 3,
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'gamma': 0.95,
        'batch_size': 256,
        'buffer_warmup': 256,
        'seed': 42,
        'coverage_ratio': 0.12,
        'safe_ratio': 0.10,
        'use_weight_scheduling': True,
        'use_assignment': True,
        'use_redundancy': True,
        'use_safety': True,
        'use_collision': True,
        'use_wandb': False,
        'wandb_project': 'uav-marl',
        'wandb_run_name': '',
        'update_repeats': 1,
        'frame_stride': 1,
        'eval_interval': 50,
        'eval_episodes': 5,
        'noise_final_scale': 0.10,
        'exp2_algorithms': ['Random', 'IDDPG', 'MADDPG', 'RS-MADDPG'],
        'exp3_groups': ['maddpg', 'assignment', 'redundancy', 'safety', 'full'],
        'exp4_scales': [3, 4, 5],
        'task_name': '',
    }

    def __init__(self, hub: EventHub):
        self.hub = hub
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._env = None
        self._agent = None
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.queue: list[str] = []
        self.active_task_id: Optional[str] = None
        self.state = self._empty_state()
        self._init_recovery()

    def _empty_state(self) -> Dict[str, Any]:
        return {
            'task_id': None,
            'task_name': '',
            'status': 'idle',
            'message': 'ready',
            'config': dict(self.DEFAULT_CONFIG),
            'episode': 0,
            'metrics': {},
            'history': [],
            'current_run': None,
            'experiment_results': [],
            'report': None,
            'llm_report': None,
            'assets': [],
            'started_at': None,
            'ended_at': None,
        }

    def _new_task_state(self, task_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        task_name = cfg.get('task_name') or self._default_task_name(cfg)
        return {
            **self._empty_state(),
            'task_id': task_id,
            'task_name': task_name,
            'status': 'waiting',
            'message': 'waiting',
            'config': cfg,
            'created_at': time.time(),
        }

    def _default_task_name(self, cfg: Dict[str, Any]) -> str:
        return (
            f"{cfg.get('algo', 'RS-MADDPG')}_"
            f"{cfg.get('experiment_mode', 'single')}_"
            f"N{cfg.get('num_agents', 3)}_"
            f"seed{cfg.get('seed', 42)}_"
            f"{cfg.get('episodes', 500)}ep"
        )

    def snapshot(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if task_id and task_id in self.tasks:
                state = self.tasks[task_id]
            elif self.active_task_id and self.active_task_id in self.tasks:
                state = self.tasks[self.active_task_id]
            else:
                state = self.state
            payload = {
                **state,
                'tasks': self._task_summaries_locked(),
                'active_task_id': self.active_task_id,
                'queue': list(self.queue),
            }
            return json.loads(json.dumps(_jsonable(payload)))

    def tasks_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'tasks': self._task_summaries_locked(),
                'active_task_id': self.active_task_id,
                'queue': list(self.queue),
            }

    def _task_summaries_locked(self) -> list:
        summaries = []
        for task in self.tasks.values():
            metrics = task.get('metrics') or {}
            config = task.get('config') or {}
            best = self._best_from_task(task)
            summaries.append({
                'task_id': task.get('task_id'),
                'task_name': task.get('task_name') or '',
                'status': task.get('status'),
                'message': task.get('message'),
                'experiment_mode': config.get('experiment_mode'),
                'algo': config.get('algo'),
                'num_agents': config.get('num_agents'),
                'episodes': config.get('episodes'),
                'episode': task.get('episode', 0),
                'coverage_rate': metrics.get('eval_coverage_rate', metrics.get('coverage_rate', 0.0)),
                'best_score': best.get('score', metrics.get('best_eval_score', 0.0)),
                'created_at': task.get('created_at'),
                'started_at': task.get('started_at'),
                'ended_at': task.get('ended_at'),
            })
        return sorted(
            summaries,
            key=lambda item: (item.get('created_at') or 0),
            reverse=True,
        )

    def _best_from_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        best = {}
        for result in task.get('experiment_results') or []:
            candidate = result.get('best_metrics') or {}
            if candidate.get('score', -1) > best.get('score', -1):
                best = candidate
        return best

    # ── Persistence & Recovery ──────────────────────────────────────

    def _init_recovery(self):
        """Recover task records from persistence file and reconcile with disk files."""
        if not TASKS_INDEX_PATH.exists():
            LOGGER.info("No tasks_index.json found — will scan existing files for recovery")
        recovered = self._load_tasks_index()
        self._scan_existing_files(recovered)
        with self._lock:
            for task_id, task in recovered.items():
                if task_id not in self.tasks:
                    task['status'] = task.get('status', 'interrupted')
                    task['message'] = task.get('message', 'recovered on restart')
                    self.tasks[task_id] = task
            if recovered:
                self._save_tasks_index_locked()
                LOGGER.info("Recovered %d previous task(s)", len(recovered))

    def _load_tasks_index(self) -> Dict[str, Dict[str, Any]]:
        """Load tasks_index.json. Returns empty dict on any error."""
        try:
            if not TASKS_INDEX_PATH.exists():
                return {}
            with open(TASKS_INDEX_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tasks = data.get('tasks', {}) if isinstance(data, dict) else {}
            valid: Dict[str, Dict[str, Any]] = {}
            for tid, task in tasks.items():
                if isinstance(task, dict) and 'task_id' in task:
                    valid[tid] = task
            return valid
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Could not load tasks_index.json: %s", exc)
            return {}

    def _save_tasks_index_locked(self):
        """Persist current task metadata (minimal summaries) to disk.
        Must be called while holding self._lock."""
        summaries: Dict[str, Any] = {}
        for task_id, task in self.tasks.items():
            # Strip large history arrays from experiment results for compact storage
            slim_results = []
            for result in (task.get('experiment_results') or []):
                slim = {k: v for k, v in result.items() if k != 'history'}
                slim_results.append(slim)
            summaries[task_id] = {
                'task_id': task.get('task_id'),
                'task_name': task.get('task_name'),
                'status': task.get('status'),
                'message': task.get('message'),
                'config': {
                    k: v for k, v in (task.get('config') or {}).items()
                    if k in {'experiment_mode', 'algo', 'episodes', 'num_agents',
                             'seed', 'coverage_ratio', 'safe_ratio',
                             'use_weight_scheduling', 'use_assignment', 'use_redundancy',
                             'use_safety', 'use_collision', 'eval_interval'}
                },
                'created_at': task.get('created_at'),
                'started_at': task.get('started_at'),
                'ended_at': task.get('ended_at'),
                'episode': task.get('episode', 0),
                'metrics': task.get('metrics'),
                'experiment_results': slim_results,
                'report': task.get('report'),
                'run_plan': task.get('run_plan'),
            }
        payload = {'updated_at': time.time(), 'tasks': summaries}
        tmp_path = TASKS_INDEX_PATH.with_suffix('.tmp')
        TASKS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            tmp_path.replace(TASKS_INDEX_PATH)
        except OSError as exc:
            LOGGER.error("Failed to persist tasks_index.json: %s", exc)

    def _scan_existing_files(self, recovered: Dict[str, Dict[str, Any]]):
        """Scan log/model directories for files that belong to unknown tasks
        and create stub entries for them. Merges into `recovered` in place."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        discovered: Dict[str, Dict[str, Any]] = {}

        # Scan log files for task_id patterns
        for path in sorted(LOG_DIR.glob('*.json')):
            task_id = self._extract_task_id_from_filename(path.stem)
            if not task_id:
                continue
            if task_id in self.tasks or task_id in recovered or task_id in discovered:
                continue
            discovered[task_id] = self._stub_from_filename(path, task_id, 'log')

        # Scan model files
        for path in sorted(MODEL_DIR.glob('*.pt')):
            task_id = self._extract_task_id_from_filename(path.stem)
            if not task_id:
                continue
            if task_id in self.tasks or task_id in recovered or task_id in discovered:
                continue
            discovered[task_id] = self._stub_from_filename(path, task_id, 'model')

        if discovered:
            LOGGER.info("Discovered %d task(s) from existing files", len(discovered))
            recovered.update(discovered)

    def _extract_task_id_from_filename(self, stem: str) -> Optional[str]:
        """Extract task_id from filename like:
        'rs_maddpg_1782868709_92de4b6f_single_seed42_logs'
        → 'rs_maddpg_1782868709_92de4b6f'

        Pattern: {algo}_{timestamp}_{uuid8}[_{suffix}_seed{seed}_{kind}]
        """
        import re
        # Match: algo_timestamp_uuid8 pattern
        match = re.match(
            r'^([a-z_]+)_(\d{10})_([0-9a-f]{8})',
            stem,
        )
        if match:
            return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
        return None

    def _stub_from_filename(self, path: Path, task_id: str,
                            source: str) -> Dict[str, Any]:
        """Create a minimal recovered task stub."""
        import re
        stem = path.stem
        seed_match = re.search(r'_seed(\d+)', stem)
        seed = int(seed_match.group(1)) if seed_match else 42
        parts = task_id.split('_')
        algo = parts[0].upper().replace('_', '-') if parts else 'unknown'
        now = time.time()
        return {
            'task_id': task_id,
            'task_name': f"{task_id} (recovered)",
            'status': 'interrupted',
            'message': f'recovered from {source} file',
            'config': {
                'experiment_mode': 'single',
                'algo': algo if algo in {'RANDOM', 'IDDPG', 'MADDPG', 'RS-MADDPG'} else 'RS-MADDPG',
                'episodes': 500,
                'num_agents': 3,
                'seed': seed,
            },
            'created_at': now,
            'started_at': now,
            'ended_at': now,
            'episode': 0,
            'metrics': {},
        }

    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """Archive all files belonging to a task and remove it from the index.

        Moves related log files, model files, and summary files to
        outputs/archive/{task_id}/ and removes the task from memory and index.
        """
        with self._lock:
            if task_id not in self.tasks:
                return {'error': f'任务 {task_id} 不存在'}

            # Remove from queue if present
            if task_id in self.queue:
                self.queue.remove(task_id)
                self._stop.set()

            task = self.tasks.pop(task_id)
            if self.active_task_id == task_id:
                self.active_task_id = None
                self.state = self._empty_state()

            self._save_tasks_index_locked()

        # Archive files outside the lock to avoid blocking
        archived_count = self._archive_task_files(task_id)

        self.hub.publish('tasks', self.tasks_snapshot())
        LOGGER.info("Deleted task %s — archived %d file(s)", task_id, archived_count)
        return {
            'ok': True,
            'task_id': task_id,
            'archived_files': archived_count,
        }

    def _archive_task_files(self, task_id: str) -> int:
        """Move all files matching task_id to archive directory."""
        archive_sub = ARCHIVE_DIR / task_id
        archive_sub.mkdir(parents=True, exist_ok=True)
        count = 0

        # Collect files from logs and models directories
        candidates: list[Path] = []
        for directory in (LOG_DIR, MODEL_DIR):
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                stem = path.stem
                if stem.startswith(task_id):
                    candidates.append(path)

        for path in candidates:
            dest = archive_sub / path.name
            try:
                # If destination exists, add a timestamp suffix
                if dest.exists():
                    dest = archive_sub / f"{path.stem}_{int(time.time())}{path.suffix}"
                path.rename(dest)
                count += 1
            except OSError as exc:
                LOGGER.warning("Failed to archive %s → %s: %s", path, dest, exc)

        # Write an archive manifest for traceability
        manifest = {
            'task_id': task_id,
            'archived_at': time.time(),
            'archived_at_iso': time.strftime(
                '%Y-%m-%dT%H:%M:%S', time.localtime(),
            ),
            'file_count': count,
        }
        manifest_path = archive_sub / 'archive_manifest.json'
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        except OSError as exc:
            LOGGER.warning("Failed to write archive manifest: %s", exc)

        return count

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        task_id = self.create_task(config)
        return self.snapshot(task_id)

    def create_task(self, config: Dict[str, Any]) -> str:
        cfg = self._sanitize_config(config)
        task_id = self._make_task_id(cfg)
        cfg['task_id'] = task_id
        should_start = False
        with self._lock:
            self.tasks[task_id] = self._new_task_state(task_id, cfg)
            self.queue.append(task_id)
            self._save_tasks_index_locked()
            if self._thread is None or not self._thread.is_alive():
                should_start = True
        if should_start:
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()
        self.hub.publish('tasks', self.tasks_snapshot())
        return task_id

    def _make_task_id(self, cfg: Dict[str, Any]) -> str:
        prefix = (
            f"{cfg.get('algo', 'task').lower().replace('-', '_')}_"
            f"{int(time.time())}_"
        )
        return prefix + uuid.uuid4().hex[:8]

    def stop(self, task_id: Optional[str] = None):
        event = 'tasks'
        snapshot_task_id = None
        should_save = False
        with self._lock:
            target = task_id or self.active_task_id
            if target and target in self.queue:
                self.queue.remove(target)
                task = self.tasks[target]
                task['status'] = 'stopped'
                task['message'] = 'stopped before running'
                task['ended_at'] = time.time()
                should_save = True
            if target and target == self.active_task_id:
                self._stop.set()
                task = self.tasks[target]
                task['message'] = 'stopping'
                event = 'status'
                snapshot_task_id = target
            if should_save:
                self._save_tasks_index_locked()
        if event == 'status':
            self.hub.publish('status', self.snapshot(snapshot_task_id))
        else:
            self.hub.publish('tasks', self.tasks_snapshot())

    def _worker_loop(self):
        while True:
            should_break = False
            with self._lock:
                if not self.queue:
                    self.active_task_id = None
                    self.state = self._empty_state()
                    should_break = True
                if should_break:
                    task_id = None
                    cfg = None
                    self._save_tasks_index_locked()
                else:
                    task_id = self.queue.pop(0)
                    task = self.tasks[task_id]
                    self.active_task_id = task_id
                    self.state = task
                    self._stop.clear()
                    task.update({
                        'status': 'running',
                        'message': 'training',
                        'episode': 0,
                        'metrics': {},
                        'history': [],
                        'current_run': None,
                        'experiment_results': [],
                        'report': None,
                        'llm_report': None,
                        'assets': [],
                        'started_at': time.time(),
                        'ended_at': None,
                    })
                    self._save_tasks_index_locked()
                    cfg = dict(task['config'])
            if should_break:
                self.hub.publish('tasks', self.tasks_snapshot())
                break
            self._publish('status', self.snapshot(task_id))
            self._run(cfg, task_id)
            with self._lock:
                self.active_task_id = None
                if self._env is not None:
                    self._env = None
                if self._agent is not None:
                    self._agent = None
            self._publish('tasks', self.tasks_snapshot())

    def _publish(self, event: str, data: Dict[str, Any]):
        payload = dict(data or {})
        if self.active_task_id and 'task_id' not in payload:
            payload['task_id'] = self.active_task_id
        self.hub.publish(event, payload)

    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(self.DEFAULT_CONFIG)
        cfg.update(config or {})

        def _clean_list(value, allowed, default):
            if isinstance(value, str):
                items = [item.strip() for item in value.split(',') if item.strip()]
            elif isinstance(value, (list, tuple, set)):
                items = list(value)
            else:
                items = list(default)
            cleaned = []
            allowed_set = set(allowed)
            for item in items:
                if item in allowed_set and item not in cleaned:
                    cleaned.append(item)
            return cleaned or list(default)

        cfg['experiment_mode'] = cfg['experiment_mode'] if cfg['experiment_mode'] in {'single', 'exp1', 'exp2', 'exp3', 'exp4'} else 'single'
        cfg['algo'] = cfg['algo'] if cfg['algo'] in {'Random', 'IDDPG', 'MADDPG', 'RS-MADDPG'} else 'RS-MADDPG'
        cfg['episodes'] = int(np.clip(int(cfg['episodes']), 1, 50000))
        cfg['num_agents'] = int(cfg['num_agents']) if int(cfg['num_agents']) in {3, 4, 5} else 3
        cfg['actor_lr'] = float(np.clip(float(cfg['actor_lr']), 1e-5, 1e-2))
        cfg['critic_lr'] = float(np.clip(float(cfg['critic_lr']), 1e-5, 1e-2))
        cfg['gamma'] = float(np.clip(float(cfg['gamma']), 0.8, 0.99))
        cfg['batch_size'] = int(np.clip(int(cfg['batch_size']), 32, 2048))
        cfg['buffer_warmup'] = int(np.clip(int(cfg['buffer_warmup']), 32, 4096))
        cfg['seed'] = int(np.clip(int(cfg['seed']), 0, 999999))
        cfg['coverage_ratio'] = float(np.clip(float(cfg['coverage_ratio']), 0.04, 0.25))
        cfg['safe_ratio'] = float(np.clip(float(cfg['safe_ratio']), 0.04, 0.25))
        cfg['use_weight_scheduling'] = bool(cfg['use_weight_scheduling'])
        cfg['use_assignment'] = bool(cfg.get('use_assignment', True))
        cfg['use_redundancy'] = bool(cfg.get('use_redundancy', True))
        cfg['use_safety'] = bool(cfg.get('use_safety', True))
        cfg['use_collision'] = bool(cfg.get('use_collision', True))
        cfg['use_wandb'] = bool(cfg['use_wandb'])
        cfg['wandb_project'] = str(cfg.get('wandb_project') or 'uav-marl').strip()[:80] or 'uav-marl'
        cfg['wandb_run_name'] = str(cfg.get('wandb_run_name') or '').strip()[:120]
        cfg['update_repeats'] = int(np.clip(int(cfg['update_repeats']), 0, 4))
        cfg['frame_stride'] = int(np.clip(int(cfg['frame_stride']), 1, 5))
        cfg['eval_interval'] = int(np.clip(int(cfg.get('eval_interval', 50)), 1, 1000))
        cfg['eval_episodes'] = int(np.clip(int(cfg.get('eval_episodes', 5)), 1, 50))
        cfg['noise_final_scale'] = float(np.clip(float(cfg.get('noise_final_scale', 0.10)), 0.0, 1.0))
        cfg['exp2_algorithms'] = _clean_list(
            cfg.get('exp2_algorithms'),
            ['Random', 'IDDPG', 'MADDPG', 'RS-MADDPG'],
            ['Random', 'IDDPG', 'MADDPG', 'RS-MADDPG'],
        )
        cfg['exp3_groups'] = _clean_list(
            cfg.get('exp3_groups'),
            ['maddpg', 'assignment', 'redundancy', 'safety', 'full'],
            ['maddpg', 'assignment', 'redundancy', 'safety', 'full'],
        )
        cfg['exp4_scales'] = [
            int(value) for value in _clean_list(cfg.get('exp4_scales'), [3, 4, 5], [3, 4, 5])
        ]
        cfg['task_name'] = str(cfg.get('task_name') or '').strip()[:120]
        return cfg

    def _run(self, cfg: Dict[str, Any], task_id: Optional[str] = None):
        try:
            self._train_loop(cfg)
            final_status = 'stopped' if self._stop.is_set() else 'complete'
            with self._lock:
                self.state['status'] = final_status
                self.state['message'] = final_status
                self.state['ended_at'] = time.time()
                self._save_tasks_index_locked()
            self._publish('complete', self.snapshot(task_id))
        except Exception as exc:
            LOGGER.exception('training failed')
            with self._lock:
                self.state['status'] = 'error'
                self.state['message'] = str(exc)
                self.state['ended_at'] = time.time()
                self._save_tasks_index_locked()
            self._publish('error', self.snapshot(task_id))
        finally:
            if self._env is not None:
                self._env.close()

    def _train_loop(self, cfg: Dict[str, Any]):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        run_plan = self._build_run_plan(cfg)
        results = []
        with self._lock:
                self.state['run_plan'] = [
                    {'id': run['run_id'], 'label': run['label'], 'algo': run['algo'],
                     'num_agents': run['num_agents'], 'episodes': run['episodes'],
                     'features': run.get('features', [])}
                    for run in run_plan
                ]

        for index, run_cfg in enumerate(run_plan, start=1):
            if self._stop.is_set():
                break
            with self._lock:
                self.state['current_run'] = {
                    'index': index,
                    'total': len(run_plan),
                    'id': run_cfg['run_id'],
                    'label': run_cfg['label'],
                    'algo': run_cfg['algo'],
                    'num_agents': run_cfg['num_agents'],
                    'features': run_cfg.get('features', []),
                }
                self.state['episode'] = 0
                self.state['metrics'] = {}
                self.state['history'] = []
            self._publish('run_start', self.snapshot())
            result = self._train_single_run(run_cfg, index, len(run_plan))
            results.append(result)
            with self._lock:
                self.state['experiment_results'] = results
            self._publish('run_end', {'run': result, 'results': results})

        report = summarize_experiment(cfg, results)
        summary_path = self._save_experiment_report(cfg, report)
        with self._lock:
            self.state['experiment_results'] = results
            self.state['report'] = report
            self.state['assets'] = self._build_assets_locked(self.state, summary_path)
        self._publish('report', report)

    def _train_single_run(self, cfg: Dict[str, Any], run_index: int,
                          total_runs: int) -> Dict[str, Any]:
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])

        env = SimpleSpreadWrapper(
            num_agents=cfg['num_agents'],
            max_cycles=_max_cycles(cfg['num_agents']),
        )
        agent = _agent_factory(cfg['algo'], env, cfg)
        self._env = env
        self._agent = agent

        coverage_radius = env.world_size * cfg['coverage_ratio']
        safe_distance = env.world_size * cfg['safe_ratio']
        total_steps = 0
        history = []
        log_path = None
        final_checkpoint = None
        wandb_run = self._start_wandb(cfg, env)
        best_score = float('-inf')
        best_metrics: Optional[Dict[str, Any]] = None
        best_checkpoint = None

        try:
            for episode in range(1, cfg['episodes'] + 1):
                if self._stop.is_set():
                    break

                if hasattr(agent, 'set_episode'):
                    agent.set_episode(episode - 1)
                noise_scale = self._exploration_scale(episode, cfg)
                if hasattr(agent, 'set_noise_scale'):
                    agent.set_noise_scale(noise_scale)
                if hasattr(agent, 'reset_episode'):
                    agent.reset_episode()
                if hasattr(agent, 'reset_noise'):
                    agent.reset_noise()

                obs, _ = env.reset()
                state = env.get_world_state()
                landmark_positions = state['landmark_positions'].copy()
                position_history = [state['agent_positions'].copy()]
                ep_reward = 0.0
                ep_steps = 0

                self._publish('episode_start', {
                    'run_index': run_index,
                    'total_runs': total_runs,
                    'run_id': cfg['run_id'],
                    'run_label': cfg['label'],
                    'episode': episode,
                    'total_episodes': cfg['episodes'],
                    'landmark_positions': landmark_positions,
                    'coverage_radius': coverage_radius,
                    'safe_distance': safe_distance,
                    'num_agents': env.num_agents,
                    'world_size': env.world_size,
                })
                self._publish('frame', {
                    'run_id': cfg['run_id'],
                    'episode': episode,
                    'step': 0,
                    'agent_positions': state['agent_positions'],
                    'landmark_positions': landmark_positions,
                })

                while True:
                    actions = {
                        name: agent.act(name, obs[name], add_noise=True)
                        if hasattr(agent, 'act') else obs[name]
                        for name in obs
                    }
                    next_obs, env_rewards, terms, truncs, _ = env.step(actions)

                    state = env.get_world_state()
                    if hasattr(agent, 'compute_shaped_reward'):
                        rewards = agent.compute_shaped_reward(
                            env_rewards,
                            state['agent_positions'],
                            state['landmark_positions'],
                        )
                    else:
                        rewards = env_rewards

                    dones = {name: bool(terms[name]) or bool(truncs[name]) for name in obs}
                    if hasattr(agent, 'buffer'):
                        agent.buffer.push(obs, actions, rewards, next_obs, dones)

                    ep_reward += float(sum(rewards.values()))
                    ep_steps += 1
                    total_steps += 1
                    position_history.append(state['agent_positions'].copy())

                    if ep_steps % cfg['frame_stride'] == 0:
                        self._publish('frame', {
                            'run_id': cfg['run_id'],
                            'episode': episode,
                            'step': ep_steps,
                            'agent_positions': state['agent_positions'],
                            'landmark_positions': state['landmark_positions'],
                        })

                    obs = next_obs
                    if all(dones.values()) or self._stop.is_set():
                        break

                update_info = {}
                if hasattr(agent, 'buffer') and hasattr(agent, 'update') and agent.buffer.is_ready(cfg['batch_size']):
                    update_steps = max(ep_steps * cfg['update_repeats'], 1)
                    for _ in range(update_steps):
                        batch = agent.buffer.sample(cfg['batch_size'], agent.device)
                        update_info = agent.update(batch)

                final_state = env.get_world_state()
                train_metrics = compute_metrics(
                    final_state['agent_positions'],
                    final_state['landmark_positions'],
                    coverage_radius,
                )
                train_metrics['avg_reward'] = ep_reward
                train_metrics['completion_steps'] = compute_completion_steps(
                    position_history,
                    final_state['landmark_positions'],
                    coverage_radius,
                )
                metrics = {
                    'metric_source': 'train',
                    'noise_scale': noise_scale,
                    'train_avg_reward': train_metrics['avg_reward'],
                    'train_coverage_rate': train_metrics['coverage_rate'],
                    'train_collision_count': train_metrics['collision_count'],
                    'train_avg_min_distance': train_metrics['avg_min_distance'],
                    'train_redundancy_rate': train_metrics['redundancy_rate'],
                    'train_covered_landmarks': train_metrics['covered_landmarks'],
                    'train_completion_steps': train_metrics['completion_steps'],
                    **train_metrics,
                }
                metrics.update(update_info)

                should_eval = (
                    episode == 1 or
                    episode == cfg['episodes'] or
                    episode % cfg['eval_interval'] == 0
                )
                if should_eval:
                    eval_metrics = self._evaluate_policy(agent, cfg, coverage_radius)
                    eval_prefixed = {f'eval_{key}': value for key, value in eval_metrics.items()}
                    metrics.update(eval_prefixed)
                    metrics.update(eval_metrics)
                    metrics['metric_source'] = 'eval'

                    eval_score = self._checkpoint_score(eval_metrics)
                    metrics['eval_score'] = eval_score
                    if eval_score > best_score:
                        best_score = eval_score
                        best_metrics = {
                            'episode': episode,
                            'score': eval_score,
                            **_jsonable(eval_metrics),
                        }
                        if hasattr(agent, 'save') and cfg['algo'] != 'Random':
                            best_checkpoint = MODEL_DIR / f"{cfg['run_id']}_seed{cfg['seed']}_best.pt"
                            agent.save(str(best_checkpoint))
                            if wandb_run is not None:
                                wandb_run.save(str(best_checkpoint), policy='now')
                        self._publish('best', {
                            'run_id': cfg['run_id'],
                            'episode': episode,
                            'metrics': best_metrics,
                            'checkpoint': str(best_checkpoint) if best_checkpoint else None,
                        })
                    metrics['best_eval_score'] = best_score if best_score > float('-inf') else 0.0

                entry = {
                    'episode': episode,
                    'total_steps': total_steps,
                    'steps': ep_steps,
                    'run_id': cfg['run_id'],
                    'run_label': cfg['label'],
                    **_jsonable(metrics),
                }
                history.append(entry)
                with self._lock:
                    self.state['episode'] = episode
                    self.state['metrics'] = entry
                    self.state['history'] = history[-1000:]

                if wandb_run is not None:
                    wandb_run.log(self._wandb_metrics(entry), step=episode)

                self._publish('episode_end', {
                    'run_index': run_index,
                    'total_runs': total_runs,
                    'run_id': cfg['run_id'],
                    'run_label': cfg['label'],
                    'episode': episode,
                    'total_episodes': cfg['episodes'],
                    'metrics': entry,
                    'progress': episode / cfg['episodes'],
                })

                if episode % 100 == 0:
                    log_path = self._save_logs(cfg, history)

            log_path = self._save_logs(cfg, history)
            if hasattr(agent, 'save') and cfg['algo'] != 'Random':
                path = MODEL_DIR / f"{cfg['run_id']}_seed{cfg['seed']}_final.pt"
                agent.save(str(path))
                final_checkpoint = path
                if wandb_run is not None:
                    wandb_run.save(str(path), policy='now')
        finally:
            if wandb_run is not None:
                wandb_run.finish()
            env.close()

        return {
            'id': cfg['run_id'],
            'label': cfg['label'],
            'algo': cfg['algo'],
            'config': _jsonable(cfg),
            'history': history,
            'best_metrics': _jsonable(best_metrics or {}),
            'best_checkpoint': str(best_checkpoint) if best_checkpoint else None,
            'final_checkpoint': str(final_checkpoint) if final_checkpoint else None,
            'log_path': str(log_path) if log_path else None,
        }

    def _exploration_scale(self, episode: int, cfg: Dict[str, Any]) -> float:
        """Linear exploration decay; keeps early exploration and reduces late noise."""
        if cfg['algo'] == 'Random':
            return 1.0
        total = max(int(cfg.get('episodes', 1)), 1)
        progress = min(max((episode - 1) / max(total - 1, 1), 0.0), 1.0)
        final_scale = float(cfg.get('noise_final_scale', 0.10))
        return 1.0 + (final_scale - 1.0) * progress

    def _evaluate_policy(self, agent, cfg: Dict[str, Any],
                         coverage_radius: float) -> Dict[str, float]:
        """Evaluate the current policy without exploration noise on fresh episodes."""
        eval_env = SimpleSpreadWrapper(
            num_agents=cfg['num_agents'],
            max_cycles=_max_cycles(cfg['num_agents']),
        )
        np_state = np.random.get_state()
        try:
            total_reward = 0.0
            totals = {
                'coverage_rate': 0.0,
                'collision_count': 0.0,
                'avg_min_distance': 0.0,
                'redundancy_rate': 0.0,
                'covered_landmarks': 0.0,
                'completion_steps': 0.0,
            }
            n = int(cfg.get('eval_episodes', 5))
            for _ in range(n):
                obs, _ = eval_env.reset()
                ep_reward = 0.0
                position_history = []
                while True:
                    actions = {
                        name: agent.act(name, obs[name], add_noise=False)
                        if hasattr(agent, 'act') else obs[name]
                        for name in obs
                    }
                    next_obs, env_rewards, terms, truncs, _ = eval_env.step(actions)
                    state = eval_env.get_world_state()
                    position_history.append(state['agent_positions'].copy())
                    ep_reward += float(sum(env_rewards.values()))
                    dones = {name: bool(terms[name]) or bool(truncs[name]) for name in obs}
                    obs = next_obs
                    if all(dones.values()):
                        break

                final_state = eval_env.get_world_state()
                episode_metrics = compute_metrics(
                    final_state['agent_positions'],
                    final_state['landmark_positions'],
                    coverage_radius,
                )
                episode_metrics['completion_steps'] = compute_completion_steps(
                    position_history,
                    final_state['landmark_positions'],
                    coverage_radius,
                )
                total_reward += ep_reward
                for key in totals:
                    totals[key] += float(episode_metrics[key])

            result = {key: value / n for key, value in totals.items()}
            result['avg_reward'] = total_reward / n
            return result
        finally:
            np.random.set_state(np_state)
            eval_env.close()

    def _checkpoint_score(self, metrics: Dict[str, Any]) -> float:
        coverage = float(metrics.get('coverage_rate', 0.0))
        collision = float(metrics.get('collision_count', 0.0))
        redundancy = float(metrics.get('redundancy_rate', 0.0))
        min_distance = float(metrics.get('avg_min_distance', 0.0))
        completion = float(metrics.get('completion_steps', 0.0))
        return round(
            coverage * 60.0 +
            max(0.0, 20.0 - collision * 8.0) +
            max(0.0, 10.0 - redundancy * 10.0) +
            max(0.0, 8.0 - min_distance * 4.0) +
            max(0.0, 2.0 - max(completion - 50.0, 0.0) * 0.04),
            4,
        )

    def _build_run_plan(self, cfg: Dict[str, Any]) -> list:
        mode = cfg.get('experiment_mode', 'single')

        def run(run_id: str, label: str, features: Optional[list] = None, **overrides):
            run_cfg = dict(cfg)
            run_cfg.update(overrides)
            task_prefix = cfg.get('task_id') or 'task'
            run_cfg['run_id'] = f"{task_prefix}_{run_id}"
            run_cfg['label'] = label
            run_cfg['features'] = features or []
            if run_cfg.get('use_wandb'):
                base_name = cfg.get('wandb_run_name') or mode
                run_cfg['wandb_run_name'] = f"{base_name}-{label}"
            return run_cfg

        if mode == 'exp1':
            trained_algo = cfg['algo'] if cfg['algo'] != 'Random' else 'RS-MADDPG'
            return [
                run('exp1_random', 'Random 基线', algo='Random'),
                run('exp1_trained', f'{trained_algo} 训练后策略', algo=trained_algo),
            ]

        if mode == 'exp2':
            definitions = {
                'Random': ('exp2_random', 'Random', ['无学习随机策略']),
                'IDDPG': ('exp2_iddpg', 'IDDPG', ['独立 DDPG']),
                'MADDPG': ('exp2_maddpg', 'MADDPG', ['集中式 Critic']),
                'RS-MADDPG': ('exp2_rs_maddpg', 'RS-MADDPG', ['集中式 Critic', '奖励塑形']),
            }
            return [
                run(run_id, label, algo=algo, features=features)
                for algo in cfg.get('exp2_algorithms', [])
                for run_id, label, features in [definitions[algo]]
            ]

        if mode == 'exp3':
            definitions = {
                'maddpg': lambda: run(
                    'exp3_maddpg', 'MADDPG baseline', algo='MADDPG',
                    features=['集中式 Critic'],
                ),
                'assignment': lambda: run(
                    'exp3_assignment', '+ Assignment', algo='RS-MADDPG',
                    use_assignment=True, use_redundancy=False, use_safety=False,
                    use_collision=False, use_weight_scheduling=False,
                    features=['目标分配奖励'],
                ),
                'redundancy': lambda: run(
                    'exp3_redundancy', '+ Assignment + Redundancy', algo='RS-MADDPG',
                    use_assignment=True, use_redundancy=True, use_safety=False,
                    use_collision=False, use_weight_scheduling=False,
                    features=['目标分配奖励', '冗余覆盖抑制'],
                ),
                'safety': lambda: run(
                    'exp3_safety', '+ Assignment + Redundancy + Safety', algo='RS-MADDPG',
                    use_assignment=True, use_redundancy=True, use_safety=True,
                    use_collision=True, use_weight_scheduling=False,
                    features=['目标分配奖励', '冗余覆盖抑制', '安全距离约束'],
                ),
                'full': lambda: run(
                    'exp3_full', 'RS-MADDPG full', algo='RS-MADDPG',
                    use_assignment=True, use_redundancy=True, use_safety=True,
                    use_collision=True, use_weight_scheduling=True,
                    features=['目标分配奖励', '冗余覆盖抑制', '安全距离约束', '动态奖励权重'],
                ),
            }
            return [definitions[group]() for group in cfg.get('exp3_groups', [])]

        if mode == 'exp4':
            return [
                run(f'exp4_n{scale}', f'{scale} 机 {scale} 目标',
                    algo='RS-MADDPG', num_agents=scale,
                    features=[f'{scale} 智能体规模'])
                for scale in cfg.get('exp4_scales', [])
            ]

        return [run('single', cfg.get('algo', 'RS-MADDPG'), algo=cfg.get('algo', 'RS-MADDPG'))]

    def _start_wandb(self, cfg: Dict[str, Any], env: SimpleSpreadWrapper):
        if not cfg.get('use_wandb'):
            return None
        try:
            import wandb
            run = wandb.init(
                project=cfg['wandb_project'],
                name=cfg['wandb_run_name'] or None,
                config={
                    **cfg,
                    'obs_dim': env.obs_dim,
                    'act_dim': env.act_dim,
                    'max_cycles': env.max_cycles,
                    'world_size': env.world_size,
                },
            )
            self._publish('wandb', {
                'status': 'active',
                'project': cfg['wandb_project'],
                'run_name': run.name,
                'url': getattr(run, 'url', None),
            })
            return run
        except Exception as exc:
            LOGGER.warning("W&B disabled after init failure: %s", exc)
            self._publish('wandb', {
                'status': 'error',
                'message': str(exc),
            })
            return None

    def _wandb_metrics(self, entry: Dict[str, Any]) -> Dict[str, float]:
        keys = {
            'avg_reward': 'reward/episode',
            'coverage_rate': 'task/coverage_rate',
            'collision_count': 'safety/collision_count',
            'avg_min_distance': 'task/avg_min_distance',
            'redundancy_rate': 'task/redundancy_rate',
            'completion_steps': 'task/completion_steps',
            'train_avg_reward': 'train/reward',
            'train_coverage_rate': 'train/coverage_rate',
            'eval_avg_reward': 'eval/reward',
            'eval_coverage_rate': 'eval/coverage_rate',
            'eval_collision_count': 'eval/collision_count',
            'eval_redundancy_rate': 'eval/redundancy_rate',
            'eval_completion_steps': 'eval/completion_steps',
            'eval_score': 'eval/score',
            'best_eval_score': 'eval/best_score',
            'noise_scale': 'exploration/noise_scale',
            'critic_loss': 'loss/critic',
            'actor_loss': 'loss/actor',
            'steps': 'episode/steps',
            'total_steps': 'episode/total_steps',
        }
        return {
            target: float(entry[source])
            for source, target in keys.items()
            if source in entry and entry[source] is not None
        }

    def _save_logs(self, cfg: Dict[str, Any], history: Optional[list] = None):
        logs = list(history) if history is not None else list(self.state.get('history', []))
        path = LOG_DIR / f"{cfg['run_id']}_seed{cfg['seed']}_logs.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        return path

    def _save_experiment_report(self, cfg: Dict[str, Any], report: Dict[str, Any]):
        prefix = cfg.get('task_id') or cfg.get('experiment_mode')
        path = LOG_DIR / f"{prefix}_seed{cfg['seed']}_summary.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_jsonable(report), f, indent=2, ensure_ascii=False)
        return path

    def _build_assets_locked(self, task: Dict[str, Any],
                             summary_path: Optional[Path] = None) -> list:
        assets = []
        task_id = task.get('task_id')
        for result in task.get('experiment_results') or []:
            label = result.get('label') or result.get('id') or 'run'
            for kind, path_key, title in [
                ('best', 'best_checkpoint', 'Best 权重'),
                ('final', 'final_checkpoint', 'Final 权重'),
                ('log', 'log_path', '训练日志'),
            ]:
                path = result.get(path_key)
                if path and Path(path).exists():
                    asset_id = f"{result.get('id')}_{kind}".replace('/', '_')
                    assets.append({
                        'asset_id': asset_id,
                        'kind': kind,
                        'label': f"{label} · {title}",
                        'filename': Path(path).name,
                        'path': path,
                        'download_url': f"/api/tasks/{task_id}/download?asset={asset_id}",
                    })
        if summary_path and Path(summary_path).exists():
            asset_id = 'summary'
            assets.append({
                'asset_id': asset_id,
                'kind': 'summary',
                'label': '结构化总结 JSON',
                'filename': Path(summary_path).name,
                'path': str(summary_path),
                'download_url': f"/api/tasks/{task_id}/download?asset={asset_id}",
            })
        return assets

    def get_asset(self, task_id: str, asset_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            for asset in task.get('assets') or []:
                if asset.get('asset_id') == asset_id:
                    return dict(asset)
        return None

    def open_models_folder(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(['open', str(MODEL_DIR)], check=False)

    def generate_deepseek_report(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            task = self.tasks.get(task_id) if task_id else self.state
            report = (task or {}).get('report')
        if not report:
            raise RuntimeError('还没有结构化实验总结，请先完成一次实验。')

        api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError('未设置 DEEPSEEK_API_KEY，无法调用 DeepSeek。')

        prompt = self._build_deepseek_prompt(report)
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        '你是多智能体强化学习课程设计报告助理。'
                        '只基于用户提供的结构化指标写结论，不编造未出现的数据。'
                        '输出中文，风格适合学生课程报告。'
                    ),
                },
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.3,
            'max_tokens': 1400,
        }
        request = urllib.request.Request(
            'https://api.deepseek.com/chat/completions',
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            },
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f'DeepSeek HTTP {exc.code}: {detail[:300]}') from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f'DeepSeek 网络连接失败: {exc.reason}') from exc

        text = body['choices'][0]['message']['content'].strip()
        result = {
            'provider': 'deepseek',
            'model': payload['model'],
            'text': text,
            'created_at': time.time(),
        }
        with self._lock:
            if task_id and task_id in self.tasks:
                self.tasks[task_id]['llm_report'] = result
            else:
                self.state['llm_report'] = result
        self._publish('llm_report', {'task_id': task_id or self.active_task_id, **result})
        return result

    def _build_deepseek_prompt(self, report: Dict[str, Any]) -> str:
        return (
            '请根据下面 JSON 实验总结，生成一份课程报告可用的实验结果分析。'
            '要求包含：1. 总体结论；2. 指标解读；3. 与计划书目标的对应关系；'
            '4. 不足与改进建议；5. 一段可直接粘贴到报告中的总结。'
            '不要虚构数据。\n\n'
            f'{json.dumps(_jsonable(report), ensure_ascii=False, indent=2)}'
        )


class RequestHandler(SimpleHTTPRequestHandler):
    server_version = 'UAVIndustrialServer/1.0'

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args):
        LOGGER.info(fmt, *args)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    @property
    def service(self) -> TrainingService:
        return self.server.service  # type: ignore[attr-defined]

    @property
    def hub(self) -> EventHub:
        return self.server.hub  # type: ignore[attr-defined]

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/state':
            task_id = parse_qs(parsed.query).get('task_id', [None])[0]
            self._send_json(self.service.snapshot(task_id))
            return
        if parsed.path == '/api/tasks':
            self._send_json(self.service.tasks_snapshot())
            return
        if parsed.path.startswith('/api/tasks/') and parsed.path.endswith('/download'):
            parts = parsed.path.strip('/').split('/')
            task_id = parts[2] if len(parts) >= 4 else ''
            asset_id = parse_qs(parsed.query).get('asset', [''])[0]
            asset = self.service.get_asset(task_id, asset_id)
            if not asset:
                self._send_json({'error': 'asset not found'}, HTTPStatus.NOT_FOUND)
                return
            self._send_asset(asset)
            return
        if parsed.path.startswith('/api/tasks/'):
            task_id = parsed.path.rstrip('/').split('/')[-1]
            snapshot = self.service.snapshot(task_id)
            if snapshot.get('task_id') != task_id:
                self._send_json({'error': 'task not found'}, HTTPStatus.NOT_FOUND)
            else:
                self._send_json(snapshot)
            return
        if parsed.path == '/api/events':
            self._handle_events()
            return
        if parsed.path == '/':
            self.path = '/index.html'
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/tasks':
            try:
                payload = self._read_json()
                task_id = self.service.create_task(payload)
                self._send_json(self.service.snapshot(task_id))
            except Exception as exc:
                self._send_json({'error': str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path.startswith('/api/tasks/') and parsed.path.endswith('/stop'):
            task_id = parsed.path.split('/')[-2]
            self.service.stop(task_id)
            self._send_json(self.service.snapshot(task_id))
            return
        if parsed.path.startswith('/api/tasks/') and parsed.path.endswith('/delete'):
            task_id = parsed.path.split('/')[-2]
            result = self.service.delete_task(task_id)
            if 'error' in result:
                self._send_json(result, HTTPStatus.NOT_FOUND)
            else:
                self._send_json(result)
            return
        if parsed.path == '/api/train/start':
            try:
                payload = self._read_json()
                state = self.service.start(payload)
                self._send_json(state)
            except Exception as exc:
                self._send_json({'error': str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == '/api/train/stop':
            payload = self._read_json()
            task_id = payload.get('task_id') if isinstance(payload, dict) else None
            self.service.stop(task_id)
            self._send_json(self.service.snapshot(task_id))
            return
        if parsed.path == '/api/report/deepseek':
            try:
                payload = self._read_json()
                task_id = payload.get('task_id') if isinstance(payload, dict) else None
                result = self.service.generate_deepseek_report(task_id)
                self._send_json(result)
            except Exception as exc:
                self._send_json({'error': str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == '/api/open/models':
            self.service.open_models_folder()
            self._send_json({'ok': True, 'path': str(MODEL_DIR)})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get('content-length', '0'))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode('utf-8'))

    def _send_json(self, data: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(_jsonable(data), ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_asset(self, asset: Dict[str, Any]):
        path = Path(asset.get('path') or '').resolve()
        if not path.exists() or not self._is_download_allowed(path):
            self._send_json({'error': 'asset file is unavailable'}, HTTPStatus.NOT_FOUND)
            return
        filename = path.name
        content_type = 'application/json' if path.suffix == '.json' else 'application/octet-stream'
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(size))
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _is_download_allowed(self, path: Path) -> bool:
        return any(root == path or root in path.parents for root in DOWNLOAD_ROOTS)

    def _handle_events(self):
        q = self.hub.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        try:
            self._write_sse('state', self.service.snapshot())
            while True:
                try:
                    event = q.get(timeout=20)
                    event_name = event.pop('event')
                    self._write_sse(event_name, event)
                except queue.Empty:
                    self._write_comment('keepalive')
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.hub.unsubscribe(q)

    def _write_comment(self, text: str):
        self.wfile.write(f": {text}\n\n".encode('utf-8'))
        self.wfile.flush()

    def _write_sse(self, event: str, data: Dict[str, Any]):
        body = json.dumps(_jsonable(data), ensure_ascii=False)
        self.wfile.write(f"event: {event}\n".encode('utf-8'))
        for line in body.splitlines() or ['{}']:
            self.wfile.write(f"data: {line}\n".encode('utf-8'))
        self.wfile.write(b'\n')
        self.wfile.flush()


class IndustrialHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, service: TrainingService, hub: EventHub):
        super().__init__(server_address, handler_class)
        self.service = service
        self.hub = hub


def main(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8600)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    _load_local_env()
    if os.environ.get('DEEPSEEK_API_KEY', '').strip():
        LOGGER.info("DeepSeek API Key loaded")
    else:
        LOGGER.warning("DEEPSEEK_API_KEY is not set; DeepSeek reports will be disabled")
    hub = EventHub()
    service = TrainingService(hub)
    server = IndustrialHTTPServer((args.host, args.port), RequestHandler, service, hub)
    url = f"http://{args.host}:{args.port}"
    LOGGER.info("Industrial UAV MARL console: %s", url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        service.stop()
        LOGGER.info("Stopping server")
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
