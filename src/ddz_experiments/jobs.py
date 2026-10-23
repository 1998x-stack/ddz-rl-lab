"""SQLite-backed local experiment scheduler with bounded subprocess concurrency.

This is a foreground supervisor: `ddz-jobs work` must be run by an operator
(e.g. systemd). No web endpoint starts arbitrary processes. On supervisor crash,
RUNNING jobs become INTERRUPTED, never automatically requeued.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid
from typing import Any, Iterator

from loguru import logger
from .runner import ExperimentRunner, _name, _read_spec


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobError(ValueError):
    pass


SCHEMA = '''
CREATE TABLE IF NOT EXISTS jobs (
 job_id TEXT PRIMARY KEY,
 experiment_id TEXT NOT NULL,
 parent_experiment TEXT,
 source_sha256 TEXT NOT NULL,
 spec_json TEXT NOT NULL,
 resume INTEGER NOT NULL CHECK (resume IN (0,1)),
 status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed','interrupted')),
 created_at TEXT NOT NULL,
 started_at TEXT,
 finished_at TEXT,
 exit_code INTEGER,
 error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_by_state ON jobs(status, created_at);
'''


class JobQueue:
    def __init__(self, jobs_dir: str | Path, runs_dir: str | Path):
        if Path(jobs_dir).is_symlink():
            raise JobError('unsafe jobs directory')
        self.root = Path(jobs_dir).resolve()
        self.runs_dir = Path(runs_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root == self.runs_dir:
            raise JobError('unsafe jobs directory')
        self.db_path = self.root / 'jobs.sqlite3'
        if self.db_path.is_symlink():
            raise JobError('jobs database must not be a symlink')
        with self._db() as db:
            db.executescript(SCHEMA)
            # In-place upgrade of v10 SQLite queues without deleting submitted jobs.
            existing = {row['name'] for row in db.execute('PRAGMA table_info(jobs)')}
            migrations = {
                'cpu_threads': 'INTEGER NOT NULL DEFAULT 1',
                'gpu_id': 'INTEGER',
                'timeout_seconds': 'INTEGER NOT NULL DEFAULT 0',
                'worker_pid': 'INTEGER',
                'retry_of': 'TEXT',
            }
            for column, declaration in migrations.items():
                if column not in existing:
                    db.execute(f'ALTER TABLE jobs ADD COLUMN {column} {declaration}')

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA busy_timeout=15000')
            db.execute('PRAGMA journal_mode=WAL')
            yield db
            db.commit()
        finally:
            db.close()

    def submit(self, spec_path: str | Path, *, resume: bool = False,
               parent_experiment: str | None = None, cpu_threads: int = 1,
               gpu_id: int | None = None, timeout_seconds: int = 0) -> dict[str, Any]:
        # Strict v9 schema validation remains the single source of truth.
        spec = _read_spec(Path(spec_path))
        experiment = _name(spec['id'])
        self._check_resources(spec, cpu_threads, gpu_id, timeout_seconds)
        if parent_experiment is not None:
            _name(parent_experiment)
            if parent_experiment == experiment:
                raise JobError('experiment cannot be its own parent')
            try:
                parent = ExperimentRunner(self.runs_dir).status(parent_experiment)
            except FileNotFoundError as exc:
                raise JobError('parent experiment does not exist') from exc
            if parent.get('status') != 'finished':
                raise JobError('parent experiment must have finished')
        if resume:
            try:
                ExperimentRunner(self.runs_dir).status(experiment)
            except FileNotFoundError as exc:
                raise JobError('cannot resume an unknown experiment') from exc
        elif (self.runs_dir / experiment).exists() and any((self.runs_dir / experiment).iterdir()):
            raise JobError('experiment exists; use resume')
        raw = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False)
        sha = hashlib.sha256(raw.encode('utf-8')).hexdigest()
        job_id = uuid.uuid4().hex
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM jobs WHERE experiment_id=? AND status IN (\'queued\',\'running\')',
                          (experiment,)).fetchone():
                raise JobError('experiment already queued or running')
            db.execute('''INSERT INTO jobs(job_id,experiment_id,parent_experiment,source_sha256,
                       spec_json,resume,status,created_at,cpu_threads,gpu_id,timeout_seconds)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                       (job_id, experiment, parent_experiment, sha, raw, int(resume),
                        'queued', _utc(), cpu_threads, gpu_id, timeout_seconds))
        logger.bind(job_id=job_id, experiment=experiment).info('training_job_queued')
        return self.get(job_id)

    @staticmethod
    def _check_resources(spec: dict, cpu_threads: int, gpu_id: int | None,
                         timeout_seconds: int) -> None:
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 32:
            raise JobError('cpu_threads must be 1..32')
        if type(timeout_seconds) is not int or not 0 <= timeout_seconds <= 86400:
            raise JobError('timeout_seconds must be 0..86400')
        if gpu_id is not None and (type(gpu_id) is not int or not 0 <= gpu_id <= 15):
            raise JobError('gpu_id must be 0..15')
        device = spec['training'].get('device', 'cpu')
        if (device == 'cuda') != (gpu_id is not None):
            raise JobError('CUDA jobs require a GPU slot; CPU/MPS jobs cannot reserve a GPU')

    def retry(self, job_id: str) -> dict[str, Any]:
        """Explicit manual retry from a hash-verified training checkpoint.

        Retries only the unfinished part of the original invocation. Never
        requeues an orphan automatically or reuses a possibly running PID.
        """
        self.get(job_id)  # Validate shape and existence.
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if row['status'] not in {'failed', 'interrupted'}:
                raise JobError('only failed/interrupted jobs can be retried')
            if db.execute("SELECT 1 FROM jobs WHERE retry_of=? AND status IN ('queued','running','succeeded')",
                          (job_id,)).fetchone():
                raise JobError('job already has an active or successful retry')
            if db.execute("SELECT 1 FROM jobs WHERE experiment_id=? AND status IN ('queued','running')",
                          (row['experiment_id'],)).fetchone():
                raise JobError('experiment is already queued or running')
            if row['status'] == 'interrupted' and row['worker_pid']:
                try:
                    os.kill(row['worker_pid'], 0)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    raise JobError('old worker PID cannot be verified as stopped') from exc
                else:
                    raise JobError('old worker PID is still present; refuse duplicate execution')
            run = self.runs_dir / row['experiment_id']
            if run.is_symlink():
                raise JobError('unsafe experiment directory')
            path = run / 'latest.pt'
            sidecar = path.with_suffix('.pt.json')
            if path.is_symlink() or sidecar.is_symlink() or not path.is_file() or not sidecar.is_file():
                raise JobError('no verified checkpoint available; cannot retry safely')
            from ddz_training.checkpoints import _digest
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            if (metadata.get('checkpoint') != 'latest.pt' or
                    metadata.get('sha256') != _digest(path) or metadata.get('inference_only') is not False):
                raise JobError('checkpoint verification failed')
            state = ExperimentRunner(self.runs_dir).status(row['experiment_id'])
            if state.get('status') == 'finished':
                raise JobError('experiment already finished')
            spec = json.loads(row['spec_json'])
            baseline = int(state.get('previous_episodes', 0))
            progressed = int(metadata['episodes_seen']) - baseline
            remaining = int(spec['training']['episodes']) - progressed
            if progressed < 0 or remaining <= 0:
                raise JobError('checkpoint progress does not permit retry')
            spec['training']['episodes'] = remaining
            raw = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False)
            digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()
            retry_id = uuid.uuid4().hex
            db.execute('''INSERT INTO jobs(job_id,experiment_id,parent_experiment,
                source_sha256,spec_json,resume,status,created_at,cpu_threads,gpu_id,
                timeout_seconds,retry_of) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                (retry_id,row['experiment_id'],row['parent_experiment'],digest,raw,1,
                 'queued',_utc(),row['cpu_threads'],row['gpu_id'],row['timeout_seconds'],job_id))
        logger.bind(job_id=retry_id,retry_of=job_id).info('training_job_retry_queued')
        return self.get(retry_id)

    def get(self, job_id: str) -> dict[str, Any]:
        if not isinstance(job_id, str) or len(job_id) != 32 or not all(c in '0123456789abcdef' for c in job_id):
            raise JobError('invalid job id')
        with self._db() as db:
            row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None:
            raise KeyError('unknown job')
        return self._public(row)

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {k: row[k] for k in ('job_id', 'experiment_id', 'parent_experiment',
                 'source_sha256', 'resume', 'status', 'created_at', 'started_at',
                 'finished_at', 'exit_code', 'error', 'cpu_threads', 'gpu_id',
                 'timeout_seconds', 'worker_pid', 'retry_of')}

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise JobError('limit outside 1..500')
        with self._db() as db:
            rows = db.execute('SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT ?',
                              (limit,)).fetchall()
        return [self._public(row) for row in rows]

    def lineage(self, experiment_id: str) -> dict[str, Any]:
        _name(experiment_id)
        with self._db() as db:
            rows = db.execute('SELECT * FROM jobs WHERE experiment_id=? OR parent_experiment=? '
                              'ORDER BY created_at', (experiment_id, experiment_id)).fetchall()
        return {'experiment_id': experiment_id, 'related_jobs': [self._public(x) for x in rows]}

    def _claim_one(self) -> sqlite3.Row | None:
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            busy_gpus = {r['gpu_id'] for r in db.execute(
                "SELECT gpu_id FROM jobs WHERE status='running' AND gpu_id IS NOT NULL")}
            row = next((candidate for candidate in db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at, job_id")
                if candidate['gpu_id'] is None or candidate['gpu_id'] not in busy_gpus), None)
            if row:
                db.execute("UPDATE jobs SET status='running',started_at=? WHERE job_id=? AND status='queued'",
                           (_utc(), row['job_id']))
        return row

    def _set_result(self, job_id: str, code: int, error: str | None = None) -> None:
        with self._db() as db:
            db.execute('''UPDATE jobs SET status=?, finished_at=?, exit_code=?, error=?
                          WHERE job_id=? AND status='running' ''',
                       ('succeeded' if code == 0 else 'failed', _utc(), code, error, job_id))

    def _mark_orphans(self) -> None:
        # Never start another supervisor beside a potentially still-running
        # orphan. A dead supervisor does not necessarily kill its child.
        with self._db() as db:
            orphaned = db.execute("SELECT worker_pid FROM jobs WHERE status='running' AND worker_pid IS NOT NULL").fetchall()
            for row in orphaned:
                try:
                    os.kill(row['worker_pid'], 0)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    raise JobError('previous worker cannot be inspected; manual intervention required') from exc
                else:
                    raise JobError('possibly active orphan worker; manual intervention required')
        with self._db() as db:
            db.execute('''UPDATE jobs SET status='interrupted',finished_at=?,
                          error='supervisor exited; inspect experiment checkpoint before manual resume'
                          WHERE status='running' ''', (_utc(),))

    def work(self, *, parallel: int = 1, once: bool = True, poll_seconds: float = .25) -> dict[str, int]:
        if type(parallel) is not int or not 1 <= parallel <= 16:
            raise JobError('parallel must be 1..16')
        if poll_seconds < .05:
            raise JobError('poll interval too short')
        lock_path = self.root / 'supervisor.lock'
        if lock_path.is_symlink():
            raise JobError('unsafe supervisor lock')
        with lock_path.open('a+b') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise JobError('supervisor already running') from exc
            self._mark_orphans()
            live: dict[str, tuple[subprocess.Popen, Any, float, int]] = {}
            completed = failed = 0
            try:
                while True:
                    for job_id, (proc, log, started, timeout) in list(live.items()):
                        rc = proc.poll()
                        timed_out = rc is None and timeout > 0 and time.monotonic()-started >= timeout
                        if timed_out:
                            self._terminate_group(proc)
                            rc = 124
                        if rc is None:
                            continue
                        log.close()
                        self._set_result(job_id, rc, 'worker timed out' if timed_out else
                                         (None if rc == 0 else 'worker returned nonzero exit code'))
                        completed += 1
                        failed += (rc != 0)
                        del live[job_id]
                        logger.bind(job_id=job_id, exit_code=rc).info('training_job_completed')
                    while len(live) < parallel:
                        row = self._claim_one()
                        if row is None:
                            break
                        job_id = row['job_id']
                        spec_file = self.root / f'{job_id}.spec.json'
                        log_path = self.root / f'{job_id}.log'
                        try:
                            if spec_file.exists() or spec_file.is_symlink():
                                raise JobError('job spec collision')
                            fd = os.open(spec_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                            with os.fdopen(fd, 'w', encoding='utf-8') as fp:
                                fp.write(row['spec_json'] + '\n')
                            fd_log = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                            log = os.fdopen(fd_log, 'wb')
                            cmd = [sys.executable, '-m', 'ddz_experiments.job_worker',
                                   '--spec', str(spec_file), '--runs-dir', str(self.runs_dir),
                                   '--sha256', row['source_sha256']]
                            if row['resume']:
                                cmd.append('--resume')
                            env = os.environ.copy()
                            for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                             'NUMEXPR_NUM_THREADS'):
                                env[variable] = str(row['cpu_threads'])
                            if row['gpu_id'] is not None:
                                env['CUDA_VISIBLE_DEVICES'] = str(row['gpu_id'])
                            else:
                                env['CUDA_VISIBLE_DEVICES'] = ''
                            package_root = str(Path(__file__).resolve().parents[1])
                            env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
                            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                                    stdout=log, stderr=subprocess.STDOUT,
                                                    cwd=self.root, close_fds=True, env=env,
                                                    start_new_session=True)
                            with self._db() as db:
                                db.execute('UPDATE jobs SET worker_pid=? WHERE job_id=?',
                                           (proc.pid, job_id))
                            live[job_id] = (proc, log, time.monotonic(), row['timeout_seconds'])
                            logger.bind(job_id=job_id, pid=proc.pid).info('training_job_started')
                        except Exception as exc:
                            if 'log' in locals() and not log.closed:
                                log.close()
                            self._set_result(job_id, 1, f'launch failed: {type(exc).__name__}')
                            failed += 1
                            completed += 1
                    if once and not live:
                        break
                    time.sleep(poll_seconds)
            except BaseException:
                # Interrupted supervisor does not pretend subprocesses completed.
                for process, log, _, _ in live.values():
                    self._terminate_group(process)
                    log.close()
                self._mark_orphans()
                raise
            return {'completed': completed, 'failed': failed}

    @staticmethod
    def _terminate_group(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
