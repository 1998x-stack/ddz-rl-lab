"""Explicit local training scheduler commands; no background service is started implicitly."""
from __future__ import annotations

import argparse
import json
from .jobs import JobQueue


def main() -> None:
    p = argparse.ArgumentParser(description='DDZ local durable training job queue')
    p.add_argument('--jobs-dir', default='runs/jobs')
    p.add_argument('--runs-dir', default='runs')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('submit')
    s.add_argument('--spec', required=True)
    s.add_argument('--resume', action='store_true')
    s.add_argument('--parent-experiment')
    q = sub.add_parser('status')
    q.add_argument('job_id')
    sub.add_parser('list')
    l = sub.add_parser('lineage')
    l.add_argument('experiment_id')
    w = sub.add_parser('work')
    w.add_argument('--parallel', type=int, default=1)
    w.add_argument('--once', action='store_true', help='drain queue and exit (without flag, poll for jobs)')
    args = p.parse_args()
    jobs = JobQueue(args.jobs_dir, args.runs_dir)
    if args.command == 'submit':
        output = jobs.submit(args.spec, resume=args.resume, parent_experiment=args.parent_experiment)
    elif args.command == 'status':
        output = jobs.get(args.job_id)
    elif args.command == 'list':
        output = jobs.list()
    elif args.command == 'lineage':
        output = jobs.lineage(args.experiment_id)
    else:
        output = jobs.work(parallel=args.parallel, once=args.once)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
