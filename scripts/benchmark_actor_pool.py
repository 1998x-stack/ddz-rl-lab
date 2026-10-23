#!/usr/bin/env python3
"""Repeatable actor collection performance baseline (no model-strength claims).

Comparisons hold seed set, initialized weights, torch threads and exploration
constant. JSON includes seed-stable result signatures and approximate RSS.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from ddz_training.actors import ActorPool

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--workers',type=int,nargs='+',default=[1,2])
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--hidden',type=int,default=32)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 1<=args.episodes<=20000 or not 1<=args.hidden<=1024 or (
            not args.workers or any(not 1<=w<=32 for w in args.workers)):
        parser.error('episodes, hidden or worker count out of bounds')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    bundle=RoleBundle(NetworkSpec.v2(args.hidden))
    weights={role: {name:t.detach().cpu().clone() for name,t in net.state_dict().items()}
             for role,net in bundle.models.items()}
    seeds=list(range(args.seed,args.seed+args.episodes))
    reports=[]
    for workers in args.workers:
        started=time.monotonic()
        with ActorPool(workers) as pool:
            results=pool.collect(seeds,spec=bundle.spec,weights=weights,historical=[],
                                 epsilon=0.,bid_epsilon=0.,max_steps=1200,
                                 historical_probability=0.,redeal_penalty=.2)
        elapsed=time.monotonic()-started
        signature=json.dumps([(r.seed,r.decisions,r.truncated,len(r.samples),
                              sum(x.return_value for x in r.samples)) for r in results],
                             sort_keys=True)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        child_rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        reports.append({'workers':workers,'episodes':len(results),'wall_s':elapsed,
                        'episodes_per_s':len(results)/elapsed,
                        'decisions_per_s':sum(r.decisions for r in results)/elapsed,
                        'samples_per_s':sum(len(r.samples) for r in results)/elapsed,
                        'parent_maxrss_kb':rss,'max_child_rss_kb':child_rss,
                        'outcome_signature_sha256':hashlib.sha256(signature.encode()).hexdigest()})
    comparable=len(set(x['outcome_signature_sha256'] for x in reports))==1
    result={'schema':'ddz-actorpool-benchmark-v1','seed':args.seed,
            'episodes_per_config':args.episodes,'torch_threads':1,
            'deterministic_across_workers':comparable,
            'results':reports,'caveats':'One host only; RSS is OS high water mark, '
                       'not a strict aggregate GPU/worker memory budget.'}
    if not comparable:
        raise RuntimeError('actor seeds produced different decisions across workers')
    payload=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:
            f.write(payload)
    print(payload,end='')

if __name__=='__main__':
    main()
