"""7B with tools and history budget on r2-seed-1052, with api-w2, api-w1 and E7 on the same seed alongside.

Run from tools/triage-eval with Python 3.12:
    PYTHONPATH=. python3.12 ../../runs/7b-sighted-r2-20261017/score_7b_sighted.py <output-dir>
Re-joins the journal and triage.jsonl through harness.episode_bridge (it does not use the
episodes.json collected on the VM) and refuses to run unless the scorer dry-run hash matches the
pre-registered one.
"""
import json
import sys
from pathlib import Path
from harness.scorer import score, report_hash, dry_run
from harness.episode_bridge import build_episodes, _read_jsonl
R = Path('../../runs')
SCHED = 'r2-seed-1052'
RODADAS = [('7b-sighted', '7b-sighted-r2-20261017'), ('api-w2', 'api-r2-w2-20261016'), ('api-w1', 'api-r2-20261015'), ('e7', 'e7-r2b-20261001')]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
_, h = dry_run()
if h != '1b9ff1c5a103c34d4d5353c3e6fda16df8ff41c95dd7f46dcc0cafae456ab3ae':
    raise SystemExit(f'dry-run hash differs from the pre-registered one: {h}')
schedule = json.loads(Path(f'scenarios/schedules/{SCHED}.json').read_text())
for name, run in RODADAS:
    d = R/run/'results'
    res = build_episodes(_read_jsonl(d/'s1-r2.events.jsonl'), _read_jsonl(d/'s1-r2.triage.jsonl', tolerant=True),
                         schedule=schedule)
    rep = score(res.episodes)
    rep['rodadas'] = [SCHED]
    rep['run'] = run
    (out/f'{name}.json').write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
    print(name, report_hash(rep)[:16], json.dumps({k: rep[k] for k in ('counts', 'h3a', 'h3c', 'recall')}, default=str))
    print('  h3b', json.dumps({k: v for k, v in rep['h3b'].items()}, default=str)[:600])
    print('  rubric', rep['rubric'], '\n  disp', rep['disposicao'], '\n  urg', rep['urgencia'])
