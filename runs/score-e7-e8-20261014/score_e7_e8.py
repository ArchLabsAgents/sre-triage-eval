"""Primary (six rounds) and sensitivity (original rule: E7 x3 + r3-seed-2425) scores for the blind 7B arm, paper section 7.2 (table T4a).

Run from tools/triage-eval with Python 3.12:
    PYTHONPATH=. python3.12 ../../runs/score-e7-e8-20261014/score_e7_e8.py <output-dir>
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
RODADAS = [('e7-r2b-20261001','s1','r2-seed-1052'),('e7-r2b-20261001','s2','r2-seed-1140'),('e7-r2b-20261001','s3','r2-seed-1283'),
           ('e8-r3-20261005','s1','r3-seed-1627'),('e8-r3-20261005','s2','r3-seed-1695'),('e8-r3-20261005','s3','r3-seed-2425')]
ORIGINAL = {'r2-seed-1052','r2-seed-1140','r2-seed-1283','r3-seed-2425'}
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
_, h = dry_run()
if h != '1b9ff1c5a103c34d4d5353c3e6fda16df8ff41c95dd7f46dcc0cafae456ab3ae':
    raise SystemExit(f'dry-run hash differs from the pre-registered one: {h}')
eps = {}
for run, s, sched in RODADAS:
    d = R/run/'results'
    res = build_episodes(_read_jsonl(d/f'{s}-r2.events.jsonl'), _read_jsonl(d/f'{s}-r2.triage.jsonl', tolerant=True),
                         schedule=json.loads(Path(f'scenarios/schedules/{sched}.json').read_text()))
    eps[sched] = res.episodes
for name, keep in (('primario', set(eps)), ('sensibilidade', ORIGINAL)):
    rep = score([e for k in sorted(keep) for e in eps[k]])
    rep['rodadas'] = sorted(keep)
    (out/f'{name}.json').write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
    print(name, report_hash(rep)[:16], json.dumps({k: rep[k] for k in ('counts','h3a','h3c','recall')}, default=str))
    print('  h3b', json.dumps({k:v for k,v in rep['h3b'].items()}, default=str)[:600])
    print('  rubric', rep['rubric'], '\n  disp', rep['disposicao'], '\n  urg', rep['urgencia'])
