"""Paper section 8 (T5): cost per triage, local against API, from the window artifacts.

Run from tools/triage-eval with Python 3.12:
    PYTHONPATH=. python3.12 ../../runs/t5-cost-20261030/compute_t5.py <output.json>

Local: harness.metrics.token_slice_cost between the first and last event of the window journals,
at the measured list price of US$0.5067/h. API: sum of cost_usd from the per-request log within the
same bounds. Everything here is list price, never an invoice. The curve and the break-even are a
projection computed from these numbers, and the JSON labels them that way.
"""
import json
import statistics
import sys
from pathlib import Path

from harness.metrics import load_episode_usage, load_vllm_queue_samples, token_slice_cost

R = Path('../../runs')
USD_PER_HOUR = 0.5067
# Sonnet 5.5 list price per MTok (amendment E8-API), used only in the no-cache projection row.
SONNET_INPUT, SONNET_OUTPUT = 2.0, 10.0
# The log was written with cache reads at US$0.20/MTok; the pricing page read on 2026-10-30 says
# 0.05x input, US$0.10. The log is not changed: the repriced row sits alongside until reconciled with the invoice.
LOGGED_CACHE_READ, PUBLISHED_CACHE_READ = 0.20, 0.10

LOCAL = {
    '7b-blind-e7': ('e7-r2b-20261001', ['s1', 's2', 's3']),
    '7b-blind-e8': ('e8-r3-20261005', ['s1', 's2', 's3']),
    '7b-sighted': ('7b-sighted-r2-20261017', ['s1']),
    '7b-budget-short': ('7b-budget-r2short-20261019', ['s1']),
    '9b-budget-short': ('9b-budget-r2short-20261020', ['s1']),
    '9b-contract-short': ('9b-contract-r2short-20261029', ['s1']),
}
API = {'sonnet-blind-w1': 'api-r2-20261015', 'sonnet-sighted-w2': 'api-r2-w2-20261016'}


def span(results, seeds):
    ts = []
    for s in seeds:
        for line in (results/f'{s}-r2.events.jsonl').read_text().splitlines():
            if line.strip():
                ev = json.loads(line)
                ts.append(ev.get('ts', ev.get('t', ev.get('timestamp'))))
    ts = [t for t in ts if isinstance(t, (int, float))]
    return min(ts), max(ts)


def usage_rows(results, seeds, start, end):
    rows = [u for s in seeds for u in load_episode_usage(results/f'{s}-r2.usage.jsonl')]
    return [u for u in rows if u.t_end >= start and u.t_start <= end]


def mtta(rows):
    return round(statistics.median(u.t_end - u.t_start for u in rows), 2) if rows else None


def tokens(rows):
    return sum(u.prompt_tokens for u in rows), sum(u.completion_tokens for u in rows)


out = {'usd_per_hour': USD_PER_HOUR, 'price_basis': 'list price, not invoice', 'local': {}, 'api': {}}
for name, (run, seeds) in LOCAL.items():
    results = R/run/'results'
    start, end = span(results, seeds)
    rows = usage_rows(results, seeds, start, end)
    sl = token_slice_cost(load_vllm_queue_samples(results/'window.vllm-queue.jsonl'), rows, start, end,
                          price_usd_per_hour=USD_PER_HOUR)
    usd_eps = [e.usd for e in sl.episodes if e.usd is not None]
    prompt, completion = tokens(rows)
    out['local'][name] = {
        'hours': round((end - start)/3600, 2), 'seeds': len(seeds), 'triages': len(rows),
        'triages_per_day_per_seed': round(len(rows)/len(seeds)/((end - start)/86400), 1),
        'usd_window': round(sl.usd_window, 2),
        'usd_per_triage_all_in': round(sl.usd_window/len(rows), 4),
        'triages_with_tokens': sum(u.completion_tokens > 0 for u in rows),
        'usd_per_triage_with_tokens': round(sl.usd_window/sum(u.completion_tokens > 0 for u in rows), 4),
        'usd_per_triage_token_slice_mean': round(statistics.mean(usd_eps), 4) if usd_eps else None,
        'usd_unattributed': round(sl.usd_unattributed, 2) if sl.usd_unattributed is not None else None,
        'slice_reason': sl.reason,
        'prompt_tokens': prompt, 'completion_tokens': completion,
        'tokens_per_triage': round((prompt + completion)/len(rows)),
        'mtta_median_s': mtta(rows),
        'projection_same_tokens_at_sonnet_list_uncached_usd_per_triage':
            round((prompt*SONNET_INPUT + completion*SONNET_OUTPUT)/1e6/len(rows), 4),
    }
for name, run in API.items():
    results = R/run/'results'
    start, end = span(results, ['s1'])
    rows = usage_rows(results, ['s1'], start, end)
    reqs = [json.loads(x) for x in (results/'s1-comparator-usage.jsonl').read_text().splitlines() if x.strip()]
    inside = [r for r in reqs if start <= r['ts'] <= end]
    usd = sum(r['cost_usd'] for r in inside)
    cache_read = sum(r['cache_read_input_tokens'] for r in inside)
    cache_write = sum(r['cache_creation_input_tokens'] for r in inside)
    plain = sum(r['input_tokens'] for r in inside)
    out['api'][name] = {
        'hours': round((end - start)/3600, 2), 'requests': len(inside), 'triages': len(rows),
        'triages_per_day': round(len(rows)/((end - start)/86400), 1),
        'usd_api': round(usd, 2), 'usd_api_per_triage': round(usd/len(rows), 4),
        'usd_api_per_triage_no_cache_at_list':
            round(((plain + cache_read + cache_write)*SONNET_INPUT + sum(r['output_tokens'] for r in inside)*SONNET_OUTPUT)/1e6/len(rows), 4),
        'usd_api_per_triage_cache_read_repriced':
            round((usd - cache_read*(LOGGED_CACHE_READ - PUBLISHED_CACHE_READ)/1e6)/len(rows), 4),
        'input_tokens_uncached': plain, 'cache_read_tokens': cache_read, 'cache_write_tokens': cache_write,
        'output_tokens': sum(r['output_tokens'] for r in inside),
        'mtta_median_s': mtta(rows),
    }

day = USD_PER_HOUR*24
sighted = out['api']['sonnet-sighted-w2']['usd_api_per_triage']
blind = out['api']['sonnet-blind-w1']['usd_api_per_triage']
curve = {}
for per_day in (10, 25, 50, 60, 96, 100, 150, 200, 500, 1000):
    curve[per_day] = {'local_gpu_usd_per_triage': round(day/per_day, 4), 'api_sighted_usd_per_triage': sighted,
                      'api_sighted_usd_per_day': round(sighted*per_day, 2)}
out['projection'] = {
    'label': 'arithmetic projection on measured unit costs, not a measurement',
    'local_usd_per_day': round(day, 2),
    # Share of the VM price that would have to belong to the cluster host, not the model, for local to break even.
    'vm_share_not_model_at_tie_vs_api_sighted':
        round(1 - sighted/out['local']['7b-sighted']['usd_per_triage_all_in'], 3),
    'breakeven_triages_per_day_vs_api_sighted': round(day/sighted, 1),
    'breakeven_triages_per_day_vs_api_sighted_cache_read_repriced':
        round(day/out['api']['sonnet-sighted-w2']['usd_api_per_triage_cache_read_repriced'], 1),
    'breakeven_triages_per_day_vs_api_blind': round(day/blind, 1),
    'breakeven_triages_per_day_vs_budget_tokens_at_sonnet_list_uncached':
        round(day/out['local']['7b-budget-short']['projection_same_tokens_at_sonnet_list_uncached_usd_per_triage'], 1),
    'curve': curve,
}
Path(sys.argv[1]).write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
print(json.dumps(out, indent=1, sort_keys=True))
