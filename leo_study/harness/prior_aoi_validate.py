"""Validate prior_aoi_extract output against (1) the archived consistency_event summaries (independent of replay),
(2) the audit ordering_pairs_verified, (3) an un-hooked replay's full pair lists (AoI binning, marginal over active)."""
import gzip, json, sys
from collections import Counter
from pathlib import Path
run, name = Path(sys.argv[1]).resolve(), sys.argv[2]
sys.path.insert(0, str(run / 'code'))
from panama_uav.remote_audit import replay_trace
cache = {(r['condition'], r['case_id']): r for r in map(json.loads, open(f'/workspace/leo_study/figures/aoi_semantic/prior/{name}_episodes_aoi.jsonl'))}
audits = {(a['condition'], a['case_id']): a for a in map(json.loads, open(run / 'val_audits.jsonl'))} if (run / 'val_audits.jsonl').exists() else {}
ok_all = True
for f in sorted((run / 'episodes').rglob('*.json.gz')):
    key = (f.parent.name, f.name[:-8]); r = cache[key]; F = r['aoi_fields']
    T = Counter()
    for row in r['aoi']:
        d = dict(zip(F, row)); c = d['count']
        T['pairs'] += c; T['relevant'] += c * d['relevant']; T['e2e'] += c * d['e2e_error']; T['ul'] += c * d['ul_error']; T['dl'] += c * d['dl_error']
        T['active'] += c * d['active']
    tr = json.load(gzip.open(f, 'rt'))
    S = Counter(); has_active = True
    for e in tr['remote_events']:
        if e['event'] != 'consistency_event': continue
        s = e['summary']; S['events'] += 1
        S['pairs'] += s['directed_pairs']; S['relevant'] += s['relevant_pairs']; S['e2e'] += s['ordering_errors']
        S['ul'] += s['uplink_errors']; S['dl'] += s['downlink_errors']
        if 'active_peer_aoi_samples' in s: S['active'] += s['active_peer_aoi_samples']
        else: has_active = False
    fr = {f_['step']: f_ for f_ in tr['frames']}; A = Counter()
    for e in tr['remote_events']:
        if e['event'] != 'consistency_event': continue
        for lag in (0, 1):
            f_ = fr.get(e['cycle'] + lag)
            if f_ is None: continue
            k = sum(not v for v in f_['completed_flags'].values()); A[lag] += k * (k - 1)
    res = replay_trace(tr, None); U = Counter()
    for snap in res['pair_consistency']:
        for p in snap['pairs']:
            U[(min(int(max(p['self_age_seconds'], p['peer_age_seconds']) / .05), 400), int(p['relevant']), int(p['e2e_ordering_error']))] += 1
    M = Counter()
    for row in r['aoi']:
        d = dict(zip(F, row)); M[(d['aoi_bin'], d['relevant'], d['e2e_error'])] += d['count']
    chk = {k: (T[k], S[k]) for k in ('pairs', 'relevant', 'e2e', 'ul', 'dl')}
    print('   active from table', T['active'], ' frames completed_flags (step=cycle, cycle+1):', A[0], A[1], ' e2e/relevant:', T['e2e'], T['relevant'])
    if has_active: chk['active'] = (T['active'], S['active'])
    if key in audits: chk['audit_ordering_pairs_verified'] = (T['pairs'], audits[key]['ordering_pairs_verified'])
    chk['unhooked_replay_table'] = (sum(M.values()), sum(U.values()) if M == U else f'MISMATCH {sum(U.values())}')
    good = all(a == b for a, b in chk.values()); ok_all &= good
    print(key, 'events', S['events'], 'delta', r['delta_seconds'], 'active_in_summary' if has_active else 'no active summary',
          'PASS' if good else 'FAIL', chk)
    del tr, res
print('ALL PASS' if ok_all else 'SOME FAIL')
