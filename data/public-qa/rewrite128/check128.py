#!/usr/bin/env python3
"""check128.py PARAGRAPHS_JSON NEW_JSON EXPECT_N [PRIOR_JSON ...]
NEW/PRIOR = {pid:[rewrites]}. Checks every pair involving a NEW rewrite (vs original, priors, other new). Longest common run of non-entity words <=5."""
import json, sys, re, collections
P = {r['pid']: r for r in json.load(open(sys.argv[1]))}; N = json.load(open(sys.argv[2])); EN = int(sys.argv[3]); X = [json.load(open(f)) for f in sys.argv[4:]]
MAXRUN = 5; _c = [0]
def tok(s):
    s = re.sub(r'"[^"]*"', ' ', s); out = []
    for w in re.findall(r"[\w'-]+", s):
        if w[0].isupper() or any(ch.isdigit() for ch in w): _c[0] += 1; out.append('#E%d' % _c[0])
        else: out.append(w.lower())
    return out
def blank(s, golds):
    for g in sorted(golds, key=len, reverse=True): s = re.sub(re.escape(g.strip(' ,.;:')), ' ANSWERX ', s, flags=re.I)
    return s
def run(a, b):
    qs, cs = tok(a), tok(b); pos = collections.defaultdict(list)
    for j, t in enumerate(cs): pos[t].append(j)
    best = 0
    for i, t in enumerate(qs):
        for j in pos.get(t, []):
            k = 0
            while i+k < len(qs) and j+k < len(cs) and qs[i+k] == cs[j+k]: k += 1
            best = max(best, k)
    return best
bad = 0; runs = []
for pid, rs in N.items():
    p = P[pid]; msgs = []
    prior = [p['context']] + [t for x in X for t in x.get(pid, [])]
    if len(rs) != EN: msgs.append(f'need {EN} rewrites, got {len(rs)}')
    for i, r in enumerate(rs, 1):
        for g in p['golds']:
            if g.strip(' ,.;:').lower() not in r.lower(): msgs.append(f'r{i}: gold "{g}" missing')
        n0, n = len(p['context'].split()), len(r.split())
        if not (0.6 * n0 <= n <= 1.5 * n0): msgs.append(f'r{i}: length {n} vs {n0}')
    pb = [blank(t, p['golds']) for t in prior]; nb = [blank(t, p['golds']) for t in rs]
    for i, a in enumerate(nb, 1):
        for j, b in enumerate(pb):
            k = run(a, b); runs.append(k)
            if k > MAXRUN: msgs.append(f'new r{i} vs prior#{j} run {k}')
        for j in range(i, len(nb)):
            k = run(a, nb[j]); runs.append(k)
            if k > MAXRUN: msgs.append(f'new r{i} vs new r{j+1} run {k}')
    if msgs: bad += 1; print(pid, p['title'], '|', '; '.join(msgs))
print('paragraphs', len(N), 'failed', bad, '| run max/median', max(runs) if runs else 0, sorted(runs)[len(runs)//2] if runs else 0)
