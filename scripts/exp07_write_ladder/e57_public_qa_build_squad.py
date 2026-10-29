#!/usr/bin/env python3
"""Freeze the SQuAD dev-v1.1 item file for job 440 (E57 §122-P). CPU only, stdlib only.
usage: e57_public_qa_build_squad.py DEV_JSON OUT_JSON
All rules are fixed below and hashed into the output; change a rule -> new hash -> new file."""
import json, sys, re, hashlib, collections

RULES = dict(version=1, seed="e57-pubqa-squad-v1", n_items=1000, n_distractors=15,
             exclude_titles=["Super_Bowl_50", "Warsaw"],  # in the existing 8192-token OUT tail
             max_run=3,        # longest common word run (question vs paragraph) <= 3
             max_ans_words=6, min_ans_chars=2, max_title_frac=1/46*2)
H = lambda s: hashlib.md5((RULES["seed"] + s).encode()).hexdigest()
tok = lambda s: re.findall(r"\w+", s.lower())
MONTHS = "january february march april may june july august september october november december".split()

def longest_run(q, c):
    qs, cs = tok(q), tok(c); pos = collections.defaultdict(list)
    for j, t in enumerate(cs): pos[t].append(j)
    best = 0
    for i, t in enumerate(qs):
        for j in pos.get(t, []):
            k = 0
            while i+k < len(qs) and j+k < len(cs) and qs[i+k] == cs[j+k]: k += 1
            best = max(best, k)
    return best

def atype(a):
    w = tok(a)
    if re.fullmatch(r"[\d,.\s%$]+", a.strip()): return "NUM"
    if any(m in w for m in MONTHS) or re.search(r"\b(1[0-9]|20)\d\d\b", a): return "DATE"
    if a[:1].isupper(): return "CAP"
    return "LOW"

def wh(q):
    ql = q.lower()
    for k in ["how many", "how much", "how long", "how old", "what year", "when", "who", "whom", "where", "which", "why", "how", "what"]:
        if re.search(r"\b" + k + r"\b", ql): return k
    return "other"

def main(src, out):
    d = json.load(open(src, encoding="utf-8"))["data"]
    pool = []; art_of = {}
    for a in d:
        for p in a["paragraphs"]:
            for qa in p["qas"]:
                ans = [x["text"] for x in qa["answers"]]
                pool.append(dict(id=qa["id"], title=a["title"], q=qa["question"].strip(), ctx=p["context"], answers=ans))
    elig = []
    for r in pool:
        g = r["answers"][0]
        if r["title"] in RULES["exclude_titles"]: continue
        if len(g) < RULES["min_ans_chars"] or len(g.split()) > RULES["max_ans_words"]: continue
        if g.lower() not in r["ctx"].lower(): continue
        r["run"] = longest_run(r["q"], r["ctx"])
        if r["run"] > RULES["max_run"]: continue
        elig.append(r)
    # round-robin over titles in hash order (stratified, deterministic)
    by = collections.defaultdict(list)
    for r in sorted(elig, key=lambda r: H(r["id"])): by[r["title"]].append(r)
    titles = sorted(by, key=lambda t: H(t)); chosen = []
    while len(chosen) < RULES["n_items"] and any(by.values()):
        for t in titles:
            if by[t] and len(chosen) < RULES["n_items"]: chosen.append(by[t].pop(0))
    # distractor candidates: first answer of every dev question (excluded titles removed)
    answers = [dict(a=r["answers"][0], title=r["title"], ty=atype(r["answers"][0]), wh=wh(r["q"])) for r in pool if r["title"] not in RULES["exclude_titles"]]
    items = []
    for r in chosen:
        gv = {a.lower().strip() for a in r["answers"]}; gl = r["answers"][0].lower()
        ty, w = atype(r["answers"][0]), wh(r["q"]); ctxl = r["ctx"].lower(); seen = set(); cand = []
        for x in answers:
            al = x["a"].lower().strip()
            if x["ty"] != ty or al in seen or al in gv or al in gl or gl in al or al in ctxl: continue
            if len(x["a"]) < 2 or len(x["a"].split()) > RULES["max_ans_words"]: continue
            seen.add(al)
            cand.append(((x["title"] != r["title"]), (x["wh"] != w), abs(len(x["a"].split()) - len(r["answers"][0].split())), H(r["id"] + al), x["a"]))
        cand.sort(); dis = [c[-1] for c in cand[:RULES["n_distractors"]]]
        if len(dis) < RULES["n_distractors"]: continue
        allc = sorted([r["answers"][0]] + dis, key=lambda s: H(r["id"] + "|ord|" + s))
        items.append(dict(id=r["id"], title=r["title"], frame=r["q"], context=r["ctx"], gold=r["answers"][0],
                          variants=sorted(set(r["answers"])), candidates=allc, gold_index=allc.index(r["answers"][0]),
                          type=ty, wh=w, run=r["run"]))
    body = dict(rules=RULES, n_items=len(items), items=items)
    s = json.dumps(body, ensure_ascii=False, sort_keys=True)
    body["sha256"] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(body, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("items", len(items), "of", len(elig), "eligible; titles", len({i["title"] for i in items}), "sha256", body["sha256"][:8])
    print("types", collections.Counter(i["type"] for i in items))

if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
