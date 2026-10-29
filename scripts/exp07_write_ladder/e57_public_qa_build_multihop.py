#!/usr/bin/env python3
"""Freeze HotpotQA-distractor-dev / MuSiQue-Ans-dev item files for the public-QA reads (E57 §122-P). CPU only, stdlib only.
usage: e57_public_qa_build_multihop.py hotpot HOTPOT_DEV_JSON OUT_JSON   (official hotpot_dev_distractor_v1.json)
       e57_public_qa_build_multihop.py musique MUSIQUE_DEV_JSONL OUT_JSON  (musique_ans_v1.0_dev.jsonl)
Item = question + its SUPPORTING paragraphs only (the window/stream contain just those), gold, 16 candidates."""
import json, sys, re, hashlib, collections

RULES = dict(version=1, seed="e57-pubqa-multihop-v1", n_items=1000, n_distractors=15,
             max_run=3, max_ans_words=6, min_ans_chars=2, hotpot_types=["bridge"], musique_answerable=True,
             unique_support_paragraphs=True)
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

def load(kind, src):
    out = []
    if kind == "hotpot":
        for r in json.load(open(src, encoding="utf-8")):
            if r["type"] not in RULES["hotpot_types"] or r["answer"].strip().lower() in ("yes", "no"): continue
            sup = sorted({t for t, _ in r["supporting_facts"]})
            paras = {t: " ".join(s.strip() for s in ss) for t, ss in r["context"]}
            if any(t not in paras for t in sup): continue
            out.append(dict(id=r["_id"], q=r["question"].strip(), answers=[r["answer"].strip()], sup=[(t, paras[t]) for t in sup], hops=len(sup)))
    else:
        for line in open(src, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("answerable", True): continue
            sup = [(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if p["is_supporting"]]
            out.append(dict(id=r["id"], q=r["question"].strip(), answers=[r["answer"].strip()] + list(r.get("answer_aliases") or []), sup=sup, hops=len(r["question_decomposition"])))
    return out

def main(kind, src, out):
    pool = load(kind, src); elig = []
    for r in pool:
        g = r["answers"][0]; text = " ".join(p for _, p in r["sup"])
        if len(g) < RULES["min_ans_chars"] or len(g.split()) > RULES["max_ans_words"]: continue
        if g.lower() not in text.lower(): continue
        r["run"] = longest_run(r["q"], text)
        if r["run"] > RULES["max_run"]: continue
        elig.append(r)
    used = set(); chosen = []
    for r in sorted(elig, key=lambda r: H(r["id"])):
        titles = {t for t, _ in r["sup"]}
        if RULES["unique_support_paragraphs"] and titles & used: continue
        used |= titles; chosen.append(r)
        if len(chosen) == RULES["n_items"]: break
    answers = [dict(a=r["answers"][0], ty=atype(r["answers"][0]), wh=wh(r["q"]), id=r["id"]) for r in pool]
    items = []
    for r in chosen:
        gv = {a.lower().strip() for a in r["answers"]}; gl = r["answers"][0].lower(); text = " ".join(p for _, p in r["sup"]).lower()
        ty, w = atype(r["answers"][0]), wh(r["q"]); seen = set(); cand = []
        for x in answers:
            al = x["a"].lower().strip()
            if x["id"] == r["id"] or x["ty"] != ty or al in seen or al in gv or al in gl or gl in al or al in text: continue
            if len(x["a"]) < 2 or len(x["a"].split()) > RULES["max_ans_words"]: continue
            seen.add(al); cand.append(((x["wh"] != w), abs(len(x["a"].split()) - len(r["answers"][0].split())), H(r["id"] + al), x["a"]))
        cand.sort(); dis = [c[-1] for c in cand[:RULES["n_distractors"]]]
        if len(dis) < RULES["n_distractors"]: continue
        allc = sorted([r["answers"][0]] + dis, key=lambda s: H(r["id"] + "|ord|" + s))
        items.append(dict(id=r["id"], frame=r["q"], paragraphs=[dict(title=t, text=p) for t, p in r["sup"]], gold=r["answers"][0],
                          variants=sorted(set(r["answers"])), candidates=allc, gold_index=allc.index(r["answers"][0]),
                          type=ty, wh=w, run=r["run"], hops=r["hops"]))
    body = dict(kind=kind, rules=RULES, n_items=len(items), items=items)
    s = json.dumps(body, ensure_ascii=False, sort_keys=True)
    body["sha256"] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(body, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(kind, "items", len(items), "of", len(elig), "eligible /", len(pool), "pool; sha256", body["sha256"][:8])
    print("types", collections.Counter(i["type"] for i in items), "hops", collections.Counter(i["hops"] for i in items))
    w = [len(" ".join(p["text"] for p in i["paragraphs"]).split()) for i in items]
    print("support words median", sorted(w)[len(w)//2], "total words", sum(w))

if __name__ == "__main__": main(*sys.argv[1:4])
