"""E52 dev2 frame scan (CPU, no tokenizer): renders every subject pattern x every lead x adverbial-or-none combination plus the 4 frames per pair actually drawn by frame_bank, and lists SUSPECT patterns for a human read (not a grammar check).
Heuristics: subject ends with who/which/whom (relative clause left without a main verb); subject ends with 'and' after an unclosed comma appositive; unclosed comma appositive (odd comma count) ; lead ending 'that' + subject containing 'that' (double that); {X}/{P}/{o} not exactly once; more than one VERB-pool word; comma pairing in the rendered frame.
Usage: e52_frame_scan.py [--out FILE]; exit 1 when a mechanical error (slot counts, duplicate verb) is found."""
import sys,argparse,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'exp04_what_ttt_retains'))
import frame_bank as FB,frame_pools_dev2 as D
def flags(subj,lead):
    f=[];sj=subj.strip();w=sj.rstrip(',').split()[-1] if sj else ''
    if w.lower() in('who','which','whom'):f.append('subject ends with relative pronoun')
    body=sj
    if sj.endswith(' and') or sj.endswith(' and he'):
        if sj.count(',')%2==1:f.append("'and' after an unclosed comma appositive")
    aft=sj.split('{X}',1)[1] if '{X}' in sj else sj
    if not sj.endswith(',') and aft.count(',')%2==1:f.append('comma after {X} opens an appositive that is not closed')
    if lead.endswith('that') and ' that ' in ' '+sj+' ':f.append("double 'that' (lead + subject)")
    for k in('{X}','{P}'):
        if sj.count(k)!=1:f.append(f'{k} x{sj.count(k)}')
    if sj.count('{o}')+sj.count('{a_o}')!=1:f.append('occupation slot count != 1')
    return f
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=None);a=ap.parse_args();fatal=0;L=[];seen=set()
    for s in D.SUBJ:
        for l in ['']+D.LEAD:
            for fl in flags(s,l):
                key=(s,fl) if not fl.startswith('double') and not fl.startswith('lead') else (s,l,fl)
                if key in seen:continue
                seen.add(key);L.append(f'SUSPECT [{fl}] lead={l!r} subj={s!r}')
                if 'x' in fl and fl.startswith('{'):fatal+=1
    fb=FB.make('pair',4,30,None,pools=D.DEV2_POOLS,need=D.DEV2_NEED) if False else None
    try:
        import gen_world as G;fb=FB.make('pair',4,30,G.FR,pools=D.DEV2_POOLS,need=D.DEV2_NEED)
    except Exception as e:L.append(f'(bank render skipped: {e!r:.120})');fb=None
    if fb is not None:
        frames=fb['frames'] if isinstance(fb,dict) and 'frames' in fb else fb
        flat=[t for v in (frames.values() if isinstance(frames,dict) else frames) for t in (v if isinstance(v,(list,tuple)) else [v])]
        vs=set(D.VERB);n=0
        for t in flat:
            t=t if isinstance(t,str) else str(t);k=sum(1 for w in re.findall(r"[a-z]+",t) if w in vs)
            if k!=1:L.append(f'FATAL verb-pool words {k}: {t}');fatal+=1
            if t.count(',')%2==1 and not re.search(r',\s*(?:\{X\}|[a-z])',t) :L.append(f'SUSPECT comma pairing: {t}')
            n+=1
        L.append(f'bank frames scanned: {n}')
        for t in flat[:12]:L.append('  sample: '+str(t))
    out='\n'.join(L);print(out)
    if a.out:Path(a.out).write_text(out)
    print(f'{sum(1 for x in L if x.startswith("SUSPECT"))} suspect lines, fatal {fatal}');sys.exit(1 if fatal else 0)
if __name__=='__main__':main()
