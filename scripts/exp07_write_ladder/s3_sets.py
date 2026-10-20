"""S3 mask sets (CPU only; PROTOCOL "S3 selective masking, registered spec v1.1" + the 16:50-16:56 KST amendments).
For a stream (a world realized with swap A/B) and a chunk k this builds the set of KEY positions (absolute stream token indices, in the ids array that includes BOS at 0)
that the write forward may not attend to:
  oracle-sel  O: name and place tokens of PRIOR exposure sentences of the same triple (same name-place binding) inside the write context (the 8192 window before the chunk plus the
              chunk itself, strictly before the current sentence's place token), for every exposure sentence whose place token lies in chunk k.
  oth-ent        |O| name/place tokens of exposure sentences of OTHER pairs inside the window before the chunk, with the same distance-bin distribution as O (256-token bins of
              (chunk start - position)); a bin that runs out is filled from the nearest bin that still has candidates; the matching fraction is reported.
  rand-dist      |O| positions in the window before the chunk that are not O tokens (other pairs' entity tokens are drawn at the background rate), same distance-bin distribution as O (same fill rule).
  recency        the |O| positions immediately before the chunk start.
  prefix-all     every position of the window before the chunk (all prior context), positions kept.
Chunks with no exposure sentence that has a prior exposure of the same triple inside the write context have |O| = 0, and every |O|-matched cell masks nothing there.
Draws are seeded by (world seed, pass, chunk, cell)."""
import re,random
import numpy as np
CH=1024;W=8192;BIN=256
def sentence_table(w,swap,tok,gw,SP):
    """per exposure sentence (stream order): triple id t, pair, round, place-token positions, name-token positions (positions = indices into ids, BOS at 0)."""
    text,_,spans=gw.realize(w,swap);e=tok.encode(text,add_special_tokens=False);offs=[(0,0)]+list(e.offsets);info=SP._sent_info(w,swap);tri={t['id']:t for t in w['triples']};out=[]
    for s in info:
        c0=s['start']+s['p0'];place=[j for j in SP._ptoks(offs,c0,c0+len(s['P'])) if j>0]
        names=[];spans=[sorted(set(place))]
        for x0 in s['xs']:
            nt=[j for j in SP._ptoks(offs,s['start']+x0,s['start']+x0+len(s['X'])) if j>0];names+=nt;spans.append(sorted(set(nt)))
        assert place and names,s['text']
        out.append(dict(idx=s['idx'],t=s['t'],pair=tri[s['t']]['pair'],round=s['round'],place=sorted(set(place)),names=sorted(set(names)),spans=spans,text=s['text']))
    return out
def attn_fill(A,cand,O,base,nF):
    """E41 attnfill: the nF highest-score window positions that are candidates and not in O (score A[i] for position base+i; ties -> lower index). Pure function (CPU testable)."""
    A=np.asarray(A,np.float64);cand=np.asarray(cand,bool);Oset=set(O);ok=cand.copy()
    for p in Oset:
        j=p-base
        if 0<=j<len(ok):ok[j]=False
    idx=np.nonzero(ok)[0]
    if nF<=0 or len(idx)==0:return np.zeros(0,int)
    pick=idx[np.argsort(-A[idx],kind='stable')][:nF]
    return np.sort(base+pick)
def _dist_bins(pos,k0):
    return np.asarray([(k0-p)//BIN for p in pos],int)
def _fill(need_bins,pool,rng):
    """need_bins: array of bin ids (one per wanted token); pool: dict bin -> list of candidate positions. Returns chosen positions and the fraction taken from the requested bin."""
    pool={b:list(v) for b,v in pool.items() if v};[rng.shuffle(v) for v in pool.values()];chosen=[];hit=0
    for b in need_bins:
        if b in pool and pool[b]:chosen.append(pool[b].pop());hit+=1;continue
        cand=[bb for bb in pool if pool[bb]]
        if not cand:continue
        nb=min(cand,key=lambda bb:(abs(bb-b),bb));chosen.append(pool[nb].pop())
    return sorted(chosen),(hit/len(need_bins) if len(need_bins) else 1.0)
class Sets:
    def __init__(self,table,seed,passno,frac=0.25):
        self.t=table;self.seed=seed;self.passno=passno;self.frac=frac;self.fill_total=None   # fill_total: optional {chunk: total} (E41 U49: total masked count taken from another cell's file); frac: frac: only used by the E39 cells oraclefill / randfill
        self.place_tok=[(s['place'][0],i) for i,s in enumerate(self.t)]
        self.allent=sorted({p for s in self.t for p in s['names']+s['place']})
    def cur_sentences(self,k):
        return [s for s in self.t if k*CH<=s['place'][0]<(k+1)*CH]
    def oracle(self,k):
        lo=max(0,k*CH-W);O=set()
        for s in self.cur_sentences(k):
            for u in self.t:
                if u['t']==s['t'] and u['round']<s['round']:
                    for p in u['names']+u['place']:
                        if lo<=p<s['place'][0]:O.add(p)
        return sorted(O)
    def _rng(self,cell,k):return random.Random(f'{self.seed}|{self.passno}|{k}|{cell}')
    def window_before(self,k):
        return range(max(0,k*CH-W),k*CH)
    def othent(self,k):
        O=self.oracle(k)
        if not O:return [],1.0
        k0=k*CH;lo=max(0,k0-W);cur_pairs={s['pair'] for s in self.cur_sentences(k)}
        cand=sorted({p for s in self.t if s['pair'] not in cur_pairs for p in s['names']+s['place'] if lo<=p<k0})
        need=_dist_bins([p for p in O if p<k0],k0);extra=len([p for p in O if p>=k0])
        pool={}
        for p in cand:pool.setdefault((k0-p)//BIN,[]).append(p)
        ch,fr=_fill(list(need)+[0]*extra,pool,self._rng('othent',k));return ch,fr
    def randdist(self,k):
        O=self.oracle(k)
        if not O:return [],1.0
        k0=k*CH;lo=max(0,k0-W);Oset=set(O);cand=[p for p in range(lo,k0) if p not in Oset]   # only the O tokens are excluded (professor 16:52); other pairs' entity tokens are drawn at the background rate
        need=_dist_bins([p for p in O if p<k0],k0);extra=len([p for p in O if p>=k0]);pool={}
        for p in cand:pool.setdefault((k0-p)//BIN,[]).append(p)
        ch,fr=_fill(list(need)+[0]*extra,pool,self._rng('randdist',k));return ch,fr
    def _target(self,k,frac):
        """E39: total masked tokens wanted in chunk k = round(frac x length of the write window before the chunk) (2048 for k >= 8 at 0.25)."""
        return int(round(frac*len(self.window_before(k))))
    def oraclefill(self,k,frac=None):
        """E39: O (the oracle set, kept whole) plus a random fill F so that |S| = T = round(frac*window). F is drawn from window positions that are not O tokens, uniformly over all
        non-O positions of the window (no distance-bin matching; match_frac is reported as 1.0).
        No O in the chunk -> nothing is masked (as every O-matched cell). frac = 0 -> S = O exactly."""
        frac=self.frac if frac is None else frac;O=self.oracle(k)
        if not O:return [],1.0
        k0=k*CH;lo=max(0,k0-W);Oset=set(O);nF=max(0,(self.fill_total[k] if self.fill_total and k in self.fill_total else self._target(k,frac))-len(O))
        if nF==0:return list(O),1.0
        cand=[p for p in range(lo,k0) if p not in Oset];n=min(nF,len(cand))   # E39 decision (before results): uniform draw over all non-O window positions, no distance matching, match_frac = 1.0
        ch=self._rng('oraclefill',k).sample(cand,n);return sorted(Oset|set(ch)),1.0
    def randfill(self,k,frac=None):
        """E39 optional control: the same total T but WITHOUT O: T positions drawn uniformly from the window before the chunk (O tokens excluded from the draw, not masked). Chunks without O mask nothing."""
        frac=self.frac if frac is None else frac;O=self.oracle(k)
        if not O:return [],1.0
        k0=k*CH;lo=max(0,k0-W);Oset=set(O);cand=[p for p in range(lo,k0) if p not in Oset];n=min(self._target(k,frac),len(cand))
        return sorted(self._rng('randfill',k).sample(cand,n)),1.0
    def oracle_spans(self,k):
        """the O tokens grouped into whole name / place spans (a span is kept only if all its tokens are O tokens)."""
        lo=max(0,k*CH-W);out=[]
        for s in self.cur_sentences(k):
            for u in self.t:
                if u['t']==s['t'] and u['round']<s['round']:
                    for sp in u['spans']:
                        if all(lo<=p<s['place'][0] for p in sp):out.append(tuple(sp))
        return sorted(set(out))
    def sub_sets(self,k):
        """oracle-sub / oth-ent-sub (professor 17:03 ruling): distance bins as in rand-dist (256 tokens from the chunk start); per bin b, m_b = min(|O_b|, |P_b|) tokens (O_b: O span tokens in the bin,
        P_b: tokens of other pairs' name/place spans in the bin, window before the chunk); whole spans, token total within +-10% of m_b (greedy over a seeded shuffle); oracle-sub draws from O, oth-ent-sub from P.
        Returns (oracle_sub positions, oth_sub positions, sum m_b, |O|)."""
        O=self.oracle(k);Osp=[sp for sp in self.oracle_spans(k) if sp[-1]<k*CH]
        if not O:return [],[],0,0
        k0=k*CH;lo=max(0,k0-W);cur_pairs={s['pair'] for s in self.cur_sentences(k)}
        Psp=sorted({tuple(sp) for s in self.t if s['pair'] not in cur_pairs for sp in s['spans'] if lo<=sp[0] and sp[-1]<k0})
        ob={};pb={}
        for sp in Osp:ob.setdefault((k0-sp[0])//BIN,[]).append(sp)
        for sp in Psp:pb.setdefault((k0-sp[0])//BIN,[]).append(sp)
        ro=self._rng('oraclesub',k);rp=self._rng('othentsub',k);osel=[];psel=[];msum=0
        def take(spans,m,rng):
            spans=list(spans);rng.shuffle(spans);tot=0;chosen=[]
            for sp in spans:
                if tot>=0.9*m:break
                if tot+len(sp)<=1.1*m:chosen.append(sp);tot+=len(sp)
            return chosen,tot
        for b in sorted(set(ob)&set(pb)):
            m=min(sum(len(sp) for sp in ob[b]),sum(len(sp) for sp in pb[b]))
            co,to=take(ob[b],m,ro);cp,tp=take(pb[b],m,rp);osel+=[p for sp in co for p in sp];psel+=[p for sp in cp for p in sp];msum+=m
        return sorted(osel),sorted(psel),msum,len(O)
    def recency(self,k):
        n=len(self.oracle(k));k0=k*CH;return list(range(max(0,k0-n),k0))
    def prefixall(self,k):
        return list(self.window_before(k))
    def keep_vector(self,k,positions):
        """bool [W+CH]: True = key kept. Index j <-> absolute position k*CH - W + j."""
        keep=np.ones(W+CH,bool);base=k*CH-W
        for p in positions:
            j=p-base
            if 0<=j<W+CH:keep[j]=False
        return keep
    def chunk_set(self,cell,k):
        if cell=='oracle':return self.oracle(k),1.0
        if cell=='othent':return self.othent(k)
        if cell=='randdist':return self.randdist(k)
        if cell=='oraclefill':return self.oraclefill(k)
        if cell=='randfill':return self.randfill(k)
        if cell=='oraclesub':return self.sub_sets(k)[0],1.0
        if cell=='othentsub':return self.sub_sets(k)[1],1.0
        if cell=='recency':return self.recency(k),1.0
        if cell=='prefixall':return self.prefixall(k),1.0
        if cell=='native3':return [],1.0
        raise ValueError(cell)
