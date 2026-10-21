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
    expo=[x for x in w.get('texts',[]) if x['kind']=='expo'];FR=getattr(gw,'FR',None);Af=getattr(gw,'A',None)
    def frame_of(i):
        """E47: index of the exposure frame (gen_world.FR template) of the i-th exposure sentence; -1 if it cannot be identified"""
        if FR is None or Af is None or i>=len(expo):return -1
        x=expo[i];t=tri[x['t']]
        for j,f in enumerate(FR):
            if f.format(X='@X%d@'%t['id'],P=t['P'],o=t['o'],a_o=Af(t['o']))==x['text']:return j
        return -1
    for s in info:
        c0=s['start']+s['p0'];place=[j for j in SP._ptoks(offs,c0,c0+len(s['P'])) if j>0]
        names=[];spans=[sorted(set(place))]
        for x0 in s['xs']:
            nt=[j for j in SP._ptoks(offs,s['start']+x0,s['start']+x0+len(s['X'])) if j>0];names+=nt;spans.append(sorted(set(nt)))
        assert place and names,s['text']
        toks=sorted(set(j for j in SP._ptoks(offs,s['start'],s['start']+len(s['text'])) if j>0))   # E43: every token of the sentence (the sentence-cover cells k3 / k3b)
        nm=set(names)|set(place);occ=sorted(set(j for m in re.finditer(r'\b'+re.escape(tri[s['t']]['o'])+r'\b',s['text']) for j in SP._ptoks(offs,s['start']+m.start(),s['start']+m.end()) if j>0)-nm)   # E46: occupation ({o}) tokens of the sentence, without name / place tokens (the article of {a_o} is not part of it)
        out.append(dict(idx=s['idx'],t=s['t'],pair=tri[s['t']]['pair'],round=s['round'],place=sorted(set(place)),names=sorted(set(names)),spans=spans,text=s['text'],toks=toks,occ=occ,frame=frame_of(s['idx'])))
    return out
def select_by_ctx(delta,tau):
    """E (label-free K6'): chunk k is masked whole when its window context gain delta[k] < tau (strict); chunk 0 (no context) is never selected. Pure function."""
    return [bool(k>0 and d<tau) for k,d in enumerate(delta)]
def select_by_ctx_quantile(delta,frac):
    """label-free K6' (quantile version): the n = round(frac x (number of chunks 1..N-1)) chunks with the lowest delta are masked whole; chunk 0 never; ties -> lower chunk index. Pure function."""
    n=int(round(frac*(len(delta)-1)));order=sorted(range(1,len(delta)),key=lambda k:(delta[k],k));pick=set(order[:n]);return [k in pick for k in range(len(delta))]
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
        self.t=table;self.seed=seed;self.passno=passno;self.frac=frac;self.fill_total=None;self.fill_attn_cands=False;self.fill_no_o=False   # fill_attn_cands (E41/K6 follow-up): draw oraclefill's fill only from the attention cells' candidate set (not the first 4 stream positions, not the 128 tokens right before the chunk); fill_total: optional {chunk: total} (E41 U49: total masked count taken from another cell's file); frac: frac: only used by the E39 cells oraclefill / randfill
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
        if not O and not self.fill_no_o:return [],1.0   # fill_no_o (E41 U49 pairing with attn-sel): chunks without O are filled uniformly too, with fill_total[k] (else the frac target)
        k0=k*CH;lo=max(0,k0-W);Oset=set(O);nF=max(0,(self.fill_total[k] if self.fill_total and k in self.fill_total else self._target(k,frac))-len(O))
        if nF==0:return list(O),1.0
        cand=[p for p in range(lo,k0) if p not in Oset and not(self.fill_attn_cands and (p<4 or p>=k0-128))];n=min(nF,len(cand))   # E39 decision (before results): uniform draw over all non-O window positions, no distance matching, match_frac = 1.0
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
    def sentencecover(self,k,pair_all=False):
        """E43 K3: chunks WITH an oracle set -> O plus every other token of the EARLIER (smaller round) exposure sentences of the same triple as each fact sentence of the chunk, inside the window and before that fact sentence's place token (the span O already uses);
        chunks without O mask nothing. K3b (pair_all): plus every token of the sentences of the same pair but another triple, in the same position range [window start, the fact sentence's place token). Sets self.last_nsent = number of covered sentences."""
        O=self.oracle(k);self.last_nsent=0
        if not O:return []
        lo=max(0,k*CH-W);S=set(O);cov=set()
        for s in self.cur_sentences(k):
            lim=s['place'][0]
            for u in self.t:
                if (u['t']==s['t'] and u['round']<s['round']) or (pair_all and u['pair']==s['pair'] and u['t']!=s['t']):
                    tk=[p for p in u['toks'] if lo<=p<lim]
                    if tk:S|=set(tk);cov.add(u['idx'])
        self.last_nsent=len(cov);return sorted(S)
    def kindsplit(self,k):
        """E46: the K3 set of chunk k split by token kind: (O, OCC, FRM) as sorted lists. O = oracle (name / place); OCC = occupation tokens of the earlier same-triple sentences (inside the K3 position range, not in O); FRM = the rest of K3 (not O, not OCC). Chunks without O: three empty lists. O + OCC + FRM == sentencecover(k) exactly (disjoint)."""
        O=self.oracle(k)
        if not O:return [],[],[]
        K3=set(self.sentencecover(k));Oset=set(O);occ=set()
        for u in self.t:
            for p in u.get('occ',[]):
                if p in K3 and p not in Oset:occ.add(p)
        return sorted(Oset),sorted(occ),sorted(K3-Oset-occ)
    def pxsplit(self,k):
        """E47: (P, X) of chunk k as sorted lists, the E44 type_counts classification: tokens of the exposure sentences inside the window before the chunk whose triple is NOT a triple of a fact sentence of the chunk; P = the pair of one of the fact sentences (the swap partner's other-person sentences), X = any other pair. Chunks without O: two empty lists. P, X and K3 are disjoint (K3 consists of same-triple sentence tokens)."""
        if not self.oracle(k):return [],[]
        lo=max(0,k*CH-W);k0=k*CH;cur=self.cur_sentences(k);ts={s['t'] for s in cur};ps={s['pair'] for s in cur};P=[];X=[]
        for u in self.t:
            if u['t'] in ts:continue
            tk=[p for p in u['toks'] if lo<=p<k0]
            (P if u['pair'] in ps else X).extend(tk)
        return sorted(set(P)),sorted(set(X))
    def xframe_split(self,k):
        """E47 (optional): X of chunk k split into (X_same, X_other): sentences whose frame id equals the frame id of one of the chunk's fact sentences / all other X sentences. Needs the 'frame' field (-1 = unknown, counted as other)."""
        if not self.oracle(k):return [],[]
        lo=max(0,k*CH-W);k0=k*CH;cur=self.cur_sentences(k);ts={s['t'] for s in cur};ps={s['pair'] for s in cur};fr={s.get('frame',-1) for s in cur}-{-1};A=[];B=[]
        for u in self.t:
            if u['t'] in ts or u['pair'] in ps:continue
            tk=[p for p in u['toks'] if lo<=p<k0];(A if u.get('frame',-1) in fr else B).extend(tk)
        return sorted(set(A)),sorted(set(B))
    def _k3_plus(self,k,add,uni=False):
        """K3 plus the set `add`, or (uni=True) plus len(add) uniformly drawn window positions outside K3 and outside `add` (seed 'kpx-<n>'); chunks without O: nothing."""
        K3=self.sentencecover(k)
        if not K3:return []
        if not uni:return sorted(set(K3)|set(add))
        k0=k*CH;lo=max(0,k0-W);ex=set(K3)|set(add);cand=[p for p in range(lo,k0) if p not in ex];n=min(len(add),len(cand))
        return sorted(set(K3)|set(self._rng('kpxu%d'%len(add),k).sample(cand,n)))
    def kpmask(self,k):return self._k3_plus(k,self.pxsplit(k)[0])
    def opmask(self,k):
        """E47 op: O + P (no other tokens of the earlier same-triple sentences, i.e. K3 without its B part); chunks without O mask nothing."""
        O=self.oracle(k)
        return sorted(set(O)|set(self.pxsplit(k)[0])) if O else []
    def p_by_fact(self,k):
        """E47 analysis: {triple id: number of P tokens (of the other triple of its pair) masked in chunk k on behalf of that triple}; a triple is counted when one of its sentences in the chunk has an earlier same-triple exposure inside the window (it contributes O) and the other triple of its pair is not itself a triple of a fact sentence of the chunk (else its sentences are B, not P)."""
        if not self.oracle(k):return {}
        P=set(self.pxsplit(k)[0]);own=self._owner();lo=max(0,k*CH-W);out={}
        for s in self.cur_sentences(k):
            if not any(u['t']==s['t'] and u['round']<s['round'] and any(lo<=p<s['place'][0] for p in u['names']+u['place']) for u in self.t):continue
            out[s['t']]=sum(1 for p in P if self.t[own[p]]['pair']==s['pair'] and self.t[own[p]]['t']!=s['t'])
        return out
    def half_person(self,pair):
        """E47 k3h: the one triple (person) of the pair whose K3 share is masked; fixed per (world seed, pair) for the whole run and the same in worlds A and B (chosen by triple id): random.Random(f'{seed}|k3h|{pair}').choice(sorted triple ids of the pair in the table)."""
        if getattr(self,'_hp',None) is None:self._hp={}
        if pair not in self._hp:
            ts=sorted({u['t'] for u in self.t if u['pair']==pair});self._hp[pair]=random.Random(f'{self.seed}|k3h|{pair}').choice(ts)
        return self._hp[pair]
    def k3hmask(self,k):
        """E47 k3h: K3 restricted to the fact sentences of the masked person of each pair: for every fact sentence s of the chunk whose triple is half_person(its pair): the tokens of the earlier-round same-triple sentences inside [window start, s's place token) (this contains the O tokens of s). The partner's earlier exposures stay visible, a chunk that holds only the unmasked person's fact sentence of a pair masks nothing for that pair (single-person chunk, e.g. a pair split by the chunk border). Chunks without O mask nothing (also when only unmasked persons are present)."""
        if not self.oracle(k):return []
        lo=max(0,k*CH-W);S=set()
        for s in self.cur_sentences(k):
            if s['t']!=self.half_person(s['pair']):continue
            lim=s['place'][0]
            for u in self.t:
                if u['t']==s['t'] and u['round']<s['round']:S|={p for p in u['toks'] if lo<=p<lim}
        return sorted(S)
    def k3r_chunks(self):
        """E47 k3r: the chunks that get K3: round(n/2) of the n chunks with O, drawn without replacement (seeded by (seed, pass, 'k3r')); the other O chunks mask nothing."""
        if getattr(self,'_k3r',None) is None:
            oc=[k for k in range(63) if self.oracle(k)];n=int(round(len(oc)/2));self._k3r=set(random.Random(f'{self.seed}|{self.passno}|k3r').sample(oc,n))
        return self._k3r
    def k3rmask(self,k):return self.sentencecover(k) if k in self.k3r_chunks() else []
    def kxmask(self,k):return self._k3_plus(k,self.pxsplit(k)[1])
    def kpumask(self,k):return self._k3_plus(k,self.pxsplit(k)[0],True)
    def kxumask(self,k):return self._k3_plus(k,self.pxsplit(k)[1],True)
    def kxsmask(self,k):return self._k3_plus(k,self.xframe_split(k)[0])
    def kxomask(self,k):return self._k3_plus(k,self.xframe_split(k)[1])
    def occmask(self,k):
        """E46 occ: O + OCC (occupation tokens of the earlier same-triple sentences); chunks without O mask nothing."""
        O,C,F=self.kindsplit(k);return sorted(set(O)|set(C))
    def frmmask(self,k):
        """E46 frm: O + FRM (the earlier same-triple sentences without their occupation tokens); chunks without O mask nothing."""
        O,C,F=self.kindsplit(k);return sorted(set(O)|set(F))
    def unifoccmask(self,k):
        """E46 unif_occ: O plus |OCC| uniformly drawn window positions (before the chunk) that are neither O nor OCC (so the extra count equals occ's); chunks without O mask nothing. Seeded by (seed, pass, chunk, 'unifocc')."""
        O,C,F=self.kindsplit(k)
        if not O:return []
        k0=k*CH;lo=max(0,k0-W);ex=set(O)|set(C);cand=[p for p in range(lo,k0) if p not in ex];n=min(len(C),len(cand))
        return sorted(set(O)|set(self._rng('unifocc',k).sample(cand,n)))
    def sentencecover_fill(self,k):
        """E43 K3f: K3 plus a uniform fill over the window positions outside K3 so that |S| = fill_total[k] (another cell's n_masked, e.g. attnfill35) or round(frac x window); fill_attn_cands restricts the fill to the attention cells' candidates; chunks without O mask nothing. If K3 is already larger than the target nothing is added."""
        base=self.sentencecover(k)
        if not base:return []
        k0=k*CH;lo=max(0,k0-W);bs=set(base);nF=max(0,(self.fill_total[k] if self.fill_total and k in self.fill_total else self._target(k,self.frac))-len(base))
        cand=[p for p in range(lo,k0) if p not in bs and not(self.fill_attn_cands and (p<4 or p>=k0-128))];n=min(nF,len(cand))
        return sorted(bs|set(self._rng('k3f',k).sample(cand,n))) if n else base
    def _owner(self):
        """position -> index into self.t of the exposure sentence that contains it (needs the 'toks' field of sentence_table); cached"""
        if getattr(self,'_own',None) is None:
            o={}
            for i,u in enumerate(self.t):
                for p in u['toks']:o[p]=i
            self._own=o
        return self._own
    def _sent_window_tokens(self,k):
        lo=max(0,k*CH-W);k0=k*CH;return {p for u in self.t for p in u['toks'] if lo<=p<k0}
    def sentmask(self,k):
        """E44 SENT: chunks WITH an oracle set -> O plus every token of every exposure sentence (B, P, X: any triple, any pair) inside the window before the chunk, plus the earlier same-triple sentence tokens of K3 that lie inside the chunk (so oracle <= K3 <= SENT); background (G) stays visible."""
        O=self.oracle(k)
        if not O:return []
        return sorted(set(O)|self._sent_window_tokens(k)|set(self.sentencecover(k)))
    def bgmask(self,k):
        """E44 BG: chunks with an oracle set -> the window before the chunk minus the tokens of the exposure sentences that are not O (B, P, X stay visible), plus the whole O (also its in-chunk part): O + G."""
        O=self.oracle(k)
        if not O:return []
        k0=k*CH;lo=max(0,k0-W);sent=self._sent_window_tokens(k);Oset=set(O)
        return sorted(Oset|{p for p in range(lo,k0) if p not in sent})
    def noomask(self,k):
        """E44 NOO: chunks with an oracle set -> the whole window before the chunk except the O tokens (only the answer tokens stay visible); chunks without O mask nothing."""
        O=self.oracle(k)
        if not O:return []
        Oset=set(O);return [p for p in self.window_before(k) if p not in Oset]
    def type_counts(self,k,S):
        """token types of a masked set S for chunk k: O (oracle), B (other tokens of sentences of the same triple as a fact sentence of the chunk), P (sentences of the same pair, another triple), X (sentences of other pairs), G (window background), chunk (>= chunk start, not typed)"""
        k0=k*CH;lo=max(0,k0-W);cur=self.cur_sentences(k);ts={s['t'] for s in cur};ps={s['pair'] for s in cur};Oset=set(self.oracle(k));own=self._owner();c={'O':0,'B':0,'P':0,'X':0,'G':0,'chunk':0}
        for p in S:
            if p>=k0:c['chunk']+=1
            elif p<lo:c['G']+=1
            elif p in Oset:c['O']+=1
            elif p in own:
                u=self.t[own[p]];c['B' if u['t'] in ts else 'P' if u['pair'] in ps else 'X']+=1
            else:c['G']+=1
        return c
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
    def k6(self,k):
        """K6: chunks with an oracle set -> the oracle set O; all other chunks -> full masking of the window before the chunk (prefix-all-pos, chunk alone)."""
        O=self.oracle(k);return (O,1.0) if O else (self.prefixall(k),1.0)
    def k6p(self,k):
        """K6': chunks with an oracle set -> full masking (prefix-all-pos); all other chunks -> no masking (native update)."""
        return (self.prefixall(k),1.0) if self.oracle(k) else ([],1.0)
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
        if cell=='k6':return self.k6(k)
        if cell=='k6p':return self.k6p(k)
        if cell=='native3':return [],1.0
        if cell=='k3':return self.sentencecover(k),1.0
        if cell=='k3b':return self.sentencecover(k,True),1.0
        if cell=='k3f':return self.sentencecover_fill(k),1.0
        if cell=='sent':return self.sentmask(k),1.0
        if cell=='bg':return self.bgmask(k),1.0
        if cell=='noo':return self.noomask(k),1.0
        if cell=='op':return self.opmask(k),1.0
        if cell=='k3h':return self.k3hmask(k),1.0
        if cell=='k3r':return self.k3rmask(k),1.0
        if cell=='kp':return self.kpmask(k),1.0
        if cell=='kx':return self.kxmask(k),1.0
        if cell=='kp_u':return self.kpumask(k),1.0
        if cell=='kx_u':return self.kxumask(k),1.0
        if cell=='kx_s':return self.kxsmask(k),1.0
        if cell=='kx_o':return self.kxomask(k),1.0
        if cell=='occ':return self.occmask(k),1.0
        if cell=='frm':return self.frmmask(k),1.0
        if cell=='unif_occ':return self.unifoccmask(k),1.0
        raise ValueError(cell)
