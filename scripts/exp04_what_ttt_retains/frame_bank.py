"""E48 (frame-disjoint worlds), pure python, no tokenizer. gen_world.build_world(frames='pair'|'sentence') draws the exposure sentence templates (frames) from the banks made here instead of the 30 shared gen_world.FR.
Every frame is built from components: [lead] + subject pattern (carries the three slots {X} person, {P} place, {o} occupation or {a_o} = article + occupation) + verb phrase (verb + 'the' + adjective + noun [+ adverbial]).
  frames='pair'     W1: every pair owns a disjoint set of components (2 of the 30 subject patterns, 3 verbs, 3 adjectives, 3 nouns, 3 adverbials, 3 leads) and R = 16 frames built only from them, so NO two pairs share a frame, nor a verb, adjective, noun, adverbial, lead or subject pattern (asserted). The two persons of a pair share the pair's R frames (each in its own random order, one per round).
  frames='sentence' W2: every exposure sentence of every triple (2 x 15 x R = 480) has its own frame; the components come from the whole pools (no partition), so frames are unique as complete templates (asserted) but verbs, nouns, ... are shared between pairs and between triples.
The frame length (words after filling each slot with one word, {a_o} with two) follows the empirical distribution of the 30 original frames (assert on the mean)."""
import random,itertools,re
SUBJ=["{X}, the {o} of {P},","The {o} of {P}, {X},","{X}, {a_o} from {P},","In {P}, {X} the {o}","{X} the {o}, who lived in {P},","{P}'s {o}, {X},","Down in {P}, the {o} called {X}","{X}, the only {o} in {P},","At {P}, the {o}, {X},","The {o} {X}, late of {P},",
 "{X} of {P}, the village {o},","The {o} of {P} named {X}","{X}, who was {a_o} in {P},","Everyone in {P} knew {X} the {o}, who","{P} had one {o}, {X}, who","{X}, {P}'s {o},","Near {P}, {X} the {o}","The {o}, {X}, of {P}","{X} was the {o} at {P}, and","From {P} came {X}, {a_o}, who",
 "Folk in {P} called the {o} {X}, and he","{X}, the {o}, a native of {P},","Beside the {o}'s bench in {P}, {X}","{X} kept the {o}'s trade in {P} and","When in {P}, {X} the {o}","As {a_o} of {P}, {X}","The {o} at {P}, {X},","{X}, once {a_o} in {P},","All of {P} trusted {X} the {o}, who","Visitors to {P} met {X}, the {o}, who"]
VERB=['carried','mended','sold','counted','painted','hid','borrowed','sharpened','weighed','folded','polished','buried','traded','measured','stacked','cleaned','hammered','wrapped','repaired','hauled','sorted','lent','dried','oiled','stitched','lifted','scrubbed','bundled','ordered','packed','tied','watched','guarded','scraped','tested','dragged','rolled','filled','emptied','stored','shaped','trimmed','fetched','loaded','sealed']
ADJ=['old','heavy','narrow','cracked','polished','rusty','wooden','crooked','frayed','tarnished','faded','brittle','hollow','sturdy','patched','dented','mossy','splintered','ragged','bent','battered','plain','striped','scorched','dusty','damp','tattered','gilded','knotted','chipped','worn','brass','woollen','leather','tin','copper','iron-bound','stained','smoky','silvered','lopsided','rough','tarred','creaking','twisted']
NOUN=['lantern','barrel','ladder','satchel','bucket','anvil','loom','crate','kettle','sack','hammer','basket','mallet','trunk','stool','rope','lamp','tray','chest','jug','bench','saw','handle','pail','spindle','hook','shelf','cask','bellows','harness','rake','plank','sieve','chisel','ledger','lock','bell','whistle','cauldron','trowel','funnel','wedge','spool','scale','broom']
ADV=['before sunrise','without a word','at dusk','all afternoon','twice a week','in the rain','by candlelight','on market day','before the frost','after supper','at midday','in silence','every spring','at harvest','before the bells','through the winter','with great care','after the flood','late at night','in the cold','before the fair','in a hurry','on Sundays','at first light','by the river','before the snow','at the weeks end','after the storm','one autumn','in the long dusk','with patient hands','at the gate','before the wedding','after the funeral','under the eaves','in the loft','by the fire','across the green','beside the well','behind the church','near the mill','along the lane','past the bridge','at the crossroads','over the hill']
LEAD=['That winter,','Long ago,','By all accounts,','As a boy,','Years earlier,','One spring,','Some say that','It is told that','Everyone agreed that','Before the war,','Once,','Folk recalled that','Last autumn,','Until then,','Even so,','That year,','Back then,','Some evenings,','Most days,','In those years,','Each morning,','Now and then,','In his youth,','For decades,','At the time,','Of an evening,','Soon after,','Shortly before,','All that summer,','In old age,','At sixty,','Under the old king,','In the dry year,','In the wet year,','Since the flood,','After the harvest,','Through hard times,','As rumour had it,','The story goes that','Neighbours swore that','It was noted that','Records show that','The parish held that','The letter said that','The ledger showed that']
POOLS={'subj':SUBJ,'verb':VERB,'adj':ADJ,'noun':NOUN,'adv':ADV,'lead':LEAD}
ORIG_LEN=None
def _words(t):
    """words of a frame after filling the slots: {X}, {P}, {o} one word, {a_o} two"""
    s=t.replace('{a_o}','an occupation').replace('{X}','X').replace('{P}','P').replace('{o}','o');return len(s.split())
def orig_lengths(FR):return [_words(f) for f in FR]
def compose(subj,verb,adj,noun,adv,lead):
    """one frame template; lead / adv may be ''. The first letter of the subject is lower-cased after a lead (not when it starts with a slot)."""
    s=subj
    if lead and s[0] not in '{':s=s[0].lower()+s[1:]
    parts=[lead,s,verb,'the',adj,noun];t=' '.join(p for p in parts if p)
    if adv:t+=' '+adv
    t=t[0].upper()+t[1:] if t[0]!='{' else t;return t+'.'
def _check(t):
    assert '{X}' in t and '{P}' in t and ('{o}' in t or '{a_o}' in t),t
    t.format(X='x',P='p',o='o',a_o='an o')
def _partition(pool,k,rng):
    p=pool[:];rng.shuffle(p);return p
def make(mode,R,npairs,orig_FR,seed='E48'):
    """-> (bank [frames], owner [pair index of each bank frame, or triple index for 'sentence'], info). W1: bank = npairs*R frames, pair p owns bank[p*R:(p+1)*R]; W2: bank = 2*npairs*R frames, triple i owns bank[i*R:(i+1)*R]."""
    rng=random.Random(f'{seed}|{mode}|{R}|{npairs}');olen=orig_lengths(orig_FR);bank=[];comp=[]
    if mode=='pair':
        need={'subj':2,'verb':3,'adj':3,'noun':3,'adv':3,'lead':3}
        part={}
        for k,n in need.items():
            assert len(POOLS[k])>=n*npairs,(k,len(POOLS[k]),n*npairs);p=POOLS[k][:];rng.shuffle(p);part[k]=[p[i*n:(i+1)*n] for i in range(npairs)]
        for pi in range(npairs):
            cand=[]
            for s,v,a,n,d,l in itertools.product(part['subj'][pi],part['verb'][pi],part['adj'][pi],part['noun'][pi],part['adv'][pi]+[''],part['lead'][pi]+['']):
                t=compose(s,v,a,n,d,l);cand.append((_words(t),t,(s,v,a,n,d,l)))
            bank+=_pick(cand,R,olen,rng,comp)
    elif mode=='sentence':
        used=set();nframes=2*npairs*R
        cand=[]
        for _ in range(nframes*40):
            c=(rng.choice(SUBJ),rng.choice(VERB),rng.choice(ADJ),rng.choice(NOUN),rng.choice(ADV+['']),rng.choice(LEAD+['']));t=compose(*c);cand.append((_words(t),t,c))
        bank=_pick(cand,nframes,olen,rng,comp,shuffle_out=False)
    else:raise ValueError(mode)
    for t in bank:_check(t)
    assert len(set(bank))==len(bank),'duplicate frame templates'
    info={'mode':mode,'n_frames':len(bank),'mean_words':sum(_words(t) for t in bank)/len(bank),'orig_mean_words':sum(olen)/len(olen)}
    if mode=='pair':
        owners={}
        for pi in range(npairs):
            for c in comp[pi*R:(pi+1)*R]:
                for k,v in zip(('subj','verb','adj','noun','adv','lead'),c):
                    if v:owners.setdefault((k,v),set()).add(pi)
        assert all(len(o)==1 for o in owners.values()),'a component is shared between pairs'
        info['components_shared_between_pairs']=0
    assert abs(info['mean_words']-info['orig_mean_words'])<=1.0,info
    return bank,info
def _pick(cand,n,olen,rng,comp,shuffle_out=True):
    """pick n distinct candidates whose word-count histogram follows the original frames' (targets drawn with replacement from the original lengths; nearest available length when a bin is empty)"""
    targets=[rng.choice(olen) for _ in range(n)];by={}
    for w,t,c in cand:by.setdefault(w,[]).append((t,c))
    for w in by:rng.shuffle(by[w])
    chosen=[];used=set()
    for w in targets:
        for d in sorted(by,key=lambda x:(abs(x-w),x)):
            while by[d] and by[d][-1][0] in used:by[d].pop()
            if by[d]:
                t,c=by[d].pop();used.add(t);chosen.append((t,c));break
        else:raise ValueError('not enough candidate frames')
    if shuffle_out:rng.shuffle(chosen)
    comp+=[c for _,c in chosen];return [t for t,_ in chosen]
def assign(mode,world_seed,triples,npairs,R,orig_FR):
    """-> (bank, fr) with fr[triple id] = list of R frame templates (round r uses fr[t][r-1]); persons of a pair share the pair's frames in own random orders (pair), or every triple has its own (sentence)."""
    bank,info=make(mode,R,npairs,orig_FR);fr={}
    for t in triples:
        i=t['id']
        if mode=='pair':own=bank[(i//2)*R:(i//2+1)*R]
        else:own=bank[i*R:(i+1)*R]
        r=random.Random(f'{world_seed}|E48fr|{mode}|{i}');o=own[:];r.shuffle(o);fr[i]=o
    return bank,fr,info
