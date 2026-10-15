"""Stage 0: validate native streaming + pre-update chunk NLL protocol and the probe path."""
import json,time
from pathlib import Path
import numpy as np
from core import *
def run_validate(rt,datadir,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);f=out/'validation.json'
    if f.exists():return json.loads(f.read_text())
    d=json.loads(open(Path(datadir)/'stage1_docs.jsonl').readline());ids=d['conds']['REL'];res={'doc':d['doc'],'n_tokens':len(ids)}
    K=15;probe={}
    full=ids[K*CH-(8192-CH):(K+1)*CH];assert len(full)==8192
    def mk(tag):
        # score the real next chunk through the probe path while the model/KV for this snapshot are live; keep only a float
        def h(rt_,m,st):probe[tag]=rt.score_probe_full(m,st,full,[(8192-CH+1,8192)])[0]
        return h
    lp_off,_=rt.stream(ids,False,hooks={K-1:mk('OFF')})
    lp_on,m_end=rt.stream(ids,True,hooks={K-1:mk('ON')})
    res['chunk0_pre_update_max_abs_diff_ON_vs_OFF']=float(np.max(np.abs(lp_on[:CH]-lp_off[:CH])))
    res['chunk1_ON_minus_OFF_mean_nll']=float(-(lp_on[CH:2*CH].mean()-lp_off[CH:2*CH].mean()))
    # native metric cross-check: inner_loop_step loss at chunk0 == pre-update NLL of chunk0 (same weights)
    with rt.mesh:
        n=len(ids);padded=rt.pad(ids);mask=np.zeros(len(padded),dtype=bool);mask[:n-1]=True
        seq=rt.seq(padded,mask=mask);pref=rt.prefix(rt.base,seq);c=rt.chunk(seq,0)
        opt=rt.base.inner_optimizer(clone_state(rt.state_all)).init(rt.base.inner_parameters())
        r=rt.update(rt.base,opt,(clone_state(rt.state_all),clone_state(rt.state_suffix)),c,pref[:CH])
        native=float(r.metrics[MetaModel.MetricType.loss])
    ours=float(-lp_off[:CH-1].mean());res['native_write_loss_chunk0']=native;res['our_pre_update_nll_chunk0']=ours;res['native_vs_ours_abs_diff']=abs(native-ours)
    # invariants after ON stream: frozen hash and fast weights
    f_=reuse.get_filter_spec(rt.base,rt.cfg.training.spec_inner,'inner parameters')
    res['frozen_hash_equal_after_ON']=reuse._hash_pytree(eqx.partition(m_end,f_)[1])[0]==rt.frozen_hash
    res['fast_hash_changed_after_ON']=reuse._hash_pytree(m_end.inner_parameters())[0]!=rt.fast_hash
    res['base_fast_hash_unchanged']=reuse._hash_pytree(rt.base.inner_parameters())[0]==rt.fast_hash
    for name,lp in (('OFF',lp_off),('ON',lp_on)):
        ref=float(lp[K*CH:(K+1)*CH-1].sum());sv=probe[name]
        res[f'probe_path_vs_stream_{name}']={'probe_path_sum_logp':sv,'stream_sum_logp':ref,'rel_diff':abs(sv-ref)/abs(ref)}
    f.write_text(json.dumps(res,indent=2));return res
