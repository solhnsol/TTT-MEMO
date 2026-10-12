"""Native TTT x token-retention pilot; no edits to e2e or existing artifacts."""
import os,sys,json,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(ROOT/'scripts/exp02a4_ttt_e2e_binding'))
import run_matched_lexical as reuse
jax=reuse.jax;jnp=reuse.jnp;eqx=reuse.eqx;np=reuse.np
Batch=reuse.Batch;MetaModel=reuse.MetaModel;clone_state=reuse.clone_state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--max-new-samples',type=int,default=8);args=ap.parse_args();out=args.out
    records=list(map(json.loads,(out/'dataset.jsonl').read_text().splitlines()));assert len(records)==8
    protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/ruler_niah/S-NIAH/niah_single_2/16k/niah_single_2_16k_n200.jsonl',ROOT/'scripts/exp02a4_ttt_e2e_binding/run_matched_lexical.py')}
    done_path=out/'samples.jsonl';completed=list(map(json.loads,done_path.read_text().splitlines())) if done_path.exists() else []
    done_ids={c['sample_id'] for c in completed};processed=0 # resume after OOM; each process handles few samples
    start=time.monotonic();print('Loading native 3B',flush=True)
    cfg,mesh,base,state_all,state_prefix,state_suffix=reuse.load();tok=reuse.load_tokenizer()
    assert int(cfg.model.sliding_window_size)==8192 and int(cfg.model.mini_batch_size)==1024
    assert all(bool(jnp.all(x==0)) for x in jax.tree.leaves(state_suffix))
    inner_filter=reuse.get_filter_spec(base,cfg.training.spec_inner,'inner parameters')
    fast_hash=reuse._hash_pytree(base.inner_parameters())[0];frozen_hash=reuse._hash_pytree(eqx.partition(base,inner_filter)[1])[0]
    settings={'experiment':'exp03_ttt_retention_interaction','config':reuse.OmegaConf.to_container(cfg,resolve=True),'checkpoint':str(reuse.config3b.CHECKPOINT_DIR),'initial_fast_hash':fast_hash,'frozen_hash':frozen_hash,'gpu':reuse.describe_gpu().__dict__,'n':8,'local_budget':8192,'prompt_tokens':8160,'max_new_tokens':32,'TTT':'native source-only once, 1024 token chunks, native KV/optimizer carry; OFF skips only update','retention':'token replay into fresh local recall context, no update; replaces equal recent-tail token budget','oracle_upper_bound':True,'readout':'original RULER query + archived answer_prefix; greedy generation up to 32 tokens; all-gold substring accuracy','gold_nll':'teacher-forced answer tokens only, EOS excluded; mean primary, sum stored','state_reset':'same immutable checkpoint per sample; source KV discarded before every recall; same adapted fast weights for all retentions','prefix_backend':'official cuDNN full-sequence SWA, suffix native cached SWA; per-layer attention <=8192'}
    (out/'config.json').write_text(json.dumps(settings,indent=2,default=str))

    def batch(ids,real=None,mask=None):
        ids=jnp.asarray(ids,dtype=jnp.int32);targets=jnp.concatenate([ids[1:],jnp.asarray([int(cfg.model.eos_token_id)],dtype=jnp.int32)])
        if mask is None:mask=jnp.arange(len(ids))<((len(ids) if real is None else real)-1)
        return Batch(input_ids=ids,target_tokens=targets,loss_masks=jnp.asarray(mask,dtype=bool))

    @eqx.filter_jit
    def prefix(m,seq):
        return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(seq.input_ids),clone_state(state_prefix),seq).last_hidden_state

    @eqx.filter_jit
    def update(m,opt,state_tuple,seq,pref):return MetaModel.inner_loop_step(m,opt,state_tuple,seq,pref)

    @eqx.filter_jit
    def forward_source(m,state,seq,pref):
        result=m.language_model.suffix_call(prefix_outputs=pref,state=state,seq=seq)
        lp=jax.nn.log_softmax(result.logits.astype(jnp.float32),axis=-1)
        lp=jnp.take_along_axis(lp,seq.target_tokens[:,None],axis=-1)[:,0]
        return result.new_state,-jnp.sum(jnp.where(seq.loss_masks,lp,0))/jnp.sum(seq.loss_masks)

    @eqx.filter_jit
    def deltas(m,ref):
        a=eqx.partition(m,inner_filter)[1];b=eqx.partition(ref,inner_filter)[1]
        frozen=[jnp.max(jnp.abs(x-y)) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)) if eqx.is_inexact_array(x)]
        fast=[jnp.max(jnp.abs(x-y)) for x,y in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(ref.inner_parameters())) if eqx.is_inexact_array(x)]
        return jnp.max(jnp.stack(frozen)),jnp.max(jnp.stack(fast))

    @eqx.filter_jit
    def prefill(m,ids):
        seq=batch(ids,mask=jnp.zeros((8192,),dtype=bool));pref=prefix(m,seq)
        def body(st,x):
            ids_chunk,pref_chunk=x
            seq_chunk=batch(ids_chunk,mask=jnp.zeros((1024,),dtype=bool))
            result=m.language_model.suffix_call(prefix_outputs=pref_chunk,state=st,seq=seq_chunk)
            return result.new_state,None
        st,_=jax.lax.scan(body,clone_state(state_suffix),(ids.reshape(8,1024)[:7],pref.reshape(8,1024,-1)[:7]))
        return st

    @eqx.filter_jit
    def next_token(m,ids,st,position):
        seq=batch(ids,mask=jnp.zeros((8192,),dtype=bool));pref=prefix(m,seq)
        chunk_seq=Batch(input_ids=seq.input_ids[-1024:],target_tokens=seq.target_tokens[-1024:],loss_masks=seq.loss_masks[-1024:])
        result=m.language_model.suffix_call(prefix_outputs=pref[-1024:],state=clone_state(st),seq=chunk_seq)
        return jnp.argmax(result.logits[position-7168].astype(jnp.float32)).astype(jnp.int32)

    @eqx.filter_jit
    def gold_score(m,ids,st,mask):
        seq=batch(ids,mask=mask);pref=prefix(m,seq)
        chunk_seq=Batch(input_ids=seq.input_ids[-1024:],target_tokens=seq.target_tokens[-1024:],loss_masks=seq.loss_masks[-1024:])
        result=m.language_model.suffix_call(prefix_outputs=pref[-1024:],state=clone_state(st),seq=chunk_seq)
        lp=jax.nn.log_softmax(result.logits.astype(jnp.float32),axis=-1)
        lp=jnp.take_along_axis(lp,chunk_seq.target_tokens[:,None],axis=-1)[:,0]
        total=jnp.sum(jnp.where(chunk_seq.loss_masks,lp,0))
        return total,total/jnp.sum(chunk_seq.loss_masks),lp

    def recall(m,r,retention):
        spec=r['recall'][retention];prompt=spec['prompt_ids'];plen=len(prompt);assert plen==8160
        ids=prompt+[int(cfg.model.eos_token_id)]*32
        assert len(ids)==8192 and len(spec['gold_ids'])<=32
        st=prefill(m,jnp.asarray(ids,dtype=jnp.int32));jax.block_until_ready(st)
        # Context prefill ends before the final chunk; every final-chunk evaluation
        # reuses this state, never its returned KV, preventing duplicate replay.
        goldids=ids.copy();goldids[plen:plen+len(spec['gold_ids'])]=spec['gold_ids']
        mask=[False]*8192
        for j in range(plen,plen+len(spec['gold_ids'])):mask[j-1]=True
        total,mean,lp=gold_score(m,jnp.asarray(goldids),clone_state(st),jnp.asarray(mask))
        values=np.asarray(lp)[[j-7168 for j,v in enumerate(mask) if v]]
        assert np.isfinite(values).all() and abs(float(values.sum())-float(total))<1e-4
        generated=[]
        for i in range(32):
            t=int(next_token(m,jnp.asarray(ids),clone_state(st),jnp.asarray(plen+i-1,dtype=jnp.int32)))
            ids[plen+i]=t;generated.append(t)
            decoded=tok.decode(generated,skip_special_tokens=False)
            if t in (128001,128009) or '\n' in decoded:break
        raw=tok.decode(generated,skip_special_tokens=False)
        prediction=tok.decode([t for t in generated if t not in (128001,128009)],skip_special_tokens=False).split('\n',1)[0].strip()
        # Existing RULER single-needle convention: answer string occurs in output.
        accuracy=int(r['gold'].lower() in prediction.lower())
        return {'accuracy':accuracy,'prediction':prediction,'raw_prediction':raw,'generated_ids':generated,'gold':r['gold'],'gold_ids':spec['gold_ids'],'gold_token_logp':values.tolist(),'gold_nll_mean':-float(mean),'gold_nll_sum':-float(total),'prompt_sha256':spec['prompt_sha256'],'prompt_tokens':plen,'max_local_tokens':8192,'retained_tokens':spec['retained_tokens'],'recent_tail_tokens':spec['recent_tail_tokens'],'retained_span_retrained':False,'source_kv_discarded':True,'recall_inner_updates':0}

    with mesh:
        for r in records:
            if r['sample_id'] in done_ids:continue
            if processed>=args.max_new_samples:break
            processed+=1
            print('Sample '+str(r['sample_id']),flush=True);ts=time.monotonic()
            source=r['source_ids'];nchunks=(len(source)+1023)//1024
            seq=batch(source+[int(cfg.model.eos_token_id)]*(nchunks*1024-len(source)),real=len(source))
            pref=prefix(base,seq);jax.block_until_ready(pref)
            # OFF traverses the same original source and native caches but no update.
            st=clone_state(state_suffix);off_history=[]
            for k in range(nchunks):
                chunk=Batch(input_ids=seq.input_ids[k*1024:(k+1)*1024],target_tokens=seq.target_tokens[k*1024:(k+1)*1024],loss_masks=seq.loss_masks[k*1024:(k+1)*1024])
                st,loss=forward_source(base,st,chunk,pref[k*1024:(k+1)*1024]);off_history.append(float(loss))
            del st
            assert tuple(map(float,deltas(base,base)))==(0.,0.)
            # ON follows the official full-prefix + native sequential suffix update.
            m=base;opt=base.inner_optimizer(clone_state(state_all)).init(base.inner_parameters());state_tuple=(clone_state(state_all),clone_state(state_suffix));history=[]
            for k in range(nchunks):
                chunk=Batch(input_ids=seq.input_ids[k*1024:(k+1)*1024],target_tokens=seq.target_tokens[k*1024:(k+1)*1024],loss_masks=seq.loss_masks[k*1024:(k+1)*1024])
                result=update(m,opt,state_tuple,chunk,pref[k*1024:(k+1)*1024]);jax.block_until_ready(result.metrics)
                frozen_step,fast_step=map(float,deltas(result.new_model,m));frozen,fast=map(float,deltas(result.new_model,base))
                assert frozen_step==frozen==0 and fast_step>0
                m=result.new_model;opt=result.new_optimizer_state;state_tuple=result.new_state
                history.append({'chunk':k,'write_loss_pre':float(result.metrics[MetaModel.MetricType.loss]),'loss_tokens':int(jnp.sum(chunk.loss_masks)),'fast_step_max_abs_delta':fast_step,'fast_cumulative_max_abs_delta':fast,'frozen_max_abs_delta':frozen})
                del result
            del seq,pref,chunk,opt,state_tuple
            cells={}
            for ttt,model in [('OFF',base),('ON',m)]:
                for retention in ('NONE','ORACLE','RANDOM'):
                    print('  readout '+ttt+' '+retention,flush=True)
                    cells[ttt+'_'+retention]=recall(model,r,retention)
                    assert tuple(map(float,deltas(model,base)))==((0.,0.) if ttt=='OFF' else (0.,fast))
                    print('  result '+json.dumps(cells[ttt+'_'+retention]),flush=True)
            record={'sample_id':r['sample_id'],'source_tokens':len(source),'source_native_chunks':nchunks,'source_loss_tokens':len(source)-1,'needle_to_query_distance':r['needle_to_query_distance'],'selection':r['selection'],'write_ON':history,'write_OFF_loss':off_history,'OFF_inner_updates':0,'ON_inner_updates':nchunks,'same_ON_fast_state_all_retention_conditions':True,'cells':cells,'seconds':time.monotonic()-ts}
            completed.append(record)
            with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            del m,cells
        assert reuse._hash_pytree(base.inner_parameters())[0]==fast_hash
        assert reuse._hash_pytree(eqx.partition(base,inner_filter)[1])[0]==frozen_hash
    if len(completed)<len(records):print('partial run',len(completed),'of',len(records));return
    summary={'n':len(completed),'invariants_passed':True,'elapsed_seconds':time.monotonic()-start,'jax_memory_stats':jax.devices()[0].memory_stats(),'conditions':{},'interaction':{}}
    for ttt in ('OFF','ON'):
        for retention in ('NONE','ORACLE','RANDOM'):
            key=ttt+'_'+retention;c=[r['cells'][key] for r in completed]
            summary['conditions'][key]={'n_correct':sum(x['accuracy'] for x in c),'accuracy':float(np.mean([x['accuracy'] for x in c])),'gold_nll_mean':float(np.mean([x['gold_nll_mean'] for x in c])),'gold_nll_sum':float(np.mean([x['gold_nll_sum'] for x in c]))}
    for retention in ('ORACLE','RANDOM'):
        paired=[]
        for r in completed:
            c=r['cells'];paired.append({'sample_id':r['sample_id'],'accuracy':(c['ON_'+retention]['accuracy']-c['ON_NONE']['accuracy'])-(c['OFF_'+retention]['accuracy']-c['OFF_NONE']['accuracy']),'nll_gain':(c['ON_NONE']['gold_nll_mean']-c['ON_'+retention]['gold_nll_mean'])-(c['OFF_NONE']['gold_nll_mean']-c['OFF_'+retention]['gold_nll_mean'])})
        summary['interaction'][retention]={'accuracy':float(np.mean([x['accuracy'] for x in paired])),'nll_gain':float(np.mean([x['nll_gain'] for x in paired])),'paired':paired}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in protected.items())
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str),flush=True)
if __name__=='__main__':main()
