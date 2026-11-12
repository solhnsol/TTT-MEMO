#!/bin/bash
# JOB 244 = E55 32.12 sentence dump with per-head attention for seeds 62, 63, 64 (GPU, forward only, three sequential calls of the per-seed job 244_sd_one_seed.sh = job 204b with the same code commit e632178 as jobs 238 / 239), then candidate-1 confirmation and the frozen learned head weighting of seed 61 (e55_sentgrid_ext.py, commit 9b93643) on each seed. A failed seed does not stop the next ones; the exit code is the number of failed seeds.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE from the measured time of job 204b (seed 61: 18 min 27 s including model load, CPU analysis a few minutes): about 20-25 min per seed, about 1.0-1.3 h for three. GPU. NOT APPROVED yet.
cd "$(dirname "$0")/../../.." 2>/dev/null || true
fail=0
for s in 62 63 64; do echo "=== SEED $s"; SEED=$s bash scripts/exp07_write_ladder/jobs_proposed/244_sd_one_seed.sh || { echo "seed $s FAILED rc=$?"; fail=$((fail+1)); }; done
# KEEP: the full dumps (per layer and head AL / AV, per token T_*, same format as seed 61) stay in $OUT, made read-only, listed with path, size, sha16, commit, seed, world in e55_sentdump_manifest.json (no cleanup script touches them; backups pull them with the rest)
OUT=${EXP07_OUT:-experiments/exp07_write_ladder}
python3 - "$OUT" <<'PY'
import sys,json,hashlib,os,subprocess,glob
out=sys.argv[1];code=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip();m={'world':'cell v3_16_1','note':'sentence dumps: per layer and head AL / AV, per token T_*, VN, ids; states NAT and SENTPQADAMH at chunks 16, 32, 64, 127; labels (fact ids) in the labels file','entries':[]}
for sd in (61,62,63,64):
    for f in (f'e55_sentdump_s{sd}.npz',f'e55_sentdump_labels_s{sd}.npz',f'e55_sentdump_s{sd}.json'):
        p=os.path.join(out,f)
        if not os.path.exists(p):continue
        h=hashlib.sha256()
        with open(p,'rb') as fh:
            for b in iter(lambda:fh.read(1<<24),b''):h.update(b)
        if f.endswith('.npz'):os.chmod(p,0o444)
        m['entries'].append({'path':p,'bytes':os.path.getsize(p),'sha16':h.hexdigest()[:16],'seed':sd,'dump_code_commit':'e632178a94ca5143d31c247244ec0bde4927e06b' if sd>=62 else '298515850467f0a362367da0f6d72447e5d382c7','manifest_written_at_head':code})
json.dump(m,open(os.path.join(out,'e55_sentdump_manifest.json'),'w'),indent=1);print('manifest:',len(m['entries']),'files')
PY
exit $fail
