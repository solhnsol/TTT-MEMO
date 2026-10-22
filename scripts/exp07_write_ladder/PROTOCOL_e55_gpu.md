# E55 단계 0 코드 인터페이스 (코드 스레드, 2026-10-22 14:30 KST 갱신)
상태: (a)(b)(c)(d)의 코드는 이미 있고 CPU(작은 무작위 e2e 모델, 진짜 SWA 코드)에서 `python3 e55_units.py`(U0~U12) 전부 통과. **3B 모델·진짜 토크나이저·진짜 데이터·GPU 경로는 한 번도 실행되지 않았다.** 아래는 파일/함수/형식. 브랜치 claude/project-thread-rwhdac.

## 파일
- `scripts/exp04_what_ttt_retains/gen_world_ctl.py`: 세계 (`build_ctl_world`, `realize(world,swap,enc,decoy)`, `build_items`, `rows`, `stream_sha`, `CELLS`, `READ_APPS=(2,4,8,16)`).
- `e55_swap.py`: `plan(arm,k,p)`, `compose_window(src,k,pl,CH,axis)`. `e55_core.py`: `compose_state`, `keep_vector`, `Ladder55(rt,save_dir,canary).run_stream(...)` (s3.Ladder3 위). `e55_run.py`: CLI `dry | unit | run | read`. `e55_read.py`: `plan(world,enc)` (CPU 읽기 계획). `e55_gpu.py`: 판독기 `Reader55`, `read_world`, `make_records`. `e55_analyze.py`: `stage0(nat_files,swd_files)` + 규칙 1–9. `e55_units.py` + `e55_cpu_harness.py`: CPU 단위검사.
## 저장 상태 (러너 출력 → 판독기 입력)
- 가중치: `OUT/states/E55_<tag>_<ARM>_s<seed>/<A|B>/state_<ARM>_<A|B>_<chunk:03d>.npz` = inner 파라미터 leaf를 `np.savez(arr_0..)` (ladder.save_state/load_state와 같은 형식, `Reader55.load_state(file, hash)`가 해시 검증).
- 로그: `OUT/E55_<tag>_<ARM>_s<seed>_<A|B>.saved.json` = `{'log':[청크별 k,gnorm,coef,update_norm,ce,sec,n_swapped_slots], 'reads':{'<read_chunk>':{'_file','_hash','canary':{'_gen_nll','_tail_nll'}}}, 'canary_w0', 'stream_hash', 'stream_sha', 'decoy_sha', 'clip_bind_frac', 'final_hash', ...}`. 판독점 청크 = `world['read_chunks']` = 등장 a회 직후 `a*p-1`, a∈READ_APPS.
## 판독 입력 / 출력
- `e55_read.plan(world,enc)` → `{'items','rows','read':{a:{'read_chunk','ctx':{name:{'end','d','tail_A','tail_B'(8192 ids)}}}}}`; 'out' 꼬리는 `ladder.squad_tail(tok,bos,forbid)`.
- `e55_gpu.read_world(rd,raw,n_items,16,plan['read'],out_tail,state_files={arm:{a:{'A':(file,hash),'B':...}}},w0_model=rt.base,only_apps,in_apps)` → `(read[arm][a][ctx][w] = (n_items,16), W0)`; W0['out']=S, W0['<ctx>@<a>']={'A':S,'B':S}. `make_records` → 팔마다 JSON `E55_<tag>_<ARM>_s<seed>.read.json`: `{arm,cell,p,M,s,bank,seed,meta:{kind,pair,gi,fi,qframe,name_id,d:{a:{ctx:[d]}}},read:{a:{ctx:{A,B}}},W0:{...}}` → `e55_analyze.stage0`.
## 서버에서만 확인되는 것
`e55_run.py unit`(GPU U1~U5, 종료코드 9=실패), 3B 속도·메모리(기록 호스트 메모리 추정은 `dry`가 출력), 진짜 토크나이저 후보 span 길이(`read`가 출력), canary.
