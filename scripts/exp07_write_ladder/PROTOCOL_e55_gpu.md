# E55 단계 0 GPU 구현 인터페이스 (코드 스레드 작성, 2026-10-22 KST)
대상: 연구실 서버(+ e2e 소스)에서 GPU 부분을 구현하는 세션. 설계: `experiments/E55-controlled-world-binding-design.md`(2.3 팔, 2.4 측정, 2.5 창 안 읽기, 4절 판정, 6절 변경 목록). 이 문서는 저장소(`scripts/exp07_write_ladder/PROTOCOL_e55_gpu.md`)에 있어 서버에서 `git fetch`로 읽을 수 있다. **상태: 이 문서의 GPU 코드는 한 번도 작성·실행되지 않았다. CPU 쪽(세계 생성, 슬롯 계획, 읽기 계획, 분석)은 가짜 토크나이저·합성 데이터로만 시험했고 실제 토크나이저/서버 값은 미확인이다.**

## 0. 이미 있는 것 (커밋 d6f8d88, CPU 전용, 수정하지 말고 import)
- `scripts/exp04_what_ttt_retains/gen_world_ctl.py`: `build_ctl_world(seed,M,s,p,n_app,bank,enc,bg)` → 세계(진짜 스트림 `chunks` + 디코이 `decoy`), `realize(world,swap,enc,decoy=False)` → 청크마다 `ids`(정확히 1024), 문장 위치 `sents`; `build_items(world)` → 이름별 문항(질문 틀 3개, 후보 16개, gi/fi, 대조 4쌍); `rows(items)`; `stream_sha(realize(...))`; `CELLS`(9개 셀 정의); `READ_APPS=(2,4,8,16)`, 판독 청크 `world['read_chunks']` = 등장 a회 직후의 청크 인덱스 `a*p-1`. `enc` = `load_enc(tokenizer.json)`. 배경은 `default_bg()`(gen_world.background, 서버 데이터 필요).
- `scripts/exp07_write_ladder/e55_swap.py`: `plan(arm,k,p)` → {lag: 'real'|'decoy'|'bg'|'mask'}, `compose_window(src,k,plan,CH)`(토큰 축이 맨 앞인 numpy 배열 기준), `write_chunk`. 팔: NAT, SWD, SWB, MSK, SWDS(동그룹), SWDO(타그룹).
- `scripts/exp07_write_ladder/e55_read.py`: `plan(world,enc)` → 문항, 행, 판독점별 문맥(창 안 `in_e0/1/3/7` 꼬리 ids, 쌍별 정보 거리 d), 기록 형식(모듈 docstring). `scripts/exp07_write_ladder/e55_analyze.py`: 기록 → 지표·규칙.
- 서버 CPU 점검 잡(제안): `jobs_proposed/164_e55_world_stats_cpu.sh`. 이 결과(풀 크기, 세계 통계, 재현 해시)가 나오기 전에는 세계가 실제 토크나이저에서 만들어지는지 모른다.

## (a) 스트림을 한 번 읽는 실행기 `e55_run.py` (신규)
- **입력**: `--seed --M --s --p --n-app 16 --bank {50,12,disjoint} --arm {NAT,SWD,SWB,MSK,SWDS,SWDO} --worlds AB --tag --keep [--smoke-chunks N]`. 가중치는 W0(rt.base)에서 시작, **한 번만** 읽음(pass 반복 없음), 쓰기 청크는 모든 팔에서 진짜 토큰(`gen_world_ctl.realize(world,swap,enc)['ids']`).
- **메커니즘**: `s3.py`의 `Ladder3.run_cell` 구조를 따르되 `NCH=63`/pass 루프를 `world['n_chunks']`/한 번으로 바꾼다. NAT은 `native_step`(플래그 끔) 또는 빈 집합 write_step 중 하나를 고르고 어느 쪽인지 기록(단위 시험 U1이 두 경로의 일치를 보인다). prefix 층 출력 `pref_nat=rt.prefix(rt.base,seq)`는 가중치와 무관(frozen)하지만 144청크(p=9, 약 147k 토큰)는 63청크 때보다 길어 메모리가 모자랄 수 있다 → 구간으로 나누는 경로가 필요한지 서버에서 먼저 확인(단계 0의 p=2는 32청크).
- **상태 저장**: 판독 청크 r마다 쓰기 후 내부 가중치 저장(`Ladder.evaluate(save_dir)`와 같은 방식, `states/<TAG>_s<seed>/<w>/state_<r>.npz`). 로그 항목 `{'read_chunk':r,'app':a,'margins':{'_file':..,'_hash':..},...}`로 e49식 해시 검증이 가능해야 한다.
- **출력**: `OUT/E55_<tag>_s<seed>_{A,B}.saved.json` = `{'log':[청크별 {k,gnorm,coef,update_norm,ce,sec,n_swapped_slots}, 판독점 항목...],'stream_hash':...,'clip_bind_frac':...}`; canary(+0.10 nat 한도: `_gen_nll`,`_tail_nll`, `Ladder.ev(canary_only)`)를 판독 청크마다.
- **검증 기준**: (1) `stream_hash` = `gen_world_ctl.stream_sha(realize(world,swap))`(A, B 각각), (2) 모든 청크 길이 1024, (3) 같은 시드로 두 번 돌린 로그가 비트 동일(결정적 연산 플래그 `xla_gpu_deterministic_ops=true`), (4) clip 결합 비율과 업데이트 노름이 로그에 있음, (5) 4청크 스모크(`tag`에 smoke 포함).

## (b) SWD 바꿔치기 (prefix 층 + suffix 층 carried KV)
- **정의**: 청크 k를 쓸 때 창(직전 8청크)의 슬롯 lag j(청크 k−j) 내용만 `e55_swap.plan(arm,k,p)`대로 디코이(또는 배경만 스트림, 또는 가림)로 바꾼다. 절대 위치(RoPE)는 같다(디코이 스트림도 같은 청크 인덱스에 기록). 쓰기 청크와 그 청크의 prefix 층 은닉은 진짜 토큰.
- **prefix 층(24)**: `Ladder3.record(ids)`를 진짜 스트림(월드 A, B 각각), 디코이, 배경만 스트림에 대해 한 번씩 → 호스트 K,V `[24,T,nh,d]`. `ctx_kv(Kh,Vh,k)` 대신 `e55_swap.compose_window`와 같은 규칙(층 축을 맨 앞에 두는 변형)으로 창 [24,8192,nh,d]를 조립해 `prefix_masked(m,ids_k,Kc,Vc,q0,keep)`에 넘긴다. keep은 NAT/SWD/SWB에서 전부 True(앞쪽 부재 슬롯은 기존 ctx_kv처럼 0 패딩 + False), MSK에서만 창 부분 False. 호스트 메모리: s3의 `record` 로그("recorded natural K,V (x GiB host)")에 있는 63청크 값을 기준으로 스트림 수(진짜 A, 디코이, 배경 = 3벌)와 청크 수를 곱해 먼저 어림해야 한다(본 환경에서는 값을 모른다).
- **suffix 층(8)**: `write_step(m,st_all,st_suf,c,pref_w,kmask)`의 `st_suf`는 진짜 스트림의 자연 읽기로 이어 온 SWA 상태다. 필요한 것: 진짜/디코이/배경 세 갈래의 상태 사슬을 **같은 현재 가중치 m**으로 각자의 청크에 `read_state`로 나란히 전진시키고(슬롯 내용은 그 청크가 처리될 당시 가중치로 만들어진 K,V여야 함 — 쓰기에 쓰인 가중치는 팔에 상관없이 진짜 실행의 것), 청크 k의 쓰기 직전에 세 사슬의 슬롯 블록을 plan대로 조립해 `st_suf`를 만든다. **SWA 상태의 배열 레이아웃(링 버퍼 여부, 위치 저장, 블록 경계)은 e2e 소스(`ttt.model`의 SWA 상태)를 읽어야 알 수 있고 이 저장소 체크아웃에는 없다. 먼저 레이아웃을 기록하고, 블록 단위 읽기/쓰기 함수 `get_block(state, j)`/`set_block(state, j, blk)`를 만들어 단위 시험으로 왕복(읽고 같은 자리에 쓰면 원상)을 확인할 것.**
- **검증 기준**: (U2) 디코이 녹화를 진짜 스트림으로 둔 SWD는 NAT과 CE·기울기 일치(비트 동일 목표, s3의 U-a 허용 오차 이내), (U5) p=9에서 SWDS는 NAT과 일치(슬롯 lag 1..8 중 9의 배수 없음), (U3) 쓰기에 들어간 청크 ids가 팔에 상관없이 진짜 청크, (U4) 진짜 디코이일 때 존재하는 슬롯에서만 K,V가 달라지고 CE/기울기(코사인 < 1)가 변함.

## (c) 후보 16개·창 안 읽기 판독기 `e55_read` GPU 경로
- **입력**: 저장 상태(판독 청크별), `e55_read.plan(world,enc)`의 `rows`/`ctx`. 점수 = 후보 장소 span 로그확률 합(e49와 동일: `stage2.item_ids(tok,slot,prefix,cand)`의 span에서 1을 뺌, fp32 `packing.Packed(ctx,raw,GP=2).r1(m32)`).
- **문맥**: 'out' = `ladder.squad_tail`(금지 이름 목록에 진짜·디코이·대조 이름과 장소 전부 넣을 것), 'in_e<lag>' = `ladder._TailCtx(rt32, tail_ids)`에 진짜 스트림 청크 e−7..e(8192 토큰, 상태의 월드 A면 A 스트림, B면 B 스트림). 후자는 꼬리 길이 8192가 기존 7168과 다르므로 `Packed`가 꼬리 길이를 1024의 배수로만 가정하는지 확인(가정은 `len(T.tail_ids)//1024`).
- **출력**: `e55_read.py` docstring의 기록 형식(`read[a][ctx]['A'|'B']` = (문항수, 16) 점수, `W0[ctx]`, `meta`에 `kind/pair/gi/fi/qframe/name_id`와 `d[a][ctx]`). W0는 세계마다 한 번(A, B 같은 값).
- **주의**: `Ladder(rt,seed,readout=True)`는 옛 세계의 Readouts를 만든다. 런타임·`to_fp32`·`load_state`·`_TailCtx`만 쓰도록 분리하거나, 옛 세계 구성의 비용을 감수할 것.
- **검증 기준**: (1) W0를 두 번 읽어 최대 |차| = 0, (2) 후보 모두 같은 토큰 길이(`world['place_len']`)와 span 길이 일치, (3) 'in' 꼬리의 첫 1024 토큰 sha = CPU 계획의 청크 e−7 sha, (4) 같은 상태를 두 번 읽은 점수 동일, (5) 읽은 가중치 해시 = 저장 로그 `_hash`, (6) 합성 점수가 아닌 실제 기록으로 `e55_analyze.rule1`..`rule9`가 오류 없이 돌아간다.

## (d) GPU 단위 시험 (GPU 필요, 세계 p=2 앞 12청크 정도로 충분)
U1 NAT 경로 일치(위 (a)), U2 디코이=진짜 항등, U3 쓰기 청크 불변, U4 바꿔치기가 실제로 무엇을 바꾸는지, U5 p=9 음성 대조 항등, U6 MSK = `s3.py --cell prefixall`의 가림(k6p 창 전체 가림)과 CE 일치, U7 디코이 녹화의 절대 위치/모양이 진짜와 같음, U8 읽기 항등(위 (c) 1·4), U9 스트림 해시·시드 재현, U10 `--smoke-chunks 4` 전 팔 한 바퀴. 각 시험은 숫자와 허용 오차를 4절에 기록, 하나라도 실패하면 단계 0을 큐에 넣지 않는다(설계 규칙 8).

## 책임과 규칙
- 실험 파일 3절은 코드 스레드, 4절은 실행 스레드가 쓴다(RULES.md). RC 세션이 위 (a)~(d)를 구현해도 괜찮다면 제안: RC는 서버 작업 트리에서 구현·GPU 단위 시험을 돌리고, 결과(코드 diff 파일 + 시험 숫자)를 코디네이터 경유로 코드 스레드에 넘긴다 → 코드 스레드가 CPU 모듈과의 인터페이스를 검토해 `claude/project-thread-rwhdac`에 커밋하고 3절에 sha를 적는다 → 실행은 그 고정 커밋으로만. 이렇게 하면 3절(코드·sha)과 4절(실행 기록)의 소유가 그대로다. RC가 `jobs/`에 파일을 추가·변경하는 일은 기존대로 Hans가 실험 실행 스레드에 직접 쓴 허락이 필요하다.
