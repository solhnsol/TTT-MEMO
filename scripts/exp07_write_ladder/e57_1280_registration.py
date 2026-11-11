"""E57 job 1280 (recompute oracle): FORMAL REGISTRATION TEXT (server repo copy; DRAFT, not frozen). Source: the section '11/12 02:00 KST 1280 v5 ...' at the top of claims-map.md in the project library (core-claims thread), copied VERBATIM at 2026-11-11T16:41Z (SOURCE_SECTION_SHA256 below). This file contains only text (no new .md file); the analysis code is e57_1280_analyze.py (v5 draft). Placeholders that must be filled BEFORE the final hash: item 1 (spec hash, commit, timestamp), item 8 (non-binding futility numeric rule), item 13 (Hans decisions a, b, c), item 15 (external timestamp service). Nothing in this registration was written after opening any 1280 or instrumentation-seed output (none exists)."""
SOURCE_SECTION_SHA256='8fc3ac1c7c91f76ababc087b5979fea7543a889a86f9a58e6794c6179e2d5bee'
PLACEHOLDERS=['item 1: spec hash, commit, timestamp','item 8: non-binding futility numeric rule','item 13: Hans decisions (a) placebo delta, (b) SESOI anchor, (c) 1284 primary','item 15: external timestamp service']
REGISTRATION_TEXT=r'''## 11/12 02:00 KST 1280 v5 초안 보고(서버; CPU만, selftest 통과; 해시 별도 도착 예정) + 등록 문안 초안 [자리표시자 3개 미확정 — Hans 결정 후 채움, 해시 전 동결 금지]
**서버 v5 반영:** 14건 모두 반영. α=.025 경계 4.011/2.109, α=.05 경계 MC 재계산 3.179/1.755(이전 값과 ±.004). 분기 청크 3개는 구간 [8,24),[24,44),[44,64)에서 하나씩 고정 난수. 플라시보 비우월 단위 = 절대 nat. MDE .026/.037은 SD 배수 1.5·ρ=0일 때 값(배수 1이면 .017/.025). 결합 검정력 n=10 ≈ .47, n=20 ≈ .78; 병목은 L_F·PL의 SD(.45). choose_k 규칙은 코드에 있음, Kieser–Friede는 미구현. 부호: v3는 처음부터 E형이었고 불일치는 v4 요청 문구에 D형을 표기하면서 생긴 것(docstring 기록). 남은 일: 자리표시자 3개(Hans), 등록 문안 15항목(아래 초안).
**등록 문안 초안(15항목, 서버는 Council 원문을 못 봐서 이 초안을 기준으로 문서화):**
1 머리말: 잡 1280 v5-final, 사양 해시·커밋·타임스탬프[채움], "새 시드(101~120) 출력과 계측 시드 91~94 출력 미열람" 확인, 개정 이력(v1 R3-leak 라우팅 → v3 부호·판정 개정 → v4 Council 14건 → v5), 이전 오라클 파일럿 1279(시드 62~64, K0=0)는 확증에서 제외.
2 가설·부호: E_cache=a−nc, E_rec=b−c, E_mask=X_MASK(차단이 NLL을 올리면 양수), D=−E는 부록; R=E_cache/E_rec; 측정 NLL = 사실 토큰의 갱신 직전 스냅샷 손실; 시드별 |c−nc|≤1e-4 게이트. 이전 리뷰의 D형 표기와 E형 부등호를 짝지어 읽으면 방향이 뒤집히는 규약 불일치가 있었음(v3 자체는 E형).
3 개입·측정: 팔 NATS/NATMASK/ORACLE(캐시)/RECOMP(재계산 오라클)/ORACLE-PL/RECOMP-PL; 측정 a, nc, b, c, a_P, b_P, X_MASK, M(마진), G=M_NATMASK−M_NATS, H=M_RECOMP−M_NATS; 사건 집합 t≥8, 시드당 32사실, 분기 청크 3개([8,24),[24,44),[44,64) 고정 난수), 재계산은 창 전체·후미 절단 없음.
4 추정량: 시드별 선형 대비 T_R=E_cache−.8E_rec, T_Q=E_rec−.7E_mask, L_F=H−.7G, L_Flow=H−.3G, 플라시보 H_PL(절대 nat).
5 게이트: 2a E_rec(하한>0, 점추정≥.25) 실패 → R·Q 불가; 2b E_mask(하한>0, 점추정≥.1) 실패 → Q만 불가; 2c G(하한>0) 실패 → F·F_low 불가; 4행(캐시 충분)은 2a만 필요.
6 추론: 시드 수준 짝 t(df=n−1), 2-look 10/20, α=.025씩 양방향(경계 4.011/2.109) 및 α=.05(3.179/1.755), 성분 기각은 어느 look에서든 처음 교차, 모든 게이트·보고 한계 같은 c_k, MC 200만 세트·RNG 시드 7/12345·MC SE ≈ .000154, "지정 시뮬레이션 모델에서의 추정 1종 오류", t(5) 견고성(보조).
7 판정표: 1차 배타 판정(PROTOCOL INVALID / DENOMINATOR NOT CERTIFIED / 3a LEAK / 3b LEAK+MASK-EXPLAINED / 3c LEAK CONFIRMED / 4 CACHE SUFFICIENT / INCONCLUSIVE) + 독립 플래그(F_low, H-gen, HALF-PL); 합성 입력 단위 테스트가 모든 행을 덮음.
8 중간 행동: 효능 중단은 목표 판정 수준 전체가 충족될 때만, 비구속 무익성 수치 규칙[채움], 종결 판정은 최종 look에서만.
9 시드: 계측 91~94(분산만 봄), 확증 101~120, 1284 201~220, 교체 목록 121~130·횟수 상한 3, 결정성 지표 max|ΔNLL| 자동 검사, NaN은 시드 교체, HALF-PL 미매칭 32사실 중 3개 초과 → 2차 무효, 시도한 모든 시드·사유 보존.
10 검정력·개정 트리거: MDE(T_R n=10 .026 / n=20 .0117, T_Q .037 / .0167; SD 배수 격자 {.5,1,2,3,4.4}×상관 {−.5,0,.5,.9}), 결합 검정력 n=10 ≈.47·n=20 ≈.78, 분기 청크 k는 계측 시드 분산만 보고 choose_k 규칙으로, 검정력이 촉발한 개정은 해시 전에만.
11 민감도·2차: 원래 1차였던 TOST ±.25는 이제 민감도로 표기, Fieller F, D형 부록, HALF-PL, E_cache>1.2·E_rec 이상 플래그.
12 공시(정확한 문구): 파일럿 시드별 X_MASK .337/.367/.383, X_ORACLE .076/.065/.079; PL−NATS +.096/+.036/−.003(평균 +.043, SD .050), NATORACLE_RAND−NATS +.048/−.010/−.022(평균 +.005), 둘 다 탐색적이며 RECOMP-PL의 분산을 보정하지 않음; df=2 주의(시드 SD 95% 상한 ~4.4배); 임계값이 파일럿 뒤에 정해졌음; 파일럿 E_cache≈.073이므로 E_rec≥.25 통과 시 4행(캐시 충분)은 사실상 도달 불가이고 누출 판정이 거의 자동으로 나올 것이 기대됨; 파일럿에 b·c·F 분산 없음; 검정력 계산이 개정을 촉발했음; 부호 규약 불일치 이력.
13 Hans 결정[미확정]: (a) 플라시보 비우월 δ = 절대 nat, 값 .5 vs .25; (b) SESOI 앵커 = K0=8 .455 nat 태그 면제 vs K0=0 보조 측정(시드 901~908) vs .5 문턱 동결; (c) 1284 1차 = RECOMP-ORACLE−RECOMP-PL 기본(두 모델 권고).
14 1284 사양: 시드 201~220, 1차(13c), 핵심 2차 HALF-PL−HALF, 새 시드 비중복, 1280 판정은 해석에만 반영하고 1차를 바꾸지 않음.
15 편차 정책: 모든 편차는 타임스탬프와 함께 기록, 탐색용 표기. 외부 타임스탬프 서비스[Hans 결정 대기: 공개 등록소 vs RFC 3161 vs 로컬 해시+서버 타임스탬프].
'''
def selftest():
    import hashlib
    assert hashlib.sha256(REGISTRATION_TEXT.strip().encode()).hexdigest()==SOURCE_SECTION_SHA256,'registration text differs from the copied source section'
    assert len(PLACEHOLDERS)==4 and '15 편차 정책' in REGISTRATION_TEXT
    print('e57_1280_registration selftest ok')
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:selftest()
