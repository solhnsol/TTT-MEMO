#!/bin/bash
# JOB 294 = E57 NEWS GAINQ BAND WRITE ARMS GAINQ128 / GAINQ512 with P=0.50 (RunPod, parallel to the lab run of job 292 with P=0.26; Hans 07:26Z "p는 .26으로 가고 .10이랑 .50도 RunPod에서 병렬로"): the unchanged script of job 292 with P=0.50; masks are generated inside the pod and checked against the det hash of this p. NOT APPROVED yet (needs Hans job name + commit + time). ESTIMATE ONLY, unmeasured: about 0.75-0.85 h, 2 states of 1.4 GB, output $OUT/e57_gainq_p0.50.
export P=0.50
exec bash "$(dirname "$0")/292_e57_news_gainq_band_write_arms.sh" "$@"
