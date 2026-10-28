#!/bin/bash
# JOB 333 = job 305 with the V / Vn masks of the dump of job 313 (e57_keyq_vmasks.py, k=512, 128 keys per GAINQ512 row, fallback = highest recall): ARMS=KQ512V_ALLVn_cumq_min
export ARMS=KQ512V_ALLVn_cumq_min
export MASKS=/home/solhnsol/workspace/TTT-MEMO/experiments/exp07_write_ladder/e57_keyq_v_k512/e57_keyq_v_masks.npz
export MASKS_SHA=2c35037c392bc0a16948be91bd6a99867b9b49010c91cea0b752de72bebfb4f3
exec bash /home/solhnsol/workspace/TTT-MEMO/scripts/exp07_write_ladder/jobs_proposed/305_e57_news_keyq_write_arm.sh
