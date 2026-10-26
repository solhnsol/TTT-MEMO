#!/bin/bash
# Upload the inputs of job 276 to the RunPod volume through a cheap CPU pod (Hans 16:48Z). Writes ONLY under /workspace/t1t2_in/ (the destination is hard-coded; nothing of the running E57 chain is touched). Usage: HOST=... PORT=... [USER=root] KEY=<private key> STAGE=<staging dir with exp07_out/ and TTT-MEMO/> bash pod_t1t2_upload.sh
set -eo pipefail
: "${USER_:=${REMOTE_USER:-root}}"; : "${HOST:?}"; : "${PORT:?}"; : "${KEY:?}"; : "${STAGE:?}"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
DEST=/workspace/t1t2_in
$SSH $USER_@$HOST "mkdir -p $DEST && df -h /workspace | tail -1"
date +%T; t0=$(date +%s)
rsync -rLtv --partial --inplace --progress -e "$SSH" "$STAGE/exp07_out" "$STAGE/TTT-MEMO" $USER_@$HOST:$DEST/
echo "copy seconds: $(( $(date +%s) - t0 ))"
scp -i $KEY -P $PORT -o StrictHostKeyChecking=accept-new "$STAGE/exp07_out.sha256sums" "$STAGE/TTT-MEMO.sha256sums" $USER_@$HOST:$DEST/
$SSH $USER_@$HOST "cd $DEST/exp07_out && sha256sum -c ../exp07_out.sha256sums && cd $DEST/TTT-MEMO && sha256sum -c ../TTT-MEMO.sha256sums && ls -la $DEST/exp07_out $DEST/exp07_out/e57_t1t2_states && ls /workspace/exp07_out/e57_news/e57_bank/bank_v1.json /workspace/exp07_out/e57_news/e57_bundle/bundle_store.pkl"
