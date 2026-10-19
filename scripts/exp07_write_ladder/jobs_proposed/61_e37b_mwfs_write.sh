#!/bin/bash
# E37b write job (PROPOSED, NOT QUEUED): chunk-only masked write ("masked-fs", level L0mwfs, block 1024, step x1), world seed 101, worlds A and B, 4 passes
# (pass-wise fresh filler = ladder.fs_streams as job 37), save mode: one inner-weight npz per pass under $OUT/states/L0mwfs_mwfskeep_s101/{A,B}/state_00{0..4}.npz (state_000 = W0).
# Same code path as the old x1 runs; the new tag keeps the files. Reproducibility check afterwards: compare the per-pass inner hashes in L0mwfs_mwfskeep_s101_{A,B}.saved.json
# with the old L0mwfs_x1_s101_{A,B}.json (field margins._hash / hash). Expected 1-2 h GPU. Start only after Hans approves (jobs/held -> jobs/).
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
$PY $S/ladder.py --level L0mwfs --scale 1 --block 1024 --passes 4 --worlds AB --world-seed 101 --mode save --tag mwfskeep
ls -l $OUT/states/L0mwfs_mwfskeep_s101/A/state_004.npz $OUT/states/L0mwfs_mwfskeep_s101/B/state_004.npz
