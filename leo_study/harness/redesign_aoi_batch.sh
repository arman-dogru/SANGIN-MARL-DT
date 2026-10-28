#!/bin/bash
# 30 A4 episodes, serial, nice 19, each started only when >= 4 GiB available. Skips episodes already in A4_episodes.jsonl.
cd /workspace/leo_study
for n in 8 16; do for r in allocated cloud edge; do for c in 0 1 2 3 4; do
  cid=agents-$n-case-00$c
  grep -q "\"route\": \"$r\", \"case_id\": \"$cid\"" redesign_aoi/A4_episodes.jsonl 2>/dev/null && continue
  while [ $(free -m | awk '/Mem:/{print $7}') -lt 4096 ]; do echo "wait mem $(date +%T)"; sleep 30; done
  nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/SANGIN-dev/.venv/bin/python harness/redesign_aoi_batch.py $r $cid 2>&1 | grep -v Warning | tail -3
done; done; done
echo ALLDONE
