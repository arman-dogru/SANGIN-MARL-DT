#!/bin/bash
cd /workspace/leo_study
for d in redesign_archive/20261027-004052/episodes/A4/W1/*/*/agents-*; do
  r=$(basename $(dirname $(dirname $d))); c=$(basename $d)
  grep -q "\"route\": \"$r\", \"case_id\": \"$c\"" redesign_aoi/A4_archive_episodes.jsonl 2>/dev/null && continue
  while [ $(free -m | awk '/Mem:/{print $7}') -lt 4096 ]; do echo "wait mem $(date +%T)"; sleep 30; done
  nice -n 19 env OMP_NUM_THREADS=1 /workspace/SANGIN-dev/.venv/bin/python harness/redesign_archive_extract.py $d 2>&1 | grep -v Warning | tail -3
done
echo ALLDONE
