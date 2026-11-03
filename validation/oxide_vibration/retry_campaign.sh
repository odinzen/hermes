#!/usr/bin/env bash
# Rerun run_campaign.sh (unchanged, resumable) after each pass until every case is done, at most 3 more passes.
# A transient CUDA error failed MgO on the first pass (2026-10-23 18:49).
cd "$(dirname "$0")"
for pass in 2 3 4; do
  while pgrep -f "bash run_campaign.sh" > /dev/null; do sleep 300; done
  left=$(for t in mgo al2o3 tio2 mgtio3 mg3po42 la2o3 y2o3 lu2o3 sc2o3 srtio3; do
           grep -q CASE_DONE "runs/$t/progress.log" 2>/dev/null || echo $t; done)
  [ -z "$left" ] && break
  echo "$(date -Is) pass $pass for: $left" >> campaign.log
  bash run_campaign.sh
done
