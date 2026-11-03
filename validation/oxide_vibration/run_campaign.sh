#!/usr/bin/env bash
# Production runs, one case at a time on the GPU. Cases with measured data first; Sc2O3 and SrTiO3
# have none and run last (the rule reports them, it does not score them). Resumable: rerun to continue.
cd "$(dirname "$0")"
PY=~/scratch/mace-gpu/bin/python
for tag in mgo al2o3 tio2 mgtio3 mg3po42 la2o3 y2o3 lu2o3 sc2o3 srtio3; do
  if grep -q CASE_DONE "runs/$tag/progress.log" 2>/dev/null; then continue; fi
  echo "$(date -Is) start $tag" >> campaign.log
  $PY run_oxides.py "$tag" >> "campaign_$tag.out" 2>&1 || echo "$(date -Is) FAILED $tag" >> campaign.log
done
echo "$(date -Is) ALL_DONE" >> campaign.log
