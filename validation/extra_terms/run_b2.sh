#!/usr/bin/env bash
# B2 phosphate MD with the oxide vibration route unchanged (run_oxides.py), queued after Arm A so runs never share
# the GPU. Up to three passes, as a transient CUDA error can fail a case.
cd "$(dirname "$0")/../oxide_vibration"
sleep 60
while pgrep -f "bash run_arm_a.sh" > /dev/null || pgrep -f "bash run_campaign.sh" > /dev/null \
      || pgrep -f "bash retry_campaign.sh" > /dev/null; do sleep 600; done
PY=~/scratch/mace-gpu/bin/python
for pass in 1 2 3; do
  for tag in ndpo4 eupo4 gdpo4 hopo4 erpo4 tmpo4 ybpo4; do
    if grep -q CASE_DONE "runs/$tag/progress.log" 2>/dev/null; then continue; fi
    echo "$(date -Is) B2 pass $pass start $tag" >> ../extra_terms/b2_campaign.log
    $PY run_oxides.py "$tag" >> "campaign_$tag.out" 2>&1 || echo "$(date -Is) FAILED $tag" >> ../extra_terms/b2_campaign.log
  done
done
echo "$(date -Is) ALL_DONE" >> ../extra_terms/b2_campaign.log
