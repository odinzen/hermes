#!/usr/bin/env bash
# Arm A vibrational MD with the oxide vibration route unchanged (run_oxides.py), queued after the oxide campaign
# so the two never share the GPU. Up to three passes, as a transient CUDA error can fail a case.
cd "$(dirname "$0")/../oxide_vibration"
while pgrep -f "bash run_campaign.sh" > /dev/null || pgrep -f "bash retry_campaign.sh" > /dev/null; do sleep 600; done
PY=~/scratch/mace-gpu/bin/python
for pass in 1 2 3; do
  for tag in nio mno coo fe2o3 cr2o3; do
    if grep -q CASE_DONE "runs/$tag/progress.log" 2>/dev/null; then continue; fi
    echo "$(date -Is) arm A pass $pass start $tag" >> ../extra_terms/arm_a_campaign.log
    $PY run_oxides.py "$tag" >> "campaign_$tag.out" 2>&1 || echo "$(date -Is) FAILED $tag" >> ../extra_terms/arm_a_campaign.log
  done
done
echo "$(date -Is) ALL_DONE" >> ../extra_terms/arm_a_campaign.log
