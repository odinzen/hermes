# Sol checklist: Hermes polymorph test

You run every step; the agent never touches Sol. Copy-paste the commands and send back what each "send me" asks for.

What it does: the 13 cases of ../polymorph_heldout/RULE.md (frozen 2026-11-02) with MACE MPA-0, one A100 each.
A short check job first compares MPA-0's forces on Sol with those on the local GPU; the runs start only if it passes.
Hermes is not on Sol, and the scoring runs locally.

The package folder is replaced whole on every upload. Results and logs live in
/scratch/<user>/hermes-polymorph-sol-results, beside it, so a replace never touches finished runs. It uses the hermes-mace
environment already built for the earlier packages, so setup takes a minute.

## Step 1. Upload the package

Upload the folder `Downloads/hermes-polymorph-sol` to /scratch/<user>/.

## Step 2. Set up and submit (Sol login node, one paste)

```
cd /scratch/$USER/hermes-polymorph-sol && dos2unix *.sh *.slurm cases.txt 2>/dev/null; bash setup_env.sh && bash submit.sh && squeue -u $USER
```

It ends with `SETUP OK`, two `Submitted` lines and the queue. Send me a screenshot of the end.

## Step 3. The check (soon after it starts)

```
cat /scratch/$USER/hermes-polymorph-sol-results/check.json
```

Send me that output. The last line should read `"verdict": "PASS"`. If it says FAIL, the runs never start; cancel
them with `scancel -u $USER -n hermes-polymorph` and send me the output.

## Step 4. Check that everything finished (about 9 hours later)

```
cd /scratch/$USER/hermes-polymorph-sol-results && grep -h "STATUS" logs/*.out | sort | uniq -c | sort -rn | tail -5; grep -l CASE_DONE */runs/*/progress.log | wc -l
```

Send me that output. Expect 13. A run that failed or ran out of time is resubmitted with `bash submit.sh` from
/scratch/$USER/hermes-polymorph-sol once `squeue -u $USER` is empty; it continues from its last finished stage.

## Step 5. Bring the results back

In the OnDemand file browser, download the whole folder /scratch/<user>/hermes-polymorph-sol-results to Downloads
(about 2 GB). Tell me when it is there.
