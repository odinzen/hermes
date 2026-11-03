# Sol checklist: Hermes validation MD

You run every step; the agent never touches Sol. Copy-paste the commands and send back what each "send me" asks for.

What it does: the molecular dynamics for three pre-registered Hermes tests (closed-shell oxides, magnetic oxides,
lanthanide phosphates), 21 cases with 200 to 640 atoms, one A100 each. A short check job runs first and compares
Sol's forces with the ones computed on the local GPU; the cases start only if it passes. Hermes itself is not on Sol:
the trajectories come back and every entropy is computed locally.

The package folder is replaced whole on every upload. Results and logs live in /scratch/<user>/hermes-md-sol-results,
beside it, so a replace never touches finished runs, and a finished case is never submitted again.

## Step 1. Upload the package

Upload the folder `Downloads/hermes-md-sol` to /scratch/<user>/.

## Step 2. Set up and submit (Sol login node, one paste)

```
cd /scratch/$USER/hermes-md-sol && dos2unix *.sh *.slurm cases.txt 2>/dev/null; bash setup_env.sh && bash submit.sh && squeue -u $USER
```

The first setup builds the hermes-mace environment (about 10 minutes) and ends with `SETUP OK`. Then two jobs are
submitted: the check job and the case array, which waits for the check. Send me a screenshot of the end.

## Step 3. The check (about 15 minutes after it starts)

```
cat /scratch/$USER/hermes-md-sol-results/check.json
```

Send me that output. `"verdict": "PASS"` means the cases are running. If it says FAIL, the cases never start; cancel
them with `scancel -u $USER -n hermes-md` and send me the output.

## Step 4. Check that everything finished (about a day later)

```
cd /scratch/$USER/hermes-md-sol-results && grep -h "STATUS" logs/*.out | sort | uniq -c | sort -rn; grep -l CASE_DONE runs/*/progress.log | wc -l
```

Send me that output. Expect 21. A case that failed or ran out of time is resubmitted with `bash submit.sh` from
/scratch/$USER/hermes-md-sol once `squeue -u $USER` is empty; it continues from its last finished stage.

## Step 5. Bring the results back

In the OnDemand file browser, open /scratch/<user>/hermes-md-sol-results and download the folders `runs` and `logs`
and the file `check.json` to Downloads (about 3 GB). Tell me when they are there.
