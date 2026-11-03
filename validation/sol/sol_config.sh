# Sol settings for the Hermes validation MD. Every script sources this file; nothing else needs editing.
export ACCOUNT="grp_klilova"        # group allocation
export PARTITION="public"            # post-May-2026 Sol: public holds the publicly available nodes, GPUs included
export QOS="public"
export GPU_GRES="gpu:a100:1"         # one A100 per case
export CPUS=8
export MEM="32G"
export TIME="24:00:00"               # a case is ~420 ps of MD on 200 to 640 atoms; stages resume if a job is cut off
export CHECK_TIME="00:30:00"

export MAMBA_MODULE="mamba/latest"
export ENV_NAME="hermes-mace"
