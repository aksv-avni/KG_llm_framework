#!/bin/bash
#SBATCH --job-name=test01        # Job name
#SBATCH --partition=dgx     # small, medium, fat, test, dgx
#SBATCH --ntasks=1               # Run a single task
#SBATCH --mem=36G                # Added memory requirement
#SBATCH --gres=gpu:1             # Request for GPU
#SBATCH --time=2-00:00:00        # time limit of 2 days. Format: D-HH:MM:SS
#SBATCH --cpus-per-task=6        # Number of CPU cores per task
#SBATCH --output=log_%j.out      # Error log output file

echo "============================== Job Started ✅ =============================================="

#Activating anaconda3
echo "Module loading........"
module purge
module load anaconda3/2024
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh

echo "==================== Activating conda environment........✅ ==============================="
source activate "/iitjhome/r24ab0001/.conda/envs/ablation" || true
AB_PYTHON=/iitjhome/r24ab0001/.conda/envs/ablation/bin/python
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Python interpreter set to: $AB_PYTHON"
echo "PYTHONPATH: $PYTHONPATH"

echo "========== Starting GPU monitor (by nvidia-smi) in background for 25 min ✅....... =========="
NUM_PING=50
WAIT=30
nohup $AB_PYTHON /iitjhome/r24ab0001/gpu_status.py $NUM_PING $WAIT \
    > gpu_log_${SLURM_JOB_ID}.out &

# authenticate to HF and set caches
hf auth login --token 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

export HF_HOME=/scratch/data/r24ab0001/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_OFFLOAD_FOLDER=/scratch/data/r24ab0001/hf_offload
mkdir -p $TRANSFORMERS_CACHE $HF_OFFLOAD_FOLDER

# run the knowledge‑graph + LLM demo script
PYTHONUNBUFFERED=1 $AB_PYTHON -u "/iitjhome/r24ab0001/kg_llm_demo/kg_llm_demo_llama.py"

# previous inline MedCLIP smoke‑test commented out
# $AB_PYTHON - <<'PYTHON'
# ...
# PYTHON

echo "============================== Job Finished ✅ =============================================="
echo "💿Free space in /scratch/data/r24ab0001:"
du -sh /scratch/data/r24ab0001
echo "💿Free space in /iitjhome/r24ab0001:"
du -sh /iitjhome/r24ab0001/