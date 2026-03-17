#!/bin/bash
#SBATCH --job-name=test01	# Job name
#SBATCH --partition=dgx	# small, medium, fat, test, dgx
#SBATCH --ntasks=1		# Run a single task
#SBATCH --mem=36G		# Added memory requirement
#SBATCH --gres=gpu:1		# Request for GPU
#SBATCH --time=2-00:00:00	# time limit of 2 days. Format: D-HH:MM:SS
#SBATCH --cpus-per-task=6	# Number of CPU cores per task
#SBATCH --output=log_%j.out	# Error log output file

echo "============================== Job Started ✅ =============================================="



#Activating anaconda3
echo "Module loading........"
module purge
module load anaconda3/2024
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh


echo "==================== Activating conda environment........✅ ==============================="

# try to activate environment; if that fails we'll still reference its python directly
source activate "/iitjhome/r24ab0001/envs/vlm" || true

# explicitly set interpreter path to avoid activation issues
VLM_PYTHON=/iitjhome/r24ab0001/envs/vlm/bin/python

echo "Python interpreter set to: $VLM_PYTHON"
echo "PYTHONPATH: $PYTHONPATH"

# load GROQ key from .env if present and export
if [ -f "/iitjhome/r24ab0001/KG_LLM_PE_diagnosis/.env" ]; then
    export $(grep -v '^#' /iitjhome/r24ab0001/KG_LLM_PE_diagnosis/.env | xargs)
    echo "Loaded GROQ_API_KEY from .env"
fi


echo "========== Starting GPU monitor (by nvidia-smi) in background for 25 min ✅....... =========="
NUM_PING=50	# The number of times to verify whether the GPU is being used
WAIT=30		# The interval between pings
nohup $VLM_PYTHON /iitjhome/r24ab0001/gpu_status.py $NUM_PING $WAIT > gpu_log_${SLURM_JOB_ID}.out & echo "========== Running training script...✅ =========="
# mpirun $VLM_PYTHON /iitjhome/r24ab0001/prog01/train.py

hf auth login --token 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
export MODEL_FLAVOR="groq"
# if using a very large HF model we may not have enough GPU RAM; load on CPU or cap GPU
export HF_MAX_GPU_MEM="40GB"    # adjust as needed, or leave empty for unconstrained
if [[ $MODEL_FLAVOR == meta-llama* ]]; then
    # force CPU only to bypass GPU OOM; GPU will still be allocated by Slurm but unused
    export CUDA_VISIBLE_DEVICES=""
    echo "NOTE: Llama model will run on CPU (CUDA_VISIBLE_DEVICES unset)"
fi


export HF_HOME=/scratch/data/r24ab0001/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
# optionally offload large models when using auto device_map
export HF_OFFLOAD_FOLDER=/scratch/data/r24ab0001/hf_offload
# redirect checkpoint file to scratch to avoid home quota issues
export CHECKPOINT_DIR=/scratch/data/r24ab0001/kg_checkpoints
mkdir -p $CHECKPOINT_DIR
# make sure the cache/offload directories exist and have plenty of space
mkdir -p $TRANSFORMERS_CACHE $HF_OFFLOAD_FOLDER

export UMLS_CACHE_DB=/scratch/data/r24ab0001/umls_cache.db



# run KG pipeline with unbuffered output
# PYTHONUNBUFFERED=1 $VLM_PYTHON -u "/iitjhome/r24ab0001/KG_LLM_PE_diagnosis/modules/kg_construction/trial_2_kg.py"
# next step : run_from_checkpoint.py to load the saved checkpoint and continue processing
PYTHONUNBUFFERED=1 $VLM_PYTHON -u "/iitjhome/r24ab0001/KG_LLM_PE_diagnosis/modules/kg_construction/run_from_checkpoint.py"


# $VLM_PYTHON "/iitjhome/r24ab0001/KG_LLM_PE_diagnosis/experiments/vlm_test_v1/1 Mar_vision_backbone_MEDVIT_IMPLEMENTATION_trial_1/vision_backbone_mvit_timm_full_implementation.py"
# $(which python) "/iitjhome/r24ab0001/KG_LLM_PE_diagnosis/experiments/vlm_test_v1/1 Mar_vision_backbone_MEDVIT_IMPLEMENTATION_trial_1/image_to_label_mapping.py"
# $(which python) "/iitjhome/r24ab0001/KG_LLM_PE_diagnosis/experiments/vlm_test_v1/1 Mar_vision_backbone_MEDVIT_IMPLEMENTATION_trial_1/build_balanced_data_100.py"

# $VLM_PYTHON "/iitjhome/r24ab0001/KG_LLM_PE_diagnosis/experiments/vlm_test_v1/1 Mar_vision_backbone_MEDVIT_IMPLEMENTATION_trial_1/build_balanced_data_100.py"

echo "============================== Job Finished ✅ =============================================="
echo "💿Free space in /scratch/data/r24ab0001:"
du -sh /scratch/data/r24ab0001
echo "💿Free space in /iitjhome/r24ab0001:"
du -sh /iitjhome/r24ab0001/
