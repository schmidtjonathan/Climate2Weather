if [ -z "$REPO_ROOT_DIR" ]; then
  echo "REPO_ROOT_DIR is not set. Please set REPO_ROOT_DIR to the root directory of the repository."
  exit 1
fi
if [ -z "$DATA_ROOT_DIR" ]; then
  echo "DATA_ROOT_DIR is not set. Please set DATA_ROOT_DIR to the root directory of the data."
  exit 1
fi

# -----------------------

echo "Begin"
timestampstr=$(date +"%Y%m%d%H%M%S")
echo "Timestamp: $timestampstr"


repo_root_dir=$REPO_ROOT_DIR
data_root_dir=$DATA_ROOT_DIR
datapath="$data_root_dir/COSMO_PATCH/COSMO_PATCH_2006-2019/train2006-2013/data/normalized_h5/train_norm-quant95.h5"

# create directory $repo_root_dir/runs if it does not exist, yet
if [ ! -d "$repo_root_dir/runs" ]; then
  echo "Training run is saved to $repo_root_dir/runs."
  echo "Creating directory $repo_root_dir/runs"
  mkdir -p "$repo_root_dir/runs"
else
  echo "Training run is saved to $repo_root_dir/runs."
fi

/usr/bin/python -u $repo_root_dir/train.py \
  --run-id $SLURM_JOB_ID \
  --run-dir $repo_root_dir/runs \
  --accelerator cuda \
  --devices $SLURM_NTASKS_PER_NODE \
  --num-nodes $SLURM_NNODES \
  --strategy ddp \
  --train-data $datapath \
  --total-ndata 180Mi \
  --spatial-res 128 \
  --num-features 4 \
  --markov-order 6 \
  --cache-data \
  --batch 512 \
  --lr 0.0001 \
  --batch-gpu 128 \
  --seed 42 \
  --log-firstdevice

