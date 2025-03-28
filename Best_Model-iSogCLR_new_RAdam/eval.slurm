#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --output=./job_output_%x.%j
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu

export http_proxy='http://10.73.132.63:8080'
export https_proxy='http://10.73.132.63:8080'

#First Executable Line
source ~/.bashrc
source clip_project_env/bin/activate

export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

data_path=./datasets
ann_path=./clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
gamma=0.8
epochs=30
optimizer=radam
ita_type=isogclr_new

CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
    --data_path ${data_path} \
    --ann_path ${ann_path} \
    --train_file ${train_file} \
    --train_image_root ${train_image_root} \
    --output_dir output/eval_${ita_type}_${optimizer}_${data}_g${gamma}_e${epochs} \
    --init_model \
    --use_amp \
    --ita_type ${ita_type} \
    --opt ${optimizer} \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --tau_init 0.01 \
    --sogclr_gamma ${gamma} \
    --eta_init 0.03 --sched cosine \
    --no-distributed \
    --epochs ${epochs} \
    --evaluate --checkpoint ./output/${ita_type}_${optimizer}_cc3m_g0.8_e30/checkpoint_30.pth \
    --zs_dataset imagenet --zs_datafolder ./datasets/imagenet/val
