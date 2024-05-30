python -m torch.distributed.launch --nnodes 1 --nproc_per_node=8 train_net_cxr.py \
    --config-file configs/pretrain/casclip_cxr_hr.yaml  \
    --data-path /data/Mao/DATASET/MIMIC-CXR/2.0.0/ \
    --initial-path /data/Mao/GLIP/  \
    --skip-test --use-tensorboard --override_output_dir output_casclip_cxr_hr

python -m torch.distributed.launch --nnodes 1 --nproc_per_node=8 train_net_cxr.py \
    --config-file configs/pretrain/casclip_cxr_hr_pnu.yaml  \
    --data-path /data/Mao/DATASET/MIMIC-CXR/2.0.0/ \
    --initial-path /data/Mao/GLIP/  \
    --skip-test --use-tensorboard --override_output_dir output_casclip_cxr_hr_pnu

python -m torch.distributed.launch --nnodes 1 --nproc_per_node=8 train_net_cxr.py \
    --config-file configs/pretrain/casclip_cxr.yaml  \
    --data-path /data/Mao/DATASET/MIMIC-CXR/2.0.0/ \
    --initial-path /data/Mao/GLIP/  \
    --skip-test --use-tensorboard --override_output_dir output_casclip_cxr

python -m torch.distributed.launch --nnodes 1 --nproc_per_node=8 train_net_cxr.py \
    --config-file configs/pretrain/casclip_cxr_pnu.yaml  \
    --data-path /data/Mao/DATASET/MIMIC-CXR/2.0.0/ \
    --initial-path /data/Mao/GLIP/  \
    --skip-test --use-tensorboard --override_output_dir output_casclip_cxr_pnu