# casCLIP
```
python -m torch.distributed.launch --nnodes 1 --nproc_per_node=8 train_net_cxr.py \
    --config-file configs/pretrain/glip_Swin_T_mimiccxr.yaml --data-path /data/Mao/DATASET/MIMIC-CXR/2.0.0/ \
    --initial-path /data/Mao/GLIP/  \
    --skip-test --use-tensorboard --override_output_dir output
```