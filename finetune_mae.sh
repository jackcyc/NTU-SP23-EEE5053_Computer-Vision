# https://github.com/facebookresearch/mae
cd ./mae/
wget -nc https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth

# dataset: bicubic; no-rotation; bright0.2; ft12; ViT-L; train on all
# training 
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py --accum_iter 2 --batch_size 128 --model vit_large_patch16 --finetune './mae_pretrain_vit_large.pth' --epochs 50 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --mixup 0 --cutmix 0 --reprob 0 --dist_eval --nb_classes 2 --data_path '/mnt/191/a/ycc/CV_Final/data/trainset' --output_dir output_dir  --log_dir output_dir

# # inference
# python inference.py