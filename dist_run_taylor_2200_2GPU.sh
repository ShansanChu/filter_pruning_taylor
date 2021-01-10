export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 Taloy_pruner.py --model resnet50 --batch-size 128 --dataset imagenet --data-dir ./data --load-pretrained-model True --pretrained-model-dir ./checkpoints/resnet50.pth --pruner TaylorPruner --fine-tune-epochs 25
#python Taloy_pruner.py --model resnet50 --batch-size 128 --dataset imagenet --mgpu True --data-dir ./data --load-pretrained-model True --pretrained-model-dir ./checkpoints/resnet50.pth --pruner TaylorPruner --fine-tune-epochs 35
