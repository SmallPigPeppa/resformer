YOUR_DATA_PATH=/ppio_net0/torch_ds/imagenet
YOUR_OUTPUT_PATH=/ppio_net0/code/resformer/image_classification/out-deit3b
python -m torch.distributed.launch --nproc_per_node 8 main.py --data-path ${YOUR_DATA_PATH} --model resformer_base_patch16_deit3b --output_dir ${YOUR_OUTPUT_PATH} --batch-size 64 --pin-mem --input-size 224 160 128 --auto-resume --distillation-type 'smooth-l1' --distillation-target cls --sep-aug --epochs 200 --drop-path 0.2 --lr 8e-4 --warmup-epochs 20 --clip-grad 5.0 --epochs 200 --cooldown-epochs 0
python /ppio_net0/code/openapi.py stop 96c9cd8dadee9f28