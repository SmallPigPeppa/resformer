#python eval_resformer.py \
#  --max_epochs 1 \
#  --accelerator gpu \
#  --devices 2 \
#  --works 4 \
#  --batch_size 512 \
#  --root /root/autodl-tmp/torch_ds/imagenet \
#  --model.weights deit_base_distilled_patch16_224.fb_in1k \
#  --model.num_classes 1000 \
#  --model.patch_size 16 \
#  --model.image_size 224 \
#  --model.resize_type interpolate \
#  --model.results_path ./L2P_exp/resformer.csv

#python eval_resformer_best.py \
#  --max_epochs 1 \
#  --accelerator gpu \
#  --devices 1 \
#  --works 4 \
#  --batch_size 512 \
#  --root /ppio_net0/torch_ds/imagenet \
#  --model.weights deit_base_distilled_patch16_224.fb_in1k \
#  --model.ckpt_path out-distill/best_checkpoint.pth \
#  --model.num_classes 1000 \
#  --model.patch_size 16 \
#  --model.image_size 224 \
#  --model.resize_type interpolate \
#  --model.results_path ./L2P_exp/resformer_distill_best.csv

python eval_resformer_best.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 1 \
  --works 4 \
  --batch_size 512 \
  --root /ppio_net0/torch_ds/imagenet \
  --model.weights deit_base_distilled_patch16_224.fb_in1k \
  --model.ckpt_path out/best_checkpoint.pth \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.resize_type interpolate \
  --model.results_path ./L2P_exp/resformer_best.csv
