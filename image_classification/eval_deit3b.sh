
# deit3b
python eval_resformer_best_deit3b.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 1 \
  --works 8 \
  --batch_size 512 \
  --root /ppio_net0/torch_ds/imagenet \
  --model.weights deit3_base_patch16_224.fb_in22k_ft_in1k \
  --model.ckpt_path out-deit3b/best_checkpoint.pth \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.resize_type interpolate \
  --model.results_path ./L2P_exp/resformer_deit3b.csv
#
## vitb
#python eval_resformer_best.py \
#  --max_epochs 1 \
#  --accelerator gpu \
#  --devices 1 \
#  --works 8 \
#  --batch_size 512 \
#  --root /ppio_net0/torch_ds/imagenet \
#  --model.weights vit_base_patch16_224.augreg_in21k_ft_in1k \
#  --model.ckpt_path out-vitb-21k/best_checkpoint.pth \
#  --model.num_classes 1000 \
#  --model.patch_size 16 \
#  --model.image_size 224 \
#  --model.resize_type interpolate \
#  --model.results_path ./L2P_exp/resformer_vitb_21k.csv
