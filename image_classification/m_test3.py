from timm.models import create_model
import models
import torch

checkpoint_path = 'resformer_base_patch16_mr_128_160_224.pth'
state_dict = torch.load(checkpoint_path)['model']

model = create_model(
    'resformer_base_patch16',
    img_size=[224],
    pretrained=False,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.2,
    drop_block_rate=None,
    use_checkpoint=False,
)

net = model
net.load_state_dict(state_dict, strict=True)