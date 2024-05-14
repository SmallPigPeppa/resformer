from timm import create_model
from models import ResFormer
import torch
import torch.nn as nn
from functools import partial
from torchinfo import summary
from thop import profile

# 创建TIMM模型
model_name = 'deit_base_distilled_patch16_224'
timm_model = create_model(model_name, pretrained=True)

# 创建自定义模型
model = ResFormer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))


# 定义一个函数来计算模型的参数数量和FLOPs
def analyze_model(model, input_size=(1, 3, 224, 224)):
    # 为模型生成一个随机输入
    inputs = torch.randn(input_size)

    # 使用thop计算FLOPs和参数
    flops, params = profile(model, inputs=(inputs,), verbose=False)

    # 打印模型的详细信息
    # print(summary(model, input_size=input_size))

    # 返回计算结果
    return flops, params


# 分析TIMM模型
flops_timm, params_timm = analyze_model(timm_model)
print(f"TIMM model - FLOPs: {flops_timm}, Params: {params_timm}")

# 分析ResFormer模型
flops_res, params_res = analyze_model(model)
print(f"ResFormer model - FLOPs: {flops_res}, Params: {params_res}")
