import json
import torch
from torchao.quantization.granularity import PerTensor, PerAxis
from torchao.quantization.observer import AffineQuantizedMSEObserver
from torchao.quantization.quant_primitives import MappingType, _fake_quantize_affine, choose_qparams_affine_with_min_max
from torchao.quantization.utils import get_block_size

def mse_reference(observer, x, low, high):
    scale, zero = choose_qparams_affine_with_min_max(
        low, high, observer.mapping_type, [], observer.target_dtype,
        eps=observer.eps, scale_dtype=observer.scale_dtype,
        zero_point_dtype=observer.zero_point_dtype,
    )
    y = _fake_quantize_affine(x, get_block_size(x.shape, observer.granularity),
                             scale, zero, observer.target_dtype)
    axes = tuple(range(x.ndim)) if isinstance(observer.granularity, PerTensor) else (1,)
    return (y.float() - x.float()).square().mean(dim=axes)

for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for magnitude in (0.004, 30000.0):
        for granularity in (PerTensor(), PerAxis(0)):
            x = torch.linspace(-magnitude, magnitude, 1024).repeat(4, 1).to(dtype)
            observer = AffineQuantizedMSEObserver(
                MappingType.SYMMETRIC, torch.int8, granularity,
                steps=64, eps=torch.finfo(torch.float32).eps,
                scale_dtype=torch.float32, zero_point_dtype=torch.int32,
            )
            observer(x)
            actual = mse_reference(observer, x, observer.min_val, observer.max_val)
            bound = x.abs().amax() if isinstance(granularity, PerTensor) else x.abs().amax(dim=1)
            losses = [mse_reference(observer, x, -bound / 64 * i, bound / 64 * i) for i in range(1, 65)]
            best = torch.stack(losses).amin(dim=0)
            print(json.dumps({"dtype":str(dtype),"magnitude":magnitude,"granularity":str(granularity),
                              "actual_error":actual.tolist(),"best_error":best.tolist(),
                              "ratio":(actual / best).tolist(),"selected_max":observer.max_val.tolist()}), flush=True)
