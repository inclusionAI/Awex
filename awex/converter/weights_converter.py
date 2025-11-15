from awex import logging
import torch

logger = logging.getLogger(__name__)


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def per_block_cast_to_fp8(x: torch.Tensor, scale_ue8m0: bool):
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_scale = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4) / 448.0
    if scale_ue8m0:
        x_scale = (
            x_scale.maximum(torch.tensor(1e-10, device=x.device)).log2().ceil().exp2()
        )
    x_scaled = (x_view / x_scale).to(torch.float8_e4m3fn)
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        x_scale.view(x_view.size(0), x_view.size(2)),
    )
