"""Mean encoding and causally aligned scaled writeback, including cached decoding."""
import torch
from torch import nn
from torch.nn import functional as F


class ScaledDecoder(nn.Module):
    """Two channel vectors w=1+delta; AdamW acts on zero-initialized delta.

    Retain the identity-plus-depthwise-convolution evaluation of the released
    experiments, including its rounding under BF16 autocast.
    """
    def __init__(self, width):
        super().__init__()
        self.delta_weight = nn.Parameter(torch.zeros(width, 2, 1))

    def forward(self, delta):
        batch, length, width = delta.shape
        correction = F.conv1d(delta.transpose(1, 2),
                              self.delta_weight.reshape(2 * width, 1, 1), groups=width)
        correction = correction.transpose(1, 2).reshape(batch, length, width, 2).transpose(2, 3)
        return delta.unsqueeze(2) + correction.to(delta.dtype)


def latent_correction(x, run_core, decoder, *, offset=0, pending=None, last_update=None):
    """Encode completed pairs; write their increments to token positions 2j+1/2j+2.

    Positions here are zero based: pair [0,1] updates positions [1,2], never [0,1].
    The first visit's RoPE positions are the completed pairs' final positions.
    """
    batch, length, width = x.shape
    if pending is not None:
        if pending.shape != (batch, offset % 2, width) or pending.device != x.device or pending.dtype != x.dtype:
            raise ValueError("pending pair changed; reset_cache() before a new sequence")
    elif offset % 2:
        raise ValueError("missing unfinished pair; reset_cache() first")
    if last_update is not None:
        if last_update.shape != (batch, 1, 2, width) or last_update.device != x.device or last_update.dtype != x.dtype:
            raise ValueError("writeback state changed; reset_cache() first")
    elif offset // 2:
        raise ValueError("missing previous update; reset_cache() first")
    values = x if pending is None else torch.cat((pending, x), dim=1)
    count, first_pair = values.size(1) // 2, offset // 2
    if count:
        z = values[:, :count * 2].reshape(batch, count, 2, width).mean(2)
        positions = torch.arange(first_pair, first_pair + count, device=x.device)
        updates = decoder(run_core(z, positions) - z)
    else:
        updates = x.new_empty(batch, 0, 2, width)
    previous = x.new_zeros(batch, 1, 2, width) if last_update is None else last_update
    available = torch.cat((previous, updates), dim=1)
    target = torch.arange(offset, offset + length, device=x.device)
    source = torch.div(target - 1, 2, rounding_mode="floor")
    slot = (target - 1).remainder(2)
    indices = (source - (first_pair - 1)) * 2 + slot
    corrected = x + available.flatten(1, 2).index_select(1, indices)
    leftover = values[:, count * 2:]
    return corrected, leftover.clone() if leftover.size(1) else None, updates[:, -1:].clone() if count else last_update
