"""Optional, prediction-conditioned shared refinement of a fixed feature map.

The image memory is encoded once per forward, never cached across forwards.
The classifier is borrowed at call time, so its parameters are not registered
twice. Zero-initialized residual outputs preserve the existing decoder.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


def _partition(x, window):
    b, h, w, c = x.shape
    return (x.reshape(b, h // window, window, w // window, window,
                      c).permute(0, 1, 3, 2, 4,
                                 5).reshape(-1, window * window, c))


def _unpartition(x, batch, height, width, window):
    return (x.reshape(batch, height // window, width // window, window, window,
                      -1).permute(0, 1, 3, 2, 4,
                                  5).reshape(batch, height, width, -1))


class _Updater(nn.Module):

    def __init__(self, channels, hidden, heads, window, mode, ffn_ratio):
        super().__init__()
        self.mode, self.heads, self.window = mode, heads, window
        self.norm = nn.LayerNorm(channels)
        self.mask_embed = nn.Linear(1, channels, bias=False)
        self.query = nn.Linear(channels, hidden)
        self.output = nn.Linear(hidden, channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_ratio), nn.GELU(),
            nn.Linear(channels * ffn_ratio, channels))
        if mode == 'conv':
            self.spatial = nn.Conv2d(
                hidden, hidden, 7, padding=3, groups=hidden)
        else:
            self.relative_bias = nn.Parameter(
                torch.zeros((2 * window - 1)**2, heads))
            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(window), torch.arange(window),
                    indexing='ij')).flatten(1)
            delta = coords[:, :, None] - coords[:, None, :]
            index = ((delta[0] + window - 1) * (2 * window - 1) + delta[1] +
                     window - 1)
            self.register_buffer('relative_index', index, persistent=False)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def _attend(self, query, key, value, shift):
        b, h, w, c = query.shape
        win = self.window
        ph, pw = (-h) % win, (-w) % win
        hp, wp = h + ph, w + pw

        # Pad projected memory, then roll all three tensors consistently.
        def windows(x):
            x = F.pad(x, (0, 0, 0, pw, 0, ph))
            x = torch.roll(x, (-shift, -shift), (1, 2))
            return _partition(x, win)

        q, k, v = (windows(x) for x in (query, key, value))
        tokens, dim = win * win, c // self.heads
        q, k, v = (
            x.reshape(-1, tokens, self.heads, dim).transpose(1, 2)
            for x in (q, k, v))
        # In rolled coordinates, tags distinguish pixels crossing the cyclic
        # image seam. Ordinary neighbouring windows may exchange information.
        rows = torch.arange(hp, device=query.device)
        cols = torch.arange(wp, device=query.device)
        regions = ((rows + shift >= hp)[:, None].long() * 2 +
                   (cols + shift >= wp)[None, :].long())
        tags = _partition(regions[None, :, :, None], win).squeeze(-1)
        allowed = tags[:, :, None] == tags[:, None, :]
        valid = ((rows < h)[:, None] & (cols < w)[None, :])
        valid = torch.roll(valid, (-shift, -shift), (0, 1))
        valid = _partition(valid[None, :, :, None], win).squeeze(-1)
        allowed = allowed & valid[:, None, :]
        # Padded queries can have no valid keys. Give only those queries a
        # harmless diagonal fallback; their outputs are cropped before reuse.
        eye = torch.eye(tokens, device=query.device, dtype=torch.bool)
        allowed = allowed | (~allowed.any(-1, keepdim=True) & eye)
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)
        mask = bias[None].to(q.dtype).expand(allowed.shape[0], -1, -1, -1)
        mask = mask.masked_fill(~allowed[:, None], float('-inf'))
        mask = mask.repeat(b, 1, 1, 1)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(-1, tokens, c)
        attended = _unpartition(attended, b, hp, wp, win)
        return torch.roll(attended, (shift, shift), (1, 2))[:, :h, :w]

    def forward(self, state, mask, memory, shift, scale):
        q = self.query(self.norm(state) + self.mask_embed(mask))
        if self.mode == 'conv':
            value = memory[0]
            update = self.spatial((q + value).permute(0, 3, 1, 2))
            update = F.gelu(update.permute(0, 2, 3, 1))
        else:
            update = self._attend(q, *memory, shift)
        state = state + scale * self.output(update)
        return state + scale * self.ffn(self.ffn_norm(state))


@MODELS.register_module()
class WCMRecurrentRefiner(nn.Module):
    """Refine NCHW features without changing their shape or the old readout.

    ``steps=0`` returns the exact input and skips memory/classifier work.
    ``return_intermediates`` returns (features, coarse_logits), including the
    initial logits. Intermediate outputs are diagnostics, not extra losses.
    Steps cannot exceed the configured schedule; use a separate config for a
    different training depth. Conv mode uses no cyclic window shift.
    """

    def __init__(self,
                 channels=128,
                 steps=3,
                 mode='transformer',
                 heads=4,
                 window_size=8,
                 shifts=None,
                 share_weights=True,
                 use_mask_feedback=True,
                 conv_channels=160,
                 ffn_ratio=2,
                 residual_scale=0.1):
        super().__init__()
        for name, number in [('channels', channels), ('steps', steps),
                             ('heads', heads), ('window_size', window_size),
                             ('conv_channels', conv_channels),
                             ('ffn_ratio', ffn_ratio)]:
            if isinstance(number,
                          bool) or not isinstance(number, int) or number < 1:
                raise ValueError(f'{name} must be a positive integer')
        if mode not in ('transformer', 'conv'):
            raise ValueError('mode must be transformer or conv')
        if channels % heads:
            raise ValueError('channels must be divisible by heads')
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError('residual_scale must be positive and finite')
        if shifts is None:
            shifts = tuple((i % 2) * (window_size // 2) for i in range(steps))
        if len(shifts) != steps or any(
                isinstance(s, bool) or not isinstance(s, int)
                or not 0 <= s < window_size for s in shifts):
            raise ValueError(
                'shifts must have one valid window shift per step')
        self.channels, self.steps, self.mode = channels, steps, mode
        self.shifts = tuple(shifts)
        self.share_weights = share_weights
        self.use_mask_feedback = use_mask_feedback
        self.residual_scale = residual_scale
        self.memory_norm = nn.LayerNorm(channels)
        hidden = channels if mode == 'transformer' else conv_channels
        self.value = nn.Linear(channels, hidden)
        self.key = nn.Linear(channels,
                             channels) if mode == 'transformer' else None
        first = _Updater(channels, hidden, heads, window_size, mode, ffn_ratio)
        self.updaters = nn.ModuleList(
            [first] if share_weights else [first] +
            [copy.deepcopy(first) for _ in range(steps - 1)])

    def forward(self,
                features,
                mask_classifier,
                steps=None,
                return_intermediates=False):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                'features must be NCHW with the configured channels')
        if min(features.shape[0], *features.shape[2:]) < 1:
            raise ValueError('empty features are not supported')
        steps = self.steps if steps is None else steps
        if isinstance(steps, bool) or not isinstance(
                steps, int) or not 0 <= steps <= self.steps:
            raise ValueError(
                'steps must be an integer within the configured schedule')

        def classify(x):
            logits = mask_classifier(x)
            if logits.shape != (features.shape[0], 1, *features.shape[2:]):
                raise ValueError(
                    'mask_classifier must return one logit per feature pixel')
            return logits

        logits = [classify(features)] if return_intermediates else []
        if steps == 0:
            return (features,
                    tuple(logits)) if return_intermediates else features
        state = features.permute(0, 2, 3, 1)
        encoded = self.memory_norm(state)
        value = self.value(encoded)
        memory = (self.key(encoded),
                  value) if self.key is not None else (value, )
        for t in range(steps):
            if self.use_mask_feedback:
                mask = classify(state.permute(0, 3, 1, 2)).sigmoid()
                mask = (2 * mask - 1).permute(0, 2, 3, 1)
            else:
                mask = state.new_zeros(*state.shape[:3], 1)
            updater = self.updaters[0 if self.share_weights else t]
            state = updater(state, mask, memory, self.shifts[t],
                            self.residual_scale)
            if return_intermediates:
                logits.append(classify(state.permute(0, 3, 1, 2)))
        result = state.permute(0, 3, 1, 2).contiguous()
        return (result, tuple(logits)) if return_intermediates else result
