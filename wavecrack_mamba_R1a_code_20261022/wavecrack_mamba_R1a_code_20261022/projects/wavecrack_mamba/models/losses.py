"""WaveCrackMamba loss suite (Spec §9, M0-7).

All mask losses follow the ``BaseDecodeHead.loss_by_feat`` call convention
(pred logits ``(B,1,H,W)``, integer target ``(B,H,W)``, ``ignore_index``) so
they sit directly in ``loss_decode`` lists. Everything computes in fp32
regardless of AMP dtype (fp16 sigmoid underflow kills dice/tversky gradients --
the TANGO incident).

Registered names all carry the ``WCM`` prefix: re-homed losses deliberately do
NOT reuse the mixercseg/tangoseg class names, so importing several projects in
one process (e.g. the repo test suite) can never double-register.

The ``ignore=255`` semantics follow Spec §8.0/§9: ignore pixels join no loss;
for topology, samples whose ignore region touches GT foreground are skipped
entirely until a verified ROI/guard-band implementation exists (§9.2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


def _prep(pred, target, ignore):
    """logits (B,1,H,W) -> fp32 prob (B,H,W); target -> fg/valid floats."""
    assert pred.shape[1] == 1, 'binary sigmoid head expected (out_channels=1)'
    prob = pred.float().sigmoid().squeeze(1)
    valid = (target != ignore).float()
    fg = (target == 1).float() * valid
    return prob, fg, valid


@MODELS.register_module()
class WCMFocalLoss(nn.Module):
    """Binary sigmoid focal loss, valid-pixel mean (Spec §9.1).

    ``alpha_fg=0.75 / gamma=2`` deliberately push recall for faint thin cracks;
    early low precision under this setting is EXPECTED behaviour (§9.1 note),
    to be judged via PR curves, not architecture changes.
    """

    def __init__(self, alpha_fg=0.75, alpha_bg=0.25, gamma=2.0,
                 loss_weight=1.0, ignore_index=255, loss_name='loss_focal'):
        super().__init__()
        self.alpha_fg, self.alpha_bg, self.gamma = alpha_fg, alpha_bg, gamma
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, ignore_index=None, **kwargs):
        ignore = self.ignore_index if ignore_index is None else ignore_index
        logits = pred.float().squeeze(1)
        valid = (target != ignore).float()
        fg = (target == 1).float() * valid
        ce = F.binary_cross_entropy_with_logits(
            logits, fg, reduction='none')
        p = logits.sigmoid()
        p_t = p * fg + (1 - p) * (1 - fg)
        alpha_t = self.alpha_fg * fg + self.alpha_bg * (1 - fg)
        focal = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return self.loss_weight * (focal * valid).sum() / \
            valid.sum().clamp(min=1)

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class WCMTverskyLoss(nn.Module):
    """Binary sigmoid Tversky, per-image then batch mean (Spec §9.1).

    Not mmseg's ``TverskyLoss``: that one one-hot-expands channels and has the
    same background-channel alignment hazard as ``DiceLoss`` for
    ``out_channels=1`` heads. FP/FN terms are restricted to valid pixels via
    explicit fg/bg masks. Empty-crack samples still contribute (their FP term
    constrains false positives, §9.1).
    """

    def __init__(self, fp_weight=0.3, fn_weight=0.7, eps=1e-6,
                 loss_weight=1.0, ignore_index=255, loss_name='loss_tversky'):
        super().__init__()
        self.fp_weight, self.fn_weight, self.eps = fp_weight, fn_weight, eps
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, ignore_index=None, **kwargs):
        ignore = self.ignore_index if ignore_index is None else ignore_index
        prob, fg, valid = _prep(pred, target, ignore)
        bg = (target == 0).float() * valid
        tp = (prob * fg).flatten(1).sum(1)
        fp = (prob * bg).flatten(1).sum(1)
        fn = ((1 - prob) * fg).flatten(1).sum(1)
        tversky = (tp + self.eps) / (
            tp + self.fp_weight * fp + self.fn_weight * fn + self.eps)
        return self.loss_weight * (1 - tversky).mean()

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class WCMRegionLoss(nn.Module):
    """``L_region = 0.5·focal + 0.5·tversky`` (Spec §9.1) as ONE module so the
    0.5/0.5 combination is registered in a single config field."""

    def __init__(self, focal=dict(), tversky=dict(), focal_weight=0.5,
                 tversky_weight=0.5, loss_weight=1.0, ignore_index=255,
                 loss_name='loss_region'):
        super().__init__()
        self.focal = WCMFocalLoss(ignore_index=ignore_index, **focal)
        self.tversky = WCMTverskyLoss(ignore_index=ignore_index, **tversky)
        self.focal_weight, self.tversky_weight = focal_weight, tversky_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * (
            self.focal_weight * self.focal(pred, target, **kwargs) +
            self.tversky_weight * self.tversky(pred, target, **kwargs))

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class WCMBinaryDiceLoss(nn.Module):
    """Binary sigmoid dice (re-homed from tangoseg/mixercseg, same formula,
    incl. the fp32 fix). mmseg's ``DiceLoss`` maps a 1-channel prediction onto
    the BACKGROUND one-hot channel and is inverted for sigmoid heads."""

    def __init__(self, smooth=1.0, loss_weight=1.0, ignore_index=255,
                 loss_name='loss_dice'):
        super().__init__()
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, ignore_index=None, **kwargs):
        ignore = self.ignore_index if ignore_index is None else ignore_index
        prob, fg, valid = _prep(pred, target, ignore)
        prob = prob * valid
        inter = (prob * fg).flatten(1).sum(1)
        denom = prob.flatten(1).sum(1) + fg.flatten(1).sum(1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return self.loss_weight * (1 - dice).mean()

    @property
    def loss_name(self):
        return self._loss_name


def _soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def _soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(x, iterations):
    """Differentiable morphological skeleton (clDice, Shit et al. CVPR'21)."""
    skel = F.relu(x - _soft_open(x))
    for _ in range(iterations):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


@MODELS.register_module()
class WCMSoftClDiceLoss(nn.Module):
    """Mask soft-clDice applied DIRECTLY to mask logits (Spec §9.2).

    - ``k=64`` at full resolution matches the max internal-radius scale; the
      64 serial morphology iterations @512² will likely blow the step-time
      budget, so the half-resolution variant ships NOW (``half_res=True``,
      ``half_res_k=32``) and is reported as an independent variant, never a
      silent replacement (§9.2). Down-sampling uses ``area`` for both prob and
      target (待澄清 #7 default, fixed by unit test).
    - ``V`` (topology-valid): ignore is never background. §9.2 skips samples
      whose ignore region touches GT foreground -- but in canonical
      ``{0,1,255}`` labels a pixel is either fg OR ignore, so a direct overlap
      test is vacuously empty. The decidable conservative form used here is
      ADJACENCY: if the dilated ignore region touches visible foreground, the
      crack likely continues into the ignore area (its skeleton would be
      truncated), so the sample is skipped until a verified ROI/guard-band
      exists. The test is evaluated at full resolution before any
      down-sampling.
    - Only samples with valid positive pixels enter the mean; empty samples are
      region-loss territory (§9.2).
    """

    def __init__(self, k=64, half_res=False, half_res_k=32, eps=1e-6,
                 loss_weight=1.0, ignore_index=255, loss_name='loss_cldice'):
        super().__init__()
        self.k = k
        self.half_res = half_res
        self.half_res_k = half_res_k
        self.eps = eps
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, ignore_index=None, **kwargs):
        ignore = self.ignore_index if ignore_index is None else ignore_index
        assert pred.shape[1] == 1
        prob = pred.float().sigmoid()
        valid = (target != ignore).float().unsqueeze(1)
        fg = (target == 1).float().unsqueeze(1) * valid
        ignore_mask = 1 - valid

        def _sum(x):
            return x.flatten(1).sum(1)

        # Per-sample gating at FULL resolution (see docstring: adjacency test).
        has_fg = _sum(fg) > 0
        ignore_touches_fg = _sum(_soft_dilate(ignore_mask) * fg) > 0
        active = (has_fg & ~ignore_touches_fg).float()

        if self.half_res:
            prob = F.avg_pool2d(prob, 2)      # 'area' downsampling
            fg = F.avg_pool2d(fg, 2)
            valid = F.avg_pool2d(valid, 2)
            k = self.half_res_k
        else:
            k = self.k

        pm = prob * valid
        y = fg
        skel_p = soft_skeleton(pm, k)
        skel_y = soft_skeleton(y, k)

        tprec = (_sum(valid * skel_p * y) + self.eps) / (
            _sum(valid * skel_p) + self.eps)
        tsens = (_sum(valid * skel_y * pm) + self.eps) / (
            _sum(valid * skel_y) + self.eps)
        cldice = 2 * tprec * tsens / (tprec + tsens + self.eps)

        loss = ((1 - cldice) * active).sum() / active.sum().clamp(min=1)
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class WCMSkeletonLoss(nn.Module):
    """Conditional skeleton-head loss: ``0.5·focal + 0.5·dice`` against the
    offline skeleton target (Spec §9.3). Focal params match the region loss.
    Not enabled by default (skeleton head is a conditional candidate)."""

    def __init__(self, loss_weight=1.0, ignore_index=255,
                 loss_name='loss_skeleton'):
        super().__init__()
        self.focal = WCMFocalLoss(ignore_index=ignore_index)
        self.dice = WCMBinaryDiceLoss(ignore_index=ignore_index)
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * (0.5 * self.focal(pred, target, **kwargs) +
                                   0.5 * self.dice(pred, target, **kwargs))

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class WCMContainmentLoss(nn.Module):
    """Cross-head containment ``Σ V·P_s·(1-P_m) / (Σ V·P_s + ε)`` (Spec §9.3).

    Only pushes skeleton predictions to stay inside the mask; mask topology is
    still owned by :class:`WCMSoftClDiceLoss`. Named cross-head, never a clDice
    substitute (§9.2 last paragraph). Conditional, off by default; called by
    the head with BOTH logits (not part of ``loss_decode`` single-pred lists).
    """

    def __init__(self, eps=1e-6, loss_weight=1.0, ignore_index=255,
                 loss_name='loss_contain'):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, skeleton_logits, mask_logits, target, ignore_index=None,
                **kwargs):
        ignore = self.ignore_index if ignore_index is None else ignore_index
        ps = skeleton_logits.float().sigmoid().squeeze(1)
        pm = mask_logits.float().sigmoid().squeeze(1)
        valid = (target != ignore).float()
        num = (valid * ps * (1 - pm)).flatten(1).sum(1)
        den = (valid * ps).flatten(1).sum(1) + self.eps
        return self.loss_weight * (num / den).mean()

    @property
    def loss_name(self):
        return self._loss_name


def decoder_grad_norms(losses, pred, target, shared_params, **loss_kwargs):
    """§9.6 diagnostic: per-loss gradient norm on shared decoder params.

    Runs one backward per loss (diagnostic cost -- call sparingly, e.g. from a
    periodic hook or a notebook, never every iter). Returns
    ``{loss_name: grad_l2_norm}`` so任务主导 (one task dominating the shared
    decoder) is checkable as Spec §9.6 requires.
    """
    norms = {}
    params = [p for p in shared_params if p.requires_grad]
    for loss_mod in losses:
        for p in params:
            p.grad = None
        value = loss_mod(pred, target, **loss_kwargs)
        grads = torch.autograd.grad(
            value, params, retain_graph=True, allow_unused=True)
        total = torch.zeros((), dtype=torch.float32)
        for g in grads:
            if g is not None:
                total = total + g.float().pow(2).sum().cpu()
        norms[loss_mod.loss_name] = float(total.sqrt())
    return norms
