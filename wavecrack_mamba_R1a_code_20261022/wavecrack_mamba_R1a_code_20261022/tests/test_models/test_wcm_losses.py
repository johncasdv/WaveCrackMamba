"""M0-7 acceptance: loss suite (Spec §9) + warm-up hook.

    python -m pytest tests/test_models/test_wcm_losses.py -v
"""
from types import SimpleNamespace

import pytest
import torch
from mmengine.registry import init_default_scope

import projects.wavecrack_mamba  # noqa: F401
from projects.wavecrack_mamba.models.hooks import LossWeightWarmupHook
from projects.wavecrack_mamba.models.losses import (
    WCMBinaryDiceLoss, WCMContainmentLoss, WCMFocalLoss, WCMRegionLoss,
    WCMSkeletonLoss, WCMSoftClDiceLoss, WCMTverskyLoss)

init_default_scope('mmseg')

ALL_MASK_LOSSES = [WCMFocalLoss(), WCMTverskyLoss(), WCMRegionLoss(),
                   WCMBinaryDiceLoss(), WCMSoftClDiceLoss(k=8),
                   WCMSkeletonLoss()]


def _line_target(b=1, h=64, w=64):
    t = torch.zeros(b, h, w, dtype=torch.long)
    t[:, 30:33, 8:56] = 1
    return t


def test_gradient_direction_towards_gt_decreases_loss():
    # Nudge predictions toward GT -> every loss must go down.
    torch.manual_seed(0)
    target = _line_target()
    pred = torch.randn(1, 1, 64, 64) * 0.5
    towards = torch.where(target.unsqueeze(1) == 1, torch.tensor(1.0),
                          torch.tensor(-1.0))
    better = pred + 0.5 * towards
    for loss in ALL_MASK_LOSSES:
        assert loss(better, target) < loss(pred, target), loss.loss_name


def test_edge_case_masks_no_nan():
    # empty mask / 1px line / 64px wide region: finite loss AND finite grads.
    cases = {
        'empty': torch.zeros(1, 64, 64, dtype=torch.long),
        '1px': torch.zeros(1, 64, 64, dtype=torch.long),
        'wide': torch.zeros(1, 64, 64, dtype=torch.long),
    }
    cases['1px'][:, 32, :] = 1
    cases['wide'][:, :, :] = 1
    for name, target in cases.items():
        for loss in ALL_MASK_LOSSES:
            pred = torch.randn(1, 1, 64, 64, requires_grad=True)
            value = loss(pred, target)
            assert torch.isfinite(value), (name, loss.loss_name)
            value.backward()
            assert torch.isfinite(pred.grad).all(), (name, loss.loss_name)


def test_ignore_pixels_do_not_participate():
    # Perturbing predictions inside the ignore region must not change any loss.
    target = _line_target()
    target[:, :, 40:] = 255  # ignore right side (off the fg line for cldice)
    pred = torch.randn(1, 1, 64, 64)
    pert = pred.clone()
    pert[:, :, :, 40:] += 100.0
    for loss in ALL_MASK_LOSSES:
        torch.testing.assert_close(loss(pred, target), loss(pert, target)), \
            loss.loss_name


def test_fp16_inputs_keep_gradients_alive():
    # The TANGO fp16 underflow regression: half logits, strongly negative.
    target = _line_target()
    for loss in (WCMBinaryDiceLoss(), WCMTverskyLoss(), WCMSoftClDiceLoss(k=8)):
        pred = (torch.randn(1, 1, 64, 64) - 12.0).half().requires_grad_(True)
        loss(pred, target).backward()
        g = pred.grad
        assert torch.isfinite(g).all() and g.abs().sum() > 0, loss.loss_name


def _gap_line_setup(k=8, half=False):
    # A line with a gap; logits confidently predict the line WITH the gap.
    target = torch.zeros(1, 64, 64, dtype=torch.long)
    target[:, 30:33, 8:56] = 1                      # GT: continuous line
    pred = torch.full((1, 1, 64, 64), -6.0)
    pred[:, :, 30:33, 8:56] = 6.0
    pred[:, :, 30:33, 28:36] = -6.0                 # the break
    loss = WCMSoftClDiceLoss(k=k, half_res=half)
    return loss, pred, target


def test_cldice_full_and_half_res_agree_on_closing_the_gap():
    # Direction-consistency (§9.2 / plan M0-7): raising logits inside the gap
    # (closing the break) must LOWER soft-clDice in both variants.
    for half in (False, True):
        loss, pred, target = _gap_line_setup(half=half)
        base = loss(pred, target)
        closed = pred.clone()
        closed[:, :, 30:33, 28:36] = 6.0
        assert loss(closed, target) < base, f'half_res={half}'


def test_cldice_gradient_reaches_mask_logits():
    loss, pred, target = _gap_line_setup()
    pred.requires_grad_(True)
    loss(pred, target).backward()
    assert pred.grad.abs().sum() > 0
    # the gap region specifically wants higher logits (negative gradient)
    assert pred.grad[:, :, 30:33, 28:36].sum() < 0


def test_cldice_empty_and_ignore_overlap_samples_are_skipped():
    # batch = [normal, empty, ignore-overlaps-fg]; only the normal sample may
    # contribute: batch loss == single-sample loss.
    loss = WCMSoftClDiceLoss(k=8)
    normal = _line_target()
    empty = torch.zeros_like(normal)
    overlap = _line_target()
    overlap[:, 28:35, 20:40] = 255          # ignore straddles the fg line
    target = torch.cat([normal, empty, overlap])
    torch.manual_seed(1)
    pred = torch.randn(3, 1, 64, 64)
    batch_loss = loss(pred, target)
    single_loss = loss(pred[:1], normal)
    torch.testing.assert_close(batch_loss, single_loss)


def test_region_loss_is_registered_combination():
    target = _line_target()
    pred = torch.randn(1, 1, 64, 64)
    region = WCMRegionLoss()
    expected = 0.5 * WCMFocalLoss()(pred, target) + \
        0.5 * WCMTverskyLoss()(pred, target)
    torch.testing.assert_close(region(pred, target), expected)


def test_containment_gradients_reach_both_heads():
    target = _line_target()
    skel = torch.randn(1, 1, 64, 64, requires_grad=True)
    mask = torch.randn(1, 1, 64, 64, requires_grad=True)
    WCMContainmentLoss()(skel, mask, target).backward()
    assert skel.grad.abs().sum() > 0 and mask.grad.abs().sum() > 0
    assert torch.isfinite(skel.grad).all() and torch.isfinite(mask.grad).all()


def test_warmup_hook_linear_schedule_and_resume_free():
    hook = LossWeightWarmupHook(schedules=[
        dict(loss_name='loss_cldice', base_weight=0.20, end_frac=0.20)])
    cldice = WCMSoftClDiceLoss(k=8)
    model = SimpleNamespace(decode_head=SimpleNamespace(
        loss_decode=[WCMRegionLoss(), cldice]))

    def run_at(it, max_iters=1000):
        runner = SimpleNamespace(model=model, iter=it, max_iters=max_iters)
        hook.before_train_iter(runner, batch_idx=0)
        return cldice.loss_weight

    assert run_at(0) == 0.0
    assert run_at(100) == pytest.approx(0.10)   # 10% progress -> half ramp
    assert run_at(200) == pytest.approx(0.20)   # 20% -> full weight
    assert run_at(900) == pytest.approx(0.20)
    # stateless: jumping straight to iter 150 (resume) gives the same value
    assert run_at(150) == pytest.approx(0.15)
