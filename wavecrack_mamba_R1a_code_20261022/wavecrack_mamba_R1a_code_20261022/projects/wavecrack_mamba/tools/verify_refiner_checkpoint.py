"""Verify real DF weights and synthetic forwards, without fitting a model."""
import argparse
import copy
import json
import logging
from pathlib import Path
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.logging import MMLogger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import mmpretrain.models  # noqa: E402,F401
import projects.dinov3_sealing.crack_modules  # noqa: E402,F401
import projects.wavecrack_mamba  # noqa: E402,F401
from mmseg.registry import MODELS  # noqa: E402
from projects.wavecrack_mamba.models.recurrent_training import (  # noqa: E402
    load_refiner_base_checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--height', type=int, default=65)
    parser.add_argument('--width', type=int, default=83)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope('mmseg')
    MMLogger.get_current_instance().setLevel(logging.WARNING)
    model = MODELS.build(cfg.model)
    model.init_weights()
    digest = load_refiner_base_checkpoint(model, args.checkpoint, args.sha256)
    baseline_cfg = copy.deepcopy(cfg.model)
    baseline_cfg.freeze_base_for_refiner = False
    baseline_cfg.decode_head.refiner = None
    baseline = MODELS.build(baseline_cfg)
    baseline.load_state_dict(
        {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith('decode_head.refiner.')
        },
        strict=True)
    model.to(args.device).eval()
    baseline.to(args.device).eval()
    torch.manual_seed(27)
    x = torch.randn(1, 3, args.height, args.width, device=args.device)
    with torch.no_grad():
        expected = baseline(x, mode='tensor')
        result = model(x, mode='tensor')
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)
    model.train()
    model(x, mode='tensor').float().square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.decode_head.refiner.parameters())
    assert all(p.grad is None for n, p in model.named_parameters()
               if not n.startswith('decode_head.refiner.'))
    report = dict(
        checkpoint_sha256=digest,
        config=args.config,
        torch_version=torch.__version__,
        device=args.device,
        scan_backend=model.axial_mixer.actual_backend_name,
        output_shape=list(result.shape),
        initial_max_abs_error=float((result - expected).abs().max()),
        trainable_parameters=sum(p.numel() for p in model.parameters()
                                 if p.requires_grad),
        frozen_base_gradient_check='PASS',
        synthetic_input=True,
        optimizer_steps=0,
        accuracy_evaluation=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
