"""Standalone evaluation of a SHA-pinned local HardCrack checkpoint."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def prepare_eval_checkpoint(source, expected_sha256, destination):
    """Strip training-only pickle objects after verifying the trusted digest."""
    data = Path(source).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError('Evaluation checkpoint SHA-256 mismatch')
    checkpoint = torch.load(
        io.BytesIO(data), map_location='cpu', weights_only=False)
    info = checkpoint.get('meta', {}).get('hardcrack_run')
    if not info:
        raise ValueError('Checkpoint has no HardCrack provenance')
    # Preserve only JSON-valued protocol metadata and tensors, so MMEngine can
    # use PyTorch's weights-only default without allowlisting HistoryBuffer.
    info = json.loads(json.dumps(info))
    state = checkpoint['state_dict']
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor state_dict')
    torch.save(
        dict(state_dict=state, meta=dict(hardcrack_run=info)), destination)
    return digest


def evaluate(config, checkpoint, expected_sha256, work_dir, phase='val'):
    cfg = Config.fromfile(config)
    if phase == 'test' and cfg.test_dataloader.dataset.ann_file != 'splits/test.txt':
        raise ValueError(
            'Frozen pilot configurations evaluate validation, not test')
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    safe_file = work / 'state_for_evaluation.pth'
    if Path(checkpoint).resolve() == safe_file.resolve():
        raise ValueError(
            'Evaluation output must not overwrite its source checkpoint')
    digest = prepare_eval_checkpoint(checkpoint, expected_sha256, safe_file)
    cfg.load_from = str(safe_file)
    cfg.resume = False
    cfg.hardcrack_evaluation_only = True
    cfg.default_hooks.checkpoint = None
    cfg.work_dir = str(work)
    for hook in cfg.custom_hooks:
        if hook.type == 'WCMFrozenRefinerHook':
            hook.checkpoint = ''
            hook.expected_sha256 = ''
    runner = Runner.from_cfg(cfg)
    metrics = runner.val() if phase == 'val' else runner.test()
    protocol = next(
        h for h in runner.hooks
        if type(h).__name__ == 'WCMHardCrackProtocolHook')
    result = dict(
        checkpoint_sha256=digest,
        checkpoint_loaded=protocol.loaded,
        original_base_path_required=False,
        optimizer_updates=0,
        phase=phase,
        smoke=protocol.info['smoke'],
        native_output_shapes=protocol.info.get('smoke_native_output_shapes',
                                               []),
        metrics={
            k: float(v) if math.isfinite(float(v)) else None
            for k, v in metrics.items()
        })
    (work / 'evaluation.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--phase', choices=('val', 'test'), default='val')
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.sha256, args.work_dir,
             args.phase)
