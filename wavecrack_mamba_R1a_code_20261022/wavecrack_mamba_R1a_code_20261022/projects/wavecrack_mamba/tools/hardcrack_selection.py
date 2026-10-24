"""Bind validation-selected weights to a completed matching training run."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path

import torch

RULE = 'val_ODS_best'
METRIC = 'val/ODS'
RECEIPT = 'completed_run.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def checked_checkpoint(path, expected_sha256):
    data = Path(path).read_bytes()
    if not expected_sha256 or digest(data) != expected_sha256.lower():
        raise ValueError('HardCrack checkpoint SHA-256 mismatch')
    # The caller pins a trusted local MMEngine artifact before legacy loading.
    return torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)


def best_validation(history):
    if not history:
        raise ValueError('No validation history for checkpoint selection')
    previous = 0
    for row in history:
        if (not isinstance(row['updates'], int) or row['updates'] <= previous
                or not math.isfinite(row['score'])
                or not 0 <= row['score'] <= 100
                or not math.isfinite(row['threshold'])
                or not 0 <= row['threshold'] <= 1):
            raise ValueError('Invalid validation selection history')
        previous = row['updates']
    return max(history, key=lambda row: row['score'])  # Earliest tie wins.


def complete_run(info, selected_path, final_path, completed_updates):
    """Call only after successful training, including final validation."""
    planned = info['planned_updates']
    history = info['validation_history']
    selected = best_validation(history)
    if completed_updates != planned or history[-1]['updates'] != planned:
        raise ValueError(
            'Training and final validation must finish before admission')
    selection = dict(
        split='val',
        metric=METRIC,
        **selected,
        checkpoint=Path(selected_path).name,
        sha256=digest(Path(selected_path).read_bytes()))
    final = dict(
        checkpoint=Path(final_path).name,
        sha256=digest(Path(final_path).read_bytes()),
        updates=planned)
    selected_info = checked_checkpoint(selected_path, selection['sha256']).get(
        'meta', {}).get('hardcrack_run', {})
    final_info = checked_checkpoint(final_path, final['sha256']).get(
        'meta', {}).get('hardcrack_run', {})
    keys = ('run_id', 'arm', 'seed', 'smoke', 'data', 'planned_updates',
            'checkpoint_rule')
    if (Path(selected_path).resolve().parent
            != Path(final_path).resolve().parent or any(
                record.get(k) != info.get(k)
                for record in (selected_info, final_info) for k in keys)
            or selected_info.get('last_validation') != selected
            or selected_info.get('updates') != selected['updates']
            or not final_info.get('is_final')
            or final_info.get('updates') != planned):
        raise ValueError(
            'Saved checkpoints do not prove completion and selection')
    return dict(
        schema_version=2,
        status='complete',
        run_id=info['run_id'],
        arm=info['arm'],
        seed=info['seed'],
        smoke=info['smoke'],
        data=info['data'],
        checkpoint_rule=RULE,
        planned_updates=planned,
        completed_updates=completed_updates,
        validation_history=copy.deepcopy(history),
        selection=selection,
        final=final)


def validate_selected_base(filename,
                           expected_sha256,
                           identity,
                           allow_smoke=False):
    path = Path(filename)
    checkpoint = checked_checkpoint(path, expected_sha256)
    info = checkpoint.get('meta', {}).get('hardcrack_run', {})
    receipt_path = path.parent / RECEIPT
    if not receipt_path.is_file():
        raise ValueError('Missing completed_run.json beside the selected base')
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if (receipt.get('schema_version') != 2
            or receipt.get('status') != 'complete'
            or receipt.get('checkpoint_rule') != RULE
            or info.get('checkpoint_rule') != RULE or receipt.get('arm') != 'm'
            or info.get('arm') != 'm' or receipt.get('data') != identity
            or info.get('data') != identity or not receipt.get('run_id')
            or receipt.get('run_id') != info.get('run_id')
            or receipt.get('seed') != info.get('seed')
            or receipt.get('smoke') != info.get('smoke')):
        raise ValueError(
            'Base is not selected from a completed matching HardCrack M run')
    if info.get('smoke') and not allow_smoke:
        raise ValueError('A smoke checkpoint cannot initialize a formal pilot')
    if not isinstance(info.get('smoke'), bool):
        raise ValueError('Missing smoke provenance')
    planned = receipt['planned_updates']
    if (not isinstance(planned, int) or planned <= 0
            or receipt['completed_updates'] != planned
            or info.get('planned_updates') != planned):
        raise ValueError('Run budget was not completed')
    winner = best_validation(receipt['validation_history'])
    if receipt['validation_history'][-1]['updates'] != planned:
        raise ValueError('Final validation is missing')
    selected = receipt['selection']
    if (selected.get('split') != 'val' or selected.get('metric') != METRIC
            or selected.get('checkpoint') != path.name
            or selected.get('sha256') != expected_sha256.lower()
            or any(selected.get(k) != winner[k]
                   for k in winner) or info.get('updates') != winner['updates']
            or info.get('last_validation') != winner):
        raise ValueError(
            'Checkpoint is not the recorded validation-best selection')
    final = receipt['final']
    if Path(final['checkpoint']).name != final['checkpoint']:
        raise ValueError('Final checkpoint must be a sibling file')
    final_checkpoint = checked_checkpoint(path.parent / final['checkpoint'],
                                          final['sha256'])
    final_info = final_checkpoint.get('meta', {}).get('hardcrack_run', {})
    if (not final_info.get('is_final') or final_info.get('updates') != planned
            or any(
                final_info.get(k) != info.get(k)
                for k in ('run_id', 'arm', 'seed', 'smoke', 'data',
                          'planned_updates', 'checkpoint_rule'))):
        raise ValueError(
            'Final checkpoint does not prove the recorded run completion')
    return dict(
        info,
        completion_receipt_sha256=digest(receipt_bytes),
        selected_metric=METRIC,
        selected_score=winner['score'])
