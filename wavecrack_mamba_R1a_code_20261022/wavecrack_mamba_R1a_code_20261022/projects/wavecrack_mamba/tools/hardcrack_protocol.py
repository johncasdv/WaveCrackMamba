"""Dataset identity, paired initialization and HardCrack checkpoint provenance."""
import copy
import hashlib
import json
import uuid
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.dist import get_world_size
from .hardcrack_ods import HardCrackODSMetric  # noqa: F401
from .hardcrack_selection import (METRIC, RECEIPT, RULE, complete_run,
                                  validate_selected_base, write_json)

EXPECTED_FILES = {
    'splits/image_split.csv':
    '0dd09d5461c334fcbc86bcc0ab9569dba4d231a16d93a4728455c7fda4ffc7a1',
    'splits/train.txt':
    '8adc0c217dd43f157bf8e3d9623b0e8a946330d97a8f80128eb471edff377d34',
    'splits/train_tiles.txt':
    'ee3e8e779f77c9b4dfe4136cc2b6c1f92109fa6d366bc39848803c5b618fe1e6',
    'splits/val.txt':
    '722f38ee420ab9ee326a8312d2fefd076efd47b357057ccdda3385cc6361516c',
    'splits/test.txt':
    '9a81bd06f49636b3e342895c5e7a1d824de75e037bf10f88f8bf6b4bc59f6188',
    'metadata/train_manifest.jsonl':
    'a140206ba80bba220c9ffb32dc88d5a034009e17f08e9a2e82a2379fb5dc63d5',
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def release_identity(root, expected=None):
    """Check the registered split/manifest snapshot, not every image byte."""
    root = Path(root)
    expected = EXPECTED_FILES if expected is None else expected
    hashes = {name: sha256((root / name).read_bytes()) for name in expected}
    for name, value in hashes.items():
        if value != expected[name]:
            raise ValueError(f'HardCrack release snapshot mismatch: {name}')
    return dict(
        dataset='HardCrack-AC',
        files=hashes,
        scope='split_and_tile_manifest_identity')


def validate_config(cfg):
    """Reject incompatible views even if the data root points to HardCrack."""
    frozen = cfg.hardcrack_arm in ('t1', 't3', 'conv3')
    if cfg.get('hardcrack_checkpoint_rule') == RULE:
        if cfg.optim_wrapper.get('accumulative_counts', 1) != 1:
            raise ValueError(
                'This update schedule does not enable gradient accumulation')
        if not cfg.get('hardcrack_smoke', False) and not frozen:
            if (cfg.train_dataloader.batch_size != 8
                    or planned_updates(cfg) != cfg.hardcrack_expected_updates
                    or cfg.param_scheduler[-1].end
                    != cfg.hardcrack_expected_updates
                    or any(s.by_epoch for s in cfg.param_scheduler)):
                raise ValueError(
                    '64-epoch schedule, batch size and iteration budget disagree'
                )
    model = cfg.model
    if (model.decode_head.type != 'WCMInversePyramidHead'
            or model.get('detail_branch') is not None
            or model.get('wavelet_bypass') is not None):
        raise ValueError('HardCrack plan requires R1a without detail/wavelet')
    if model.slide_mode != 'count' or tuple(
            model.test_cfg.crop_size) != (512, 512) or tuple(
                model.test_cfg.stride) != (384, 384):
        raise ValueError(
            'HardCrack requires native 512/384 count-fused sliding')
    roots = set()
    for phase, split in [('train', 'train_tiles'), ('val', 'val'),
                         ('test', 'val' if frozen else 'test')]:
        ds = cfg[f'{phase}_dataloader'].dataset
        roots.add(str(Path(ds.data_root).resolve()))
        if ds.type != 'DeepCrackPairDataset' or ds.ann_file != f'splits/{split}.txt':
            raise ValueError(f'Incorrect HardCrack {phase} input')
        kinds = [t['type'] for t in ds.pipeline]
        allowed = ['LoadImageFromFile', 'LoadAnnotations', 'BinarizeSeg']
        if phase == 'train' and not frozen:
            allowed += ['RandomFlip']
        allowed += ['PackSegInputs']
        if kinds != allowed or ds.pipeline[2].thr != 0:
            raise ValueError(f'Unexpected resampling/label pipeline: {phase}')
        if ds.get('indices') is not None and not cfg.get(
                'hardcrack_smoke', False):
            raise ValueError('Subset indices are permitted only in smoke runs')
    if len(roots) != 1:
        raise ValueError('Mixed dataset roots are not allowed')
    if bool(model.get('freeze_base_for_refiner', False)) != frozen:
        raise ValueError('Incorrect base-freezing policy')
    if cfg.hardcrack_arm != 'nm':
        if model.axial_mixer.block.core.backend != 'mamba':
            raise ValueError('Mamba arms require the real mamba backend')


def validate_base(filename, expected_sha256, identity, allow_smoke=False):
    return validate_selected_base(
        filename, expected_sha256, identity, allow_smoke=allow_smoke)


def planned_updates(cfg):
    if cfg.train_cfg.type == 'IterBasedTrainLoop':
        return int(cfg.train_cfg.max_iters)
    indices = cfg.train_dataloader.dataset.get('indices')
    count = (len(indices) if isinstance(indices, (list, tuple)) else indices)
    count = 20996 if count is None else count
    batch = cfg.train_dataloader.batch_size
    batches = count // batch if cfg.train_dataloader.drop_last else (
        count + batch - 1) // batch
    return int(cfg.train_cfg.max_epochs * batches)


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_common(model, model_cfg, seed):
    """Load common tensors from one deterministic no-Mamba construction."""
    from mmseg.registry import MODELS
    cfg = copy.deepcopy(model_cfg)
    cfg.axial_mixer = None
    cfg.freeze_base_for_refiner = False
    cfg.decode_head.refiner = None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = MODELS.build(cfg)
        reference.init_weights()
    state = reference.state_dict()
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(
            not name.startswith('axial_mixer.')
            for name in incompatible.missing_keys):
        raise ValueError('Unexpected common-initialization mismatch')
    return state_digest(state)


def initialize_refiner(model, refiner_cfg, seed):
    """Same T1/T3 tensors; matching Conv3 submodules copy from T3 as well."""
    from mmseg.registry import MODELS
    cfg = copy.deepcopy(refiner_cfg)
    cfg.mode = 'transformer'
    cfg.steps = 3
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = MODELS.build(cfg)
    source = reference.state_dict()
    target = model.decode_head.refiner.state_dict()
    common = {
        k: v
        for k, v in source.items()
        if k in target and v.shape == target[k].shape
    }
    model.decode_head.refiner.load_state_dict(common, strict=False)
    return state_digest(common)


@HOOKS.register_module()
class WCMHardCrackProtocolHook(Hook):
    """Record executable protocol identity and initialize paired common weights."""
    priority = 'HIGHEST'

    def before_run(self, runner):
        cfg = runner.cfg
        validate_config(cfg)
        identity = release_identity(cfg.train_dataloader.dataset.data_root)
        self.info = dict(
            arm=cfg.hardcrack_arm,
            seed=cfg.randomness.seed,
            smoke=cfg.get('hardcrack_smoke', False),
            data=identity,
            slide_mode='count',
            checkpoint_rule=cfg.get('hardcrack_checkpoint_rule', 'final'),
            run_id=uuid.uuid4().hex,
            planned_updates=planned_updates(cfg),
            validation_history=[])
        if self.info['checkpoint_rule'] == RULE:
            if get_world_size() != 1:
                raise ValueError(
                    'This schedule is calibrated for one GPU per run')
            if (not cfg.get('hardcrack_evaluation_only', False)
                    and cfg.default_hooks.checkpoint.save_best != METRIC):
                raise ValueError('Checkpoint selection must use val/ODS')
        self.loaded = False
        self.training_started = False

    def after_load_checkpoint(self, runner, checkpoint):
        saved = checkpoint.get('meta', {}).get('hardcrack_run', {})
        for key in ('arm', 'seed', 'smoke', 'data', 'slide_mode'):
            if saved.get(key) != self.info.get(key):
                raise ValueError(f'HardCrack resume metadata mismatch: {key}')
        if runner._resume and any(
                saved.get(k) != self.info.get(k)
                for k in ('checkpoint_rule', 'planned_updates')):
            raise ValueError(
                'Resume cannot silently change the training budget or selection rule'
            )
        self.info = copy.deepcopy(saved)
        self.loaded = True

    def before_train(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        if runner.cfg.get('hardcrack_evaluation_only', False):
            raise ValueError(
                'An evaluation-only configuration cannot start training')
        if runner._load_from is not None and not runner._resume:
            raise ValueError(
                'Fresh training uses public initialization or the strict base hook'
            )
        if runner._resume:
            if not self.loaded:
                raise ValueError(
                    'Resume requested without a matching checkpoint')
        elif self.info['arm'] in ('nm', 'm'):
            self.info['common_init_sha256'] = initialize_common(
                model, runner.cfg.model, self.info['seed'])
        else:
            frozen = next(h for h in runner.cfg.custom_hooks
                          if h.type == 'WCMFrozenRefinerHook')
            base = validate_base(
                frozen.checkpoint,
                frozen.expected_sha256,
                self.info['data'],
                allow_smoke=self.info['smoke'])
            self.info['base_seed'] = base['seed']
            self.info['base_sha256'] = frozen.expected_sha256
            self.info['base_run_id'] = base['run_id']
            self.info['base_receipt_sha256'] = base[
                'completion_receipt_sha256']
            self.info['refiner_common_init_sha256'] = initialize_refiner(
                model, runner.cfg.model.decode_head.refiner, self.info['seed'])
        if self.info['checkpoint_rule'] == RULE:
            write_json(
                Path(runner.work_dir) / RECEIPT,
                dict(status='running', run_id=self.info['run_id']))
        self.training_started = True
        output = Path(runner.work_dir) / 'hardcrack_run.json'
        output.write_text(
            json.dumps(self.info, indent=2) + '\n', encoding='utf8')

    def before_val(self, runner):
        if not self.loaded and not self.training_started:
            raise ValueError(
                'Evaluation requires a matching loaded checkpoint')

    def before_test(self, runner):
        self.before_val(runner)

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        if not self.info['smoke']:
            return
        for sample in data_batch['data_samples']:
            labels = set(sample.gt_sem_seg.data.unique().tolist())
            if not labels <= {0, 1}:
                raise ValueError('Smoke found non-binary training labels')
            if tuple(sample.gt_sem_seg.data.shape[-2:]) != (512, 512):
                raise ValueError(
                    'Smoke training input is not an official tile')
        if not all(
                torch.isfinite(torch.as_tensor(value)).all()
                for value in outputs.values()):
            raise ValueError('Non-finite smoke training output')

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.info['smoke']:
            return
        shapes = self.info.setdefault('smoke_native_output_shapes', [])
        for sample in outputs:
            shape = list(sample.pred_sem_seg.data.shape[-2:])
            if shape != list(sample.ori_shape[:2]):
                raise ValueError('Native-resolution output shape mismatch')
            shapes.append(shape)

    def after_val_epoch(self, runner, metrics):
        if not self.training_started or self.info['checkpoint_rule'] != RULE:
            return
        expected = len(runner.val_dataloader.dataset)
        if metrics.get('val/num_images') != expected:
            raise ValueError(
                'ODS selection did not cover the declared validation set')
        row = dict(
            updates=int(runner.iter),
            score=float(metrics[METRIC]),
            threshold=float(metrics['val/ODS_threshold']))
        history = self.info.setdefault('validation_history', [])
        if history and row['updates'] <= history[-1]['updates']:
            raise ValueError('Duplicate or out-of-order validation event')
        history.append(row)
        self.info['last_validation'] = row

    def before_save_checkpoint(self, runner, checkpoint):
        saved = copy.deepcopy(self.info)
        # CheckpointHook supplies the completed iteration count for both loops
        # and for best-checkpoint saves after validation (no off-by-one).
        saved['updates'] = int(checkpoint['meta']['iter'])
        saved['is_final'] = saved['updates'] >= runner.max_iters
        checkpoint.setdefault('meta', {})['hardcrack_run'] = saved

    def after_train(self, runner):
        if self.info['checkpoint_rule'] != RULE:
            return
        if runner.iter != runner.max_iters:
            write_json(
                Path(runner.work_dir) / RECEIPT,
                dict(
                    status='incomplete',
                    run_id=self.info['run_id'],
                    completed_updates=int(runner.iter)))
            return
        selected = runner.message_hub.get_info('best_ckpt')
        final = runner.message_hub.get_info('last_ckpt')
        if not selected or not final:
            raise ValueError(
                'Completed training requires selected and final checkpoints')
        receipt = complete_run(self.info, selected, final, int(runner.iter))
        write_json(Path(runner.work_dir) / RECEIPT, receipt)
