"""Strict checkpoint admission for frozen-base refinement pilots."""
import hashlib
import io
import pickle
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _state_dict(checkpoint):
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict):
        raise TypeError('checkpoint must contain a state_dict mapping')
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state


def validate_refiner_checkpoint(model, checkpoint, base_only=False):
    """Validate all keys and shapes BEFORE mutating any model parameter."""
    model = getattr(model, 'module', model)
    state = _state_dict(checkpoint)
    target = model.state_dict()
    extra = set(state) - set(target)
    missing = set(target) - set(state)
    refiner_keys = {k for k in target if k.startswith('decode_head.refiner.')}
    if not refiner_keys:
        raise ValueError('model has no recurrent refiner')
    if base_only:
        if set(state) & refiner_keys:
            raise ValueError(
                'base checkpoint contains refiner weights; use resume')
        permitted = refiner_keys
    else:
        permitted = set()
    bad_shapes = [
        k for k in set(state) & set(target)
        if not torch.is_tensor(state[k]) or state[k].shape != target[k].shape
    ]
    if extra or missing != permitted or bad_shapes:
        raise ValueError(
            f'Incompatible checkpoint: unexpected={sorted(extra)}, '
            f'missing={sorted(missing - permitted)}, '
            f'shape_mismatch={sorted(bad_shapes)}')
    return state


def load_refiner_base_checkpoint(model, filename, expected_sha256=None):
    """Load a trusted local base checkpoint with only new refiner keys absent."""
    filename = Path(filename)
    # Deserialize exactly the bytes whose digest was checked, rather than
    # reopening a path that might change between verification and loading.
    data = filename.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError('base checkpoint SHA-256 mismatch')
    try:
        checkpoint = torch.load(
            io.BytesIO(data), map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        # Existing MMEngine DF files contain HistoryBuffer / NumPy metadata.
        # Do not automatically trust arbitrary pickle globals. The caller must
        # explicitly pin a known local checkpoint before legacy loading.
        if not expected_sha256:
            raise ValueError(
                'legacy checkpoint requires a trusted expected_sha256'
            ) from error
        checkpoint = torch.load(
            io.BytesIO(data), map_location='cpu', weights_only=False)
    state = validate_refiner_checkpoint(model, checkpoint, base_only=True)
    model = getattr(model, 'module', model)
    model.load_state_dict(state, strict=False)
    return digest


@HOOKS.register_module()
class WCMFrozenRefinerHook(Hook):
    """Load after init_weights; validate full checkpoints before Runner resumes.

    Freezing is done by WCMSegmentor before optimizer construction. This hook
    handles checkpoint admission only and does not silently change a model's
    training policy. Normal load_from is forbidden in a fresh pilot.
    """
    priority = 'VERY_HIGH'

    def __init__(self, checkpoint, expected_sha256=None):
        self.checkpoint = checkpoint
        self.expected_sha256 = expected_sha256
        self._validated_full_checkpoint = False
        self._base_admitted = False

    def before_run(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        if not getattr(model, 'freeze_base_for_refiner', False):
            raise ValueError(
                'frozen pilot requires freeze_base_for_refiner=True')

    def after_load_checkpoint(self, runner, checkpoint):
        # MMEngine calls this hook before copying checkpoint weights.
        validate_refiner_checkpoint(runner.model, checkpoint, base_only=False)
        self._validated_full_checkpoint = True

    def before_train(self, runner):
        if runner._resume:
            if not self._validated_full_checkpoint or not runner._has_loaded:
                raise ValueError(
                    'resume requested but no complete refiner checkpoint was loaded'
                )
        else:
            if runner._load_from is not None:
                raise ValueError(
                    'fresh pilot must use the strict base hook, not load_from')
            if not self.checkpoint or not Path(self.checkpoint).is_file():
                raise FileNotFoundError(
                    'set WCM_REFINER_BASE_CHECKPOINT to a local DF checkpoint')
            digest = load_refiner_base_checkpoint(runner.model,
                                                  self.checkpoint,
                                                  self.expected_sha256)
            self._base_admitted = True
            runner.logger.info(f'[WCM refiner] base_sha256={digest}')
        model = getattr(runner.model, 'module', runner.model)
        model.train()
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if not trainable or any(not n.startswith('decode_head.refiner.')
                                for n in trainable):
            raise ValueError(
                'only refiner parameters may be trainable in a frozen pilot')
        runner.logger.info(
            '[WCM refiner] base frozen/eval; final classifier keeps state gradients'
        )

    def _require_evaluation_weights(self, runner):
        if not self._base_admitted and not (self._validated_full_checkpoint
                                            and runner._has_loaded):
            raise ValueError(
                'evaluation requires a complete loaded refiner checkpoint')

    def before_val(self, runner):
        self._require_evaluation_weights(runner)

    def before_test(self, runner):
        self._require_evaluation_weights(runner)
