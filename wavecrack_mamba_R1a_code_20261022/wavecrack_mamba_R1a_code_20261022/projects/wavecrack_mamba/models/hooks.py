"""Training hooks (Spec §9.6 warm-up schedules, §11.3 mandatory log fields)."""
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LossWeightWarmupHook(Hook):
    """Linear 0->1 warm-up of selected loss weights (Spec §9.6).

    ``schedules`` example (T1: topology warm-up over the first 20%)::

        dict(type='LossWeightWarmupHook',
             schedules=[dict(loss_name='loss_cldice', base_weight=0.20,
                             end_frac=0.20)])

    The scale is RECOMPUTED from ``iter/max_iters`` every train iter instead of
    keeping internal state, so ``--resume`` restores the exact schedule for
    free (the mmengine resume/scheduler pitfalls never apply).
    """

    def __init__(self, schedules):
        for s in schedules:
            assert {'loss_name', 'base_weight', 'end_frac'} <= set(s), s
            assert 0 < s['end_frac'] <= 1, s
        self.schedules = schedules

    @staticmethod
    def _iter_losses(model):
        model = getattr(model, 'module', model)  # unwrap DDP-style wrappers
        loss_decode = model.decode_head.loss_decode
        if isinstance(loss_decode, (list, nn.ModuleList)):
            return list(loss_decode)
        return [loss_decode]

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        progress = runner.iter / max(runner.max_iters, 1)
        losses = self._iter_losses(runner.model)
        for sched in self.schedules:
            scale = min(1.0, progress / sched['end_frac'])
            for loss_mod in losses:
                if getattr(loss_mod, 'loss_name', None) == sched['loss_name']:
                    loss_mod.loss_weight = sched['base_weight'] * scale


@HOOKS.register_module()
class WCMRunInfoHook(Hook):
    """Log the §11.3 mandatory experiment fields once at train start:
    actual scan backend/operator, slide mode, active losses. The pretrained
    source/sha256 fields live as config comments (M0-2); this hook surfaces
    the runtime-resolved facts that configs cannot know statically."""

    def before_train(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        backend = 'none (mixer disabled)'
        mixer = getattr(model, 'axial_mixer', None)
        if mixer is not None:
            backend = mixer.actual_backend_name
        losses = [getattr(loss, 'loss_name', type(loss).__name__)
                  for loss in LossWeightWarmupHook._iter_losses(model)]
        runner.logger.info(
            f'[WCM §11.3] scan_backend={backend} | '
            f"slide_mode={getattr(model, 'slide_mode', 'n/a')} | "
            f'active_losses={losses}')
