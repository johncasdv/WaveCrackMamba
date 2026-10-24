"""Axial SSM core with selectable backend (Spec §8.4 backend table).

- ``mamba``: real ``mamba_ssm.Mamba`` selective scan. The only backend valid
  for Mamba experiment conclusions; raises if the library is absent. Runs in
  the WSL ``mmseg_env`` on this machine (see the §8-3 environment report).
- ``gru``  : pure-PyTorch debug backend (CPU tests, pipeline verification).
  Never a source of Mamba performance conclusions.
- ``auto`` : ``mamba`` if importable else fall back to ``gru`` with a warning.

The core is *unidirectional*: bidirectionality lives in the caller, which runs
the same core on the flipped sequence and flips the result back
(``bidirectional_weight_sharing=true`` -- Spec §8.4 step 2).
"""
import importlib.util
import warnings

import torch.nn as nn
from mmengine.model import BaseModule

from mmseg.registry import MODELS


def mamba_available():
    return importlib.util.find_spec('mamba_ssm') is not None


@MODELS.register_module()
class AxialSSMCore(BaseModule):
    """Unidirectional sequence core: ``[B*, L, d_model] -> [B*, L, d_model]``."""

    def __init__(self, d_model=96, d_state=16, d_conv=4, expand=2,
                 backend='mamba', init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert backend in ('mamba', 'gru', 'auto'), backend
        self.d_model = d_model
        self.requested_backend = backend
        if backend == 'auto':
            if mamba_available():
                backend = 'mamba'
            else:
                warnings.warn(
                    'AxialSSMCore backend=auto: mamba_ssm not available, '
                    'falling back to the gru DEBUG backend. Results must not '
                    'be reported as Mamba (Spec §8.4).')
                backend = 'gru'
        if backend == 'mamba':
            if not mamba_available():
                raise ImportError(
                    "backend='mamba' but mamba_ssm is not installed. On this "
                    'machine run inside the WSL mmseg_env (§8-3 report), or '
                    "use backend='gru' for debugging only.")
            from mamba_ssm import Mamba
            self.core = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                              expand=expand)
            self.actual_backend_name = (
                f'mamba_ssm.Mamba(d_state={d_state},d_conv={d_conv},'
                f'expand={expand})')
        else:
            # Debug stand-in: capacity roughly matched via hidden=d_model*expand
            # then projected back. Recurrence direction = sequence order, same
            # as Mamba's causal scan.
            hidden = d_model * expand
            self.gru = nn.GRU(d_model, hidden, batch_first=True)
            self.gru_proj = nn.Linear(hidden, d_model)
            self.actual_backend_name = f'torch.GRU(hidden={hidden})'
        self.backend = backend

    def forward(self, seq):
        if self.backend == 'mamba':
            return self.core(seq)
        out, _ = self.gru(seq)
        return self.gru_proj(out)
