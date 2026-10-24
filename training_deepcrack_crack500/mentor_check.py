"""Check the relocated offline environment before mentor-machine training."""

import hashlib
import json
from pathlib import Path

import mmcv
import mmengine
import mmpretrain
import mmseg
import torch
from mamba_ssm import Mamba


ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / 'weights' / 'wcm_convnext_tiny_imagenet_mmpretrain.pth'
EXPECTED_SHA256 = 'b2ad0058499a3ea31634fe4fba49fe67a183750b10225b6874a1eeb4a16a31ec'


def main():
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    assert torch.__version__ == '2.1.0+cu118', torch.__version__
    assert mmcv.__version__ == '2.1.0', mmcv.__version__
    assert mmseg.__version__ == '1.2.2', mmseg.__version__
    assert mmpretrain.__version__ == '1.2.0', mmpretrain.__version__
    assert WEIGHTS.is_file(), WEIGHTS
    hasher = hashlib.sha256()
    with WEIGHTS.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    digest = hasher.hexdigest()
    assert digest == EXPECTED_SHA256, digest

    model = Mamba(d_model=96, d_state=16, d_conv=4, expand=2).cuda()
    output = model(torch.randn(2, 32, 96, device='cuda'))
    output.mean().backward()
    print(json.dumps(dict(
        status='passed', torch=torch.__version__, mmcv=mmcv.__version__,
        mmseg=mmseg.__version__, mmpretrain=mmpretrain.__version__,
        mmengine=mmengine.__version__, gpu=torch.cuda.get_device_name(0),
        mamba_forward_backward=True, weights_sha256=digest), indent=2))


if __name__ == '__main__':
    main()
