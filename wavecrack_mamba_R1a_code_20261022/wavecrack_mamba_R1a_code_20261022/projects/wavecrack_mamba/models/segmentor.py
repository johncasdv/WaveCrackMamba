import torch
import torch.nn.functional as F

from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.utils import resize
from mmseg.registry import MODELS


@MODELS.register_module()
class WCMSegmentor(EncoderDecoder):
    """WaveCrackMamba segmentor.

    Assembly (all optional modules identity/near-zero at init):

    - ``detail_branch`` (M0-3): near-zero ``inject`` added to trunk stage-0
      features; ``skip1/skip2`` consumed by the decoder (M0-6).
    - ``wavelet_bypass`` (M0-4): ``proj2`` feeds the mixer's direction router;
      ``proj1/proj2`` feed the decoder's semantic gates (M0-6).
    - ``axial_mixer`` (M0-5): transforms trunk 1/4 features between stage-1 and
      stage-2 via the trunk's ``transform_after_stage0`` hook.

    Input padding (Spec §8.1, M0-6): inputs are padded right/bottom-only to a
    multiple of 16 inside ``extract_feat`` -- reflect by default, replication
    when the image is too small for reflect. Outputs are strictly cropped back;
    padding therefore never reaches label evaluation. Doing this inside the
    model (rather than a DataPreprocessor subclass) keeps whole/slide/tensor
    paths and all three input branches (trunk/detail/wavelet) consistent with
    one padded view.

    Slide fusion (M0-6, §10-8 resolution): ``slide_mode='hann'`` weights each
    window by a 2D Hann window (clamped at ``1e-2`` so border pixels keep
    nonzero weight) -- smooth-weight fusion required by Spec §12.2;
    ``'count'`` keeps the mmseg uniform average as the comparison mode. With a
    single window both modes reduce exactly to whole-image inference.
    """

    def __init__(self, backbone, decode_head, detail_branch=None,
                 wavelet_bypass=None, axial_mixer=None, slide_mode='hann',
                 router_hf1=False, freeze_base_for_refiner=False, **kwargs):
        super().__init__(backbone=backbone, decode_head=decode_head, **kwargs)
        assert slide_mode in ('hann', 'count'), slide_mode
        self.slide_mode = slide_mode
        # router_hf1 (R1 rewire, 小波设计审查 v0.1 F2): Spec §8.2 requires the
        # router-input ablation to include an HF1+HF2 form, but only HF2@1/4
        # was ever wired -- the level whose 4px support has already mixed thin
        # cracks into aggregate texture. With this switch the router cue is
        # cat([proj2, avgpool2(proj1)]) (96+48=144ch at 1/4): level-1 carries
        # the strongest 1-2px crack response. Consumers must set the mixer's
        # ``hf2_channels`` to the concatenated width.
        assert not router_hf1 or wavelet_bypass is not None, (
            'router_hf1=True requires a wavelet_bypass')
        self.router_hf1 = router_hf1
        self.detail_branch = (
            MODELS.build(detail_branch) if detail_branch is not None else None)
        self.wavelet_bypass = (
            MODELS.build(wavelet_bypass) if wavelet_bypass is not None else None)
        self.axial_mixer = (
            MODELS.build(axial_mixer) if axial_mixer is not None else None)
        self.freeze_base_for_refiner = freeze_base_for_refiner
        if freeze_base_for_refiner:
            refiner = getattr(self.decode_head, 'refiner', None)
            if refiner is None:
                raise ValueError('freeze_base_for_refiner requires an enabled refiner')
            self.requires_grad_(False)
            refiner.requires_grad_(True)
            self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'freeze_base_for_refiner', False):
            for child in self.children():
                child.eval()
            self.decode_head.refiner.train(mode)
        return self

    # ------------------------------------------------------------ padding
    @staticmethod
    def _pad_to_16(x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 16, (-w) % 16
        if ph or pw:
            # reflect needs pad < dim; degrade to replicate for tiny inputs
            mode = 'reflect' if (ph < h and pw < w) else 'replicate'
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        return x

    def extract_feat(self, inputs):
        inputs = self._pad_to_16(inputs)
        detail = self.detail_branch(inputs) if self.detail_branch is not None \
            else None
        wavelet = self.wavelet_bypass(inputs) if self.wavelet_bypass is not None \
            else None
        inject = detail['inject'] if detail is not None else None
        transform = None
        if self.axial_mixer is not None:
            hf2 = wavelet['proj2'] if wavelet is not None else None
            if self.router_hf1 and wavelet is not None:
                hf2 = torch.cat(
                    [wavelet['proj2'], F.avg_pool2d(wavelet['proj1'], 2)], 1)
            def transform(feat):
                return self.axial_mixer(feat, hf2=hf2)
        feats = self.backbone(inputs, inject_after_stage0=inject,
                              transform_after_stage0=transform)
        if detail is not None:
            feats['detail'] = detail
        if wavelet is not None:
            feats['wavelet'] = wavelet
        return feats

    # --------------------------------------------- forward paths (crop back)
    def encode_decode(self, inputs, batch_img_metas):
        # Padded logits must be CROPPED to the input size before any resize to
        # img_shape -- resizing the padded map would squeeze pad content into
        # the geometry.
        h, w = inputs.shape[-2:]
        seg_logits = self.decode_head.forward(self.extract_feat(inputs))
        seg_logits = seg_logits[:, :, :h, :w]
        img_shape = batch_img_metas[0]['img_shape'][:2]
        if tuple(img_shape) != (h, w):
            seg_logits = resize(seg_logits, size=img_shape, mode='bilinear',
                                align_corners=self.align_corners)
        return seg_logits

    def _forward(self, inputs, data_samples=None):
        h, w = inputs.shape[-2:]
        return self.decode_head.forward(self.extract_feat(inputs))[:, :, :h, :w]

    # -------------------------------------------------------- slide fusion
    def _hann_window(self, h, w, device, dtype):
        wh = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
        ww = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
        return torch.clamp(wh.view(-1, 1) * ww.view(1, -1), min=1e-2)

    def slide_inference(self, inputs, batch_img_metas):
        if self.slide_mode == 'count':
            return super().slide_inference(inputs, batch_img_metas)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        weight = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                win = self._hann_window(y2 - y1, x2 - x1, inputs.device,
                                        preds.dtype)
                preds[:, :, y1:y2, x1:x2] += crop_seg_logit * win
                weight[:, :, y1:y2, x1:x2] += win
        assert (weight == 0).sum() == 0
        return preds / weight
