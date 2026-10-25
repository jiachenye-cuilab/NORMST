"""Lossless storage of repeated tile CLS features; FP32 projection throughout."""
import time
import torch
from .geometry import stencil


def compact_features(features, valid, settings):
    """Verify every source value before discarding any repeated GPU storage."""
    if features.device.type != 'cpu' or valid.device.type != 'cpu':
        raise ValueError('compact features must be prepared on CPU before device transfer')
    channels, height, width = features.shape[1:]
    dimension, tile = int(settings['dimension']), int(settings['tile_cells'])
    offset_y, offset_x = settings.get('crop_offset_yx', [0, 0])
    if channels != 2*dimension+3 or features.dtype != torch.float32:
        raise ValueError('compact storage requires the verified CLS/local/RGB FP32 layout')
    yy = (torch.arange(height)+offset_y)//tile
    xx = (torch.arange(width)+offset_x)//tile
    indices = (yy[:, None]*(int(xx[-1])+1)+xx[None, :]).flatten()
    _, indices = torch.unique(indices, sorted=True, return_inverse=True)
    count = int(indices.max())+1
    mask = valid[0, 0].flatten()
    representatives = torch.full((count,), -1, dtype=torch.long)
    visible = torch.nonzero(mask, as_tuple=False).flatten()
    representatives[indices[visible]] = visible
    representatives.clamp_min_(0)
    raw = features[0].flatten(1)
    compact = raw[:dimension, representatives].contiguous()
    for begin in range(0, dimension, 32):
        end = min(begin+32, dimension)
        restored = compact[begin:end, indices]*mask
        if not torch.equal(restored, raw[begin:end]):
            raise ValueError('CLS features do not reconstruct exactly from tile representatives')
    local = features[:, dimension:].clone(memory_format=torch.contiguous_format)
    return compact, local, indices


class CompactHistologyMixin:
    """Compact a fixed image; expression contexts and parameter restores may vary.

    A different histology image must construct a fresh model, so cached source
    tensors can never silently outlive the checkpoint features they represent.
    """
    def _prepare_compact_histology(self, settings):
        self.compact_histology = bool(settings)
        if not settings:
            return
        started = time.perf_counter()
        compact, local, indices = compact_features(self.histology_features, self.feature_valid, settings)
        self.register_buffer('_compact_cls', compact, persistent=False)
        self.register_buffer('_compact_local', local, persistent=False)
        self.register_buffer('_compact_indices', indices, persistent=False)
        self._bind_compact_projection()
        self.compact_histology_record = dict(source_reconstruction_exact=True,
            preparation_seconds=time.perf_counter()-started,
            original_bytes=self.histology_features.numel()*self.histology_features.element_size(),
            compact_bytes=sum(value.numel()*value.element_size() for value in (compact, local, indices)))

    def _bind_compact_projection(self):
        self.he_encoder[0].compact_inputs = (self._compact_cls, self._compact_local,
                                            self._compact_indices, self.feature_valid)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if getattr(self, 'compact_histology', False):
            if assign:
                raise ValueError('compact histology requires copying into existing tensors (assign=False)')
            for name in ('histology_features', 'feature_valid'):
                incoming = state_dict.get(name)
                current = getattr(self, name)
                if incoming is not None and incoming.data_ptr() != current.data_ptr():
                    if not torch.equal(incoming.cpu(), current.cpu()):
                        raise ValueError('a different frozen image requires constructing a fresh model')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _apply(self, fn, recurse=True):
        if not getattr(self, 'compact_histology', False):
            return super()._apply(fn, recurse=recurse)
        # Keep the original FP32 buffer for the unchanged checkpoint format.
        # Only its lossless compact representation moves to the accelerator.
        original = self._buffers.pop('histology_features')
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._buffers['histology_features'] = original
        self._bind_compact_projection()
        return result

    def bypass_features(self, points):
        if not getattr(self, 'compact_histology', False):
            return super().bypass_features(points)
        if self.bypass_mode != 'local8':
            raise ValueError('compact histology currently supports the fixed local8 recipe')
        indices, weights = stencil(points, self.bounds, 8., self.basis_shape)
        source_valid = self.feature_valid[0, 0].flatten()
        valid = (weights*source_valid[indices]).sum(1)
        local = self._compact_local[0].flatten(1)
        values = local.new_zeros(len(points), self.histology_features.shape[1])
        for corner in range(4):
            rows = indices[:, corner]
            raw = torch.cat((self._compact_cls[:, self._compact_indices[rows]], local[:, rows]), 0)
            values = values + raw.T*(weights[:, corner]*source_valid[rows])[:, None]
        return self.image_bypass(values)*valid[:, None]
