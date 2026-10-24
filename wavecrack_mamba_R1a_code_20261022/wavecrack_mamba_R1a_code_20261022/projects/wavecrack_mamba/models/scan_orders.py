"""Explicit axial scan order/inverse permutations (M0-5 test anchor).

The mixer's actual implementation uses reshape/flip for speed, but the scan
*semantics* are defined here as explicit index permutations, and the tests
assert the fast path matches these. Any future scan topology (X4 / SS2D
four-path) must register its orders here and pass the same bijection tests.

Why this exists: the official SCSegamba ``SAVSS_layer.py`` builds its third
diagonal inverse as ``o3_inverse[idx] = len(o1) - 1`` -- a constant, so the
"inverse" collapses to broadcasting the last step. ``is_bijective_pair`` below
is written so that exactly this class of bug fails loudly.
"""
import torch

SCAN_KEYS = ('row_fwd', 'row_rev', 'col_fwd', 'col_rev')


def axial_scan_orders(h, w, device=None):
    """Return ``{key: (order, inverse)}`` index permutations over ``h*w``.

    ``order[i]`` is the flat (row-major) pixel index visited at sequence step
    ``i``. ``inverse`` satisfies ``order[inverse] == arange`` and
    ``inverse[order] == arange`` (a true permutation inverse).

    - ``row_fwd``: rows scanned left->right (identity over row-major layout).
    - ``row_rev``: each row scanned right->left (flip along W, *not* a global
      sequence reversal -- the axial reverse happens inside each row).
    - ``col_fwd``: columns scanned top->bottom (column-major layout).
    - ``col_rev``: each column scanned bottom->top (flip along H).
    """
    idx = torch.arange(h * w, device=device).reshape(h, w)
    orders = {
        'row_fwd': idx.reshape(-1),
        'row_rev': idx.flip(1).reshape(-1),
        'col_fwd': idx.t().reshape(-1),
        'col_rev': idx.flip(0).t().reshape(-1),
    }
    return {k: (o, torch.argsort(o)) for k, o in orders.items()}


def is_bijective_pair(order, inverse):
    """True iff ``order`` is a permutation and ``inverse`` is its inverse.

    Catches the SCSegamba ``o3_inverse`` constant-index bug: a constant (or any
    non-permutation) inverse fails the round-trip identities below.
    """
    n = order.numel()
    if inverse.numel() != n:
        return False
    arange = torch.arange(n, device=order.device)
    if not torch.equal(torch.sort(order.reshape(-1))[0], arange):
        return False
    if not torch.equal(order.reshape(-1)[inverse.reshape(-1)], arange):
        return False
    return torch.equal(inverse.reshape(-1)[order.reshape(-1)], arange)


def unfold_axial(x, key):
    """Serialise ``x [B, C, H, W]`` into ``[B*, L, C]`` per the scan ``key``.

    This is the reshape/flip fast path the mixer uses; tests assert it equals
    gathering by ``axial_scan_orders`` so implementation and semantics can
    never drift apart.
    """
    b, c, h, w = x.shape
    if key == 'row_fwd':
        return x.permute(0, 2, 3, 1).reshape(b * h, w, c)
    if key == 'row_rev':
        return x.flip(3).permute(0, 2, 3, 1).reshape(b * h, w, c)
    if key == 'col_fwd':
        return x.permute(0, 3, 2, 1).reshape(b * w, h, c)
    if key == 'col_rev':
        return x.flip(2).permute(0, 3, 2, 1).reshape(b * w, h, c)
    raise KeyError(key)


def fold_axial(seq, key, b, c, h, w):
    """Inverse of :func:`unfold_axial`: ``[B*, L, C] -> [B, C, H, W]``."""
    if key == 'row_fwd':
        return seq.reshape(b, h, w, c).permute(0, 3, 1, 2)
    if key == 'row_rev':
        return seq.reshape(b, h, w, c).permute(0, 3, 1, 2).flip(3)
    if key == 'col_fwd':
        return seq.reshape(b, w, h, c).permute(0, 3, 2, 1)
    if key == 'col_rev':
        return seq.reshape(b, w, h, c).permute(0, 3, 2, 1).flip(2)
    raise KeyError(key)
