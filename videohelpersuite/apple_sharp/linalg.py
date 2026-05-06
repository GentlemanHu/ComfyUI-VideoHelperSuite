"""Contains linear algebra related utility functions.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rotation_matrices_from_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert batch of quaternions into rotations matrices.

    Args:
        quaternions: The quaternions convert to matrices.

    Returns:
        The rotations matrices corresponding to the (normalized) quaternions.
    """
    device = quaternions.device
    shape = quaternions.shape[:-1]

    quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
    real_part = quaternions[..., 0]
    vector_part = quaternions[..., 1:]

    vector_cross = get_cross_product_matrix(vector_part)
    real_part = real_part[..., None, None]

    matrix_outer = vector_part[..., :, None] * vector_part[..., None, :]
    matrix_diag = real_part.square() * eyes(3, shape=shape, device=device)
    matrix_cross_1 = 2 * real_part * vector_cross
    matrix_cross_2 = vector_cross @ vector_cross

    return matrix_outer + matrix_diag + matrix_cross_1 + matrix_cross_2


def quaternions_from_rotation_matrices(matrices: torch.Tensor) -> torch.Tensor:
    """Convert batch of rotation matrices to quaternions (w-first).

    Pure PyTorch implementation using Shepperd's method. Runs on GPU.

    Args:
        matrices: Rotation matrices of shape [..., 3, 3].

    Returns:
        Quaternions of shape [..., 4] in (w, x, y, z) convention.
    """
    if not matrices.shape[-2:] == (3, 3):
        raise ValueError(f"matrices have invalid shape {matrices.shape}")

    batch_shape = matrices.shape[:-2]
    R = matrices.reshape(-1, 3, 3)
    n = R.shape[0]

    q = torch.empty(n, 4, device=R.device, dtype=R.dtype)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0 -- w is largest component
    m1 = trace > 0
    if m1.any():
        s = (trace[m1] + 1.0).clamp(min=1e-10).sqrt() * 2
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
        q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s
        q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal element
    m2 = (~m1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if m2.any():
        s = (1 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]).clamp(min=1e-10).sqrt() * 2
        q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
        q[m2, 1] = 0.25 * s
        q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal element
    m3 = (~m1) & (~m2) & (R[:, 1, 1] > R[:, 2, 2])
    if m3.any():
        s = (1 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]).clamp(min=1e-10).sqrt() * 2
        q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
        q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
        q[m3, 2] = 0.25 * s
        q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal element
    m4 = (~m1) & (~m2) & (~m3)
    if m4.any():
        s = (1 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1]).clamp(min=1e-10).sqrt() * 2
        q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s
        q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s
        q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s
        q[m4, 3] = 0.25 * s

    return q.reshape(batch_shape + (4,))


def get_cross_product_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Generate cross product matrix for vector exterior product."""
    if not vectors.shape[-1] == 3:
        raise ValueError("Only 3-dimensional vectors are supported")
    device = vectors.device
    shape = vectors.shape[:-1]
    unit_basis = eyes(3, shape=shape, device=device)
    # We compute the matrix by multiplying each column of unit_basis with the
    # corresponding vector.
    return torch.cross(vectors[..., :, None], unit_basis, dim=-2)


def eyes(
    dim: int, shape: tuple[int, ...], device: torch.device | str | None = None
) -> torch.Tensor:
    """Create batch of identity matrices."""
    return torch.eye(dim, device=device).broadcast_to(shape + (dim, dim)).clone()


def quaternion_product(q1, q2):
    """Compute dot product between two quaternions."""
    real_1 = q1[..., :1]
    real_2 = q2[..., :1]
    vector_1 = q1[..., 1:]
    vector_2 = q2[..., 1:]

    real_out = real_1 * real_2 - (vector_1 * vector_2).sum(dim=-1, keepdim=True)
    vector_out = real_1 * vector_2 + real_2 * vector_1 + torch.cross(vector_1, vector_2)
    return torch.concatenate([real_out, vector_out], dim=-1)


def quaternion_conj(q):
    """Get conjugate of a quaternion."""
    real = q[..., :1]
    vector = q[..., 1:]
    return torch.concatenate([real, -vector], dim=-1)


def project(u: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project tensor u to unit basis a."""
    unit_u = F.normalize(u, dim=-1)
    inner_prod = (unit_u * basis).sum(dim=-1, keepdim=True)
    return inner_prod * u
