import numpy as np
import torch
import torch.nn.functional as F

from model_angelo.utils.torch_utils import is_ndarray, shared_cat


def get_affine(rot_matrix, shift):
    is_torch = torch.is_tensor(rot_matrix) and torch.is_tensor(shift)
    is_numpy = is_ndarray(rot_matrix) and is_ndarray(shift)

    if is_torch or is_numpy:
        if len(rot_matrix.shape) == len(shift.shape):
            return shared_cat((rot_matrix, shift), dim=-1, is_torch=is_torch)
        elif len(rot_matrix.shape) == len(shift.shape) + 1:
            return shared_cat((rot_matrix, shift[..., None]), dim=-1, is_torch=is_torch)
        else:
            raise ValueError(
                f"get_affine does not support rotation matrix of shape {rot_matrix.shape}"
                f"and shift of shape {shift.shape} "
            )
    else:
        raise ValueError(
            f"get_affine does not support different types for rot_matrix and shift, ie one is a numpy array, "
            f"the other is a torch tensor "
        )


def random_affine_from_translation(translation):
    bcd = torch.rand(*translation.shape[:-1], 3, device=translation.device) * 2 - 1
    bcdt = torch.cat((bcd, translation), dim=-1)
    return bcdt_to_affine(bcdt)


def init_affine_from_translation(translation):
    affine = torch.zeros(*translation.shape[:-1], 3, 4, device=translation.device)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    affine[..., 2, 2] = 1.0
    affine[..., :, -1] = translation
    return affine


def init_random_affine_from_translation(translation):
    v, w = torch.rand_like(translation), torch.rand_like(translation)
    rot = rots_from_two_vecs(v, w)
    return get_affine(rot, translation)


def affine_mul_rots(affine, rots):
    num_unsqueeze_dims = len(rots.shape) - len(affine.shape)
    if num_unsqueeze_dims > 0:
        new_shape = affine.shape[:-2] + num_unsqueeze_dims * (1,) + (3, 4)
        affine = affine.view(*new_shape)
    rotation = affine[..., :3, :3] @ rots
    return get_affine(rotation, get_affine_translation(affine))


def affine_mul_vecs(affine, vecs):
    num_unsqueeze_dims = len(vecs.shape) - len(affine.shape) + 1
    if num_unsqueeze_dims > 0:
        new_shape = affine.shape[:-2] + num_unsqueeze_dims * (1,) + (3, 4)
        affine = affine.view(*new_shape)
    return torch.einsum(
        "...ij, ...j-> ...i", get_affine_rot(affine), vecs
    ) + get_affine_translation(affine)


def affine_rot_vecs(affine, vecs):
    num_unsqueeze_dims = len(vecs.shape) - len(affine.shape) + 1
    if num_unsqueeze_dims > 0:
        new_shape = affine.shape[:-2] + num_unsqueeze_dims * (1,) + (3, 4)
        affine = affine.view(*new_shape)
    return torch.einsum("...ij, ...j-> ...i", get_affine_rot(affine), vecs)


def affine_add_vecs(affine, vecs):
    affine[..., :, -1] += vecs
    return affine


def get_affine_translation(affine):
    return affine[..., :, -1]


def get_affine_rot(affine):
    return affine[..., :3, :3]


def affine_composition(a1, a2):
    """
    Does the operation a1 o a2
    """
    rotation = get_affine_rot(a1) @ get_affine_rot(a2)
    translation = affine_mul_vecs(a1, get_affine_translation(a2))
    return get_affine(rotation, translation)


def rots_from_two_vecs(e1_unnormalized, e2_unnormalized):
    e1 = F.normalize(e1_unnormalized, p=2, dim=-1)
    c = torch.einsum("...i,...i->...", e2_unnormalized, e1)[..., None]  # dot product
    e2 = e2_unnormalized - c * e1
    e2 = F.normalize(e2, p=2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)


def affine_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane):
    rotation = rots_from_two_vecs(
        e1_unnormalized=origin - point_on_neg_x_axis,
        e2_unnormalized=point_on_xy_plane - origin,
    )
    return get_affine(rotation, origin)


def invert_affine(affine):
    inv_rots = get_affine_rot(affine).transpose(-1, -2)
    t = torch.einsum("...ij,...j->...i", inv_rots, affine[..., :, -1])
    inv_shift = -t
    return get_affine(inv_rots, inv_shift)


def affine_to_tensor_flat9(affine):
    return torch.stack(
        [affine[..., :, 0], affine[..., :, 1], affine[..., :, -1]], dim=-1
    )


def affine_to_tensor_flat12(affine):
    return torch.cat(
        [
            affine[..., 0, :3],
            affine[..., 1, :3],
            affine[..., 2, :3],
            affine[..., :, 3],
        ],
        dim=-1,
    )


def fill_rotation_matrix(xx, xy, xz, yx, yy, yz, zx, zy, zz):
    R = torch.zeros(*xx.shape, 3, 3).to(xx.device)
    R[..., 0, 0] = xx
    R[..., 0, 1] = xy
    R[..., 0, 2] = xz

    R[..., 1, 0] = yx
    R[..., 1, 1] = yy
    R[..., 1, 2] = yz

    R[..., 2, 0] = zx
    R[..., 2, 1] = zy
    R[..., 2, 2] = zz
    return R


def fill_rotation_matrix_np(xx, xy, xz, yx, yy, yz, zx, zy, zz):
    R = np.zeros(*xx.shape, 3, 3)
    R[..., 0, 0] = xx
    R[..., 0, 1] = xy
    R[..., 0, 2] = xz

    R[..., 1, 0] = yx
    R[..., 1, 1] = yy
    R[..., 1, 2] = yz

    R[..., 2, 0] = zx
    R[..., 2, 1] = zy
    R[..., 2, 2] = zz
    return R


def bcdt_to_affine(bcdt):
    # bcdt is the output of the network, the shape is (..., 6) where
    b, c, d, t = bcdt[..., 0], bcdt[..., 1], bcdt[..., 2], bcdt[..., 3:]
    a = torch.ones_like(b)
    abcd = F.normalize(torch.stack([a, b, c, d], dim=-1), p=2, dim=-1)
    rotation = torch.zeros(*t.shape, 3).to(t.device)
    abcd2 = abcd ** 2
    ab, bc = 2 * abcd[..., 0] * abcd[..., 1], 2 * abcd[..., 1] * abcd[..., 2]
    bd, ac = 2 * abcd[..., 1] * abcd[..., 3], 2 * abcd[..., 0] * abcd[..., 2]
    ad, cd = 2 * abcd[..., 0] * abcd[..., 3], 2 * abcd[..., 2] * abcd[..., 3]

    rotation = fill_rotation_matrix(
        xx=abcd2[..., 0] + abcd2[..., 1] - abcd2[..., 2] - abcd2[..., 3],
        xy=bc - ad,
        xz=bd + ac,
        yx=bc + ad,
        yy=abcd2[..., 0] - abcd2[..., 1] + abcd2[..., 2] - abcd2[..., 3],
        yz=cd - ab,
        zx=bd - ac,
        zy=cd + ab,
        zz=abcd2[..., 0] - abcd2[..., 1] - abcd2[..., 2] + abcd2[..., 3],
    )

    return get_affine(rotation, t)


def affine_to_k3(affine):
    R = get_affine_rot(affine)
    k3 = torch.zeros(*R.shape[:-2], 4, 4).to(R.device)

    # Diagonal elements
    r11, r22, r33 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    k3[..., 0, 0] = r11 - r22 - r33
    k3[..., 1, 1] = r22 - r11 - r33
    k3[..., 2, 2] = r33 - r11 - r22
    k3[..., 3, 3] = r11 + r22 + r33

    # k12 and k21
    r21, r12 = R[..., 1, 0], R[..., 0, 1]
    k3[..., 0, 1] = r21 + r12
    k3[..., 1, 0] = r21 + r12

    # k13 and k31
    r31, r13 = R[..., 2, 0], R[..., 0, 2]
    k3[..., 0, 2] = r31 + r13
    k3[..., 2, 0] = r31 + r13

    # k14 and k41
    r23, r32 = R[..., 1, 2], R[..., 2, 1]
    k3[..., 0, 3] = r23 - r32
    k3[..., 3, 0] = r23 - r32

    # k23 and k32
    k3[..., 1, 2] = r32 + r23
    k3[..., 2, 1] = r32 + r23

    # k24 and k42
    k3[..., 1, 3] = r31 - r13
    k3[..., 3, 1] = r31 - r13

    # k34 and k43
    k3[..., 2, 3] = r12 - r21
    k3[..., 3, 2] = r12 - r21

    return k3 / 3


def affine_to_bcdt(affines):
    k3 = affine_to_k3(affines)
    _, Q = torch.linalg.eigh(k3)
    bcd = Q[..., :3, -1]
    return torch.cat((bcd, get_affine_translation(affines)), dim=-1)


def affine_from_tensor4x4(m):
    assert m.shape[-1] == 4 == m.shape[-2]
    return get_affine(m[..., :3, :3], m[..., :3, -1])


def stop_rot_grad(affine):
    return get_affine(get_affine_rot(affine).detach(), get_affine_translation(affine))


def vecs_to_local_affine(affine, vecs):
    return affine_mul_vecs(invert_affine(affine), vecs)


def affines_to_local_affine(local_affine, other_affines):
    """
    Let local_affine be A and other_affines be A'
    We want to find affine A'' such that composing A with A'' gives you A'.
    In other words, if you (A) want to orient yourself with the other (A'), what update should you undergo (A'')

    The actual operation, given this formulation: A = (R, t) is
    A'' = (R'', t'') = (R^TR', R^Tt' - t)
    Notice that the affine translation is equivalent to
    vecs_to_local_affine(local_affine, get_affine_translation(other_affines))
    """
    return affine_composition(invert_affine(local_affine), other_affines)
