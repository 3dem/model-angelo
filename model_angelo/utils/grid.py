import argparse
from collections import namedtuple
from itertools import product
from typing import List, Tuple

import einops
import matplotlib.pylab as plt
import mrcfile as mrc
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from scipy.ndimage import convolve, gaussian_filter

from model_angelo.utils.affine_utils import get_affine
from model_angelo.utils.grid_rotations_to_rotation_mat import (
    grid_rotations_to_rotation_mat,
)
from model_angelo.utils.rotation_utils import get_rot_matrix


def apply_rotation(
    img_stack_ht: torch.Tensor, rot_matrix: torch.Tensor, dev=None
) -> torch.Tensor:
    if not dev:
        dev = img_stack_ht.device

    # Generate the 3x4 matrix for affine grid RR = (E|S) = (eye|shift)
    image_size = img_stack_ht.shape[-1]
    batch_size = img_stack_ht.shape[0]

    S = torch.zeros(batch_size, 3, 1).to(dev)
    RR = torch.cat([rot_matrix, S], 2)

    # generate affine Grid
    grid = torch.nn.functional.affine_grid(
        RR, [batch_size, 1, image_size, image_size, image_size], align_corners=False
    )
    grid = grid.to(dev)

    # apply shift
    img_stack_out = torch.nn.functional.grid_sample(
        input=img_stack_ht.reshape(
            batch_size, 1, image_size, image_size, image_size
        ).float(),
        grid=grid.float(),
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return img_stack_out


def center_of_mass(box):
    (Z, Y, X) = box.shape

    x_mean = np.mean(box, axis=(0, 1))
    x = np.sum(x_mean * range(X)) / np.sum(x_mean)

    y_mean = np.mean(box, axis=(0, 2))
    y = np.sum(y_mean * range(Y)) / np.sum(y_mean)

    z_mean = np.mean(box, axis=(1, 2))
    z = np.sum(z_mean * range(Z)) / np.sum(z_mean)

    return [x, y, z]


def get_batch_com(batch_box):
    (B, Z, Y, X) = batch_box.shape
    mesh = torch.flip(get_lattice_meshgrid(Z), [-1])[None]
    mesh -= mesh.mean(dim=[0, 1, 2, 3], keepdim=True)
    return torch.mean(batch_box[..., None].relu() * mesh, dim=[-4, -3, -2])


def get_batch_anti_com(batch_box):
    (B, Z, Y, X) = batch_box.shape
    mesh = torch.flip(get_lattice_meshgrid(Z), [-1])[None]
    mesh -= mesh.mean(dim=[0, 1, 2, 3], keepdim=True)
    batch_wise_max = torch.max(batch_box.view(B, -1), dim=-1)[0][..., None, None, None]
    return torch.mean(
        (batch_wise_max - batch_box)[..., None].relu() * mesh, dim=[-4, -3, -2]
    )


# TODO: Change save_mrc API to accept an MRCObject and a filename only
# From now on, grids should be accompanied inside an MRCObject so that the global_origin
# and voxel_sizes come too
def save_mrc(grid, voxel_size, origin, filename):
    (z, y, x) = grid.shape
    o = mrc.new(filename, overwrite=True)
    o.header["cella"].x = x * voxel_size
    o.header["cella"].y = y * voxel_size
    o.header["cella"].z = z * voxel_size
    o.header["origin"].x = origin[0]
    o.header["origin"].y = origin[1]
    o.header["origin"].z = origin[2]
    out_box = np.reshape(grid, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.update_header_stats()
    o.flush()
    o.close()


MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin"])


def load_mrc(mrc_fn: str, multiply_global_origin: bool = True) -> MRCObject:
    mrc_file = mrc.open(mrc_fn, "r")
    voxel_size = float(mrc_file.voxel_size.x)

    if voxel_size <= 0:
        raise RuntimeError(f"Seems like the MRC file: {mrc_fn} does not have a header.")
    
    c = mrc_file.header["mapc"]
    r = mrc_file.header["mapr"]
    s = mrc_file.header["maps"]

    global_origin = mrc_file.header["origin"]
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    global_origin[0] += mrc_file.header["nxstart"]
    global_origin[1] += mrc_file.header["nystart"]
    global_origin[2] += mrc_file.header["nzstart"]

    if multiply_global_origin:
        global_origin *= mrc_file.voxel_size.x

    if c == 1 and r == 2 and s == 3:
        grid = mrc_file.data
    elif c == 3 and r == 2 and s == 1:
        grid = np.moveaxis(mrc_file.data, [0, 1, 2], [2, 1, 0])
    elif c == 2 and r == 1 and s == 3:
        grid = np.moveaxis(mrc_file.data, [1, 2, 0], [2, 1, 0])
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")

    return MRCObject(grid, voxel_size, global_origin)


def load_mrc_header(mrc_fn: str):
    mrc_file = mrc.open(mrc_fn, "r", header_only=True)
    return mrc_file.header


def moveaxis_coordinates(coordinates, source, destination):
    source, destination = np.array(source).flatten(), np.array(destination).flatten()
    sort_idx = np.argsort(source)
    return np.stack(
        [
            coordinates[..., destination[sort_idx[0]]],
            coordinates[..., destination[sort_idx[1]]],
            coordinates[..., destination[sort_idx[2]]],
        ],
        axis=-1,
    )


def flip_coordinates(coordinates):
    return torch.stack(
        [coordinates[..., 1], coordinates[..., 2], coordinates[..., 0]], dim=-1
    )


def flip_coordinates_np(coordinates):
    return np.stack(
        [coordinates[..., 1], coordinates[..., 2], coordinates[..., 0]], axis=-1
    )


def get_fourier_shells(f):
    (z, y, x) = f.shape
    Z, Y, X = np.meshgrid(
        np.linspace(-z // 2, z // 2 - 1, z),
        np.linspace(-y // 2, y // 2 - 1, y),
        np.linspace(0, x - 1, x),
        indexing="ij",
    )
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R = np.fft.ifftshift(R, axes=(0, 1))
    return R


def get_spectral_indices(f):
    (z, y, x) = f.shape
    Z, Y, X = np.meshgrid(
        np.linspace(-z // 2, z // 2 - 1, z),
        np.linspace(-y // 2, y // 2 - 1, y),
        np.linspace(0, x - 1, x),
        indexing="ij",
    )
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R = np.round(np.fft.ifftshift(R, axes=(0, 1)))
    return R


def fourier_shell_avg(f, R=None):
    (fiz, fiy, fix) = f.shape

    if R is None:
        R = get_spectral_indices(f)

    R = np.round(R).astype(int)
    avg = np.zeros(fix)

    for r in range(fix):
        i = r == R
        avg[r] = np.sum(f[i]) / np.sum(i)

    return avg


def create_positional_encoding_freqs(enc_dim, d2):
    freqs = np.arange(enc_dim, dtype=np.float32)
    freqs = (
        2 * np.pi * d2 * (1.0 / d2) ** (freqs / (enc_dim - 1))
    )  # 2 * pi to 2 * D2 * pi
    freqs = freqs.reshape(*[1] * 3, -1)
    return freqs


def positional_encoding(coords, freqs):
    coords = np.linalg.norm(coords, ord=2, axis=-1, keepdims=True)
    k = coords * freqs
    s = np.sin(k)
    c = np.cos(k)
    x = np.concatenate((s, c), -1)
    return x.transpose(3, 0, 1, 2).astype(np.float32)


def spectral_amplitude(f, R=None):
    return fourier_shell_avg(np.abs(f), R)


def spectral_power(f, R=None):
    return fourier_shell_avg(np.square(np.abs(f)), R)


def get_fourier_shell_resolution(ft_size, voxel_size):
    res = np.zeros(ft_size)
    res[1:] = np.arange(1, ft_size) / (2 * voxel_size * ft_size)
    return res


def make_power_spectra(ft_size, voxel_size, bfac):
    r = get_fourier_shell_resolution(ft_size, voxel_size)
    return np.exp(-bfac * r * r), r


def make_cubic(box):
    bz = np.array(box.shape)
    s = np.max(box.shape)
    s += s % 2
    if np.all(box.shape == s):
        return box, np.zeros(3, dtype=np.int64), bz
    nbox = np.zeros((s, s, s))
    c = np.array(nbox.shape) // 2 - bz // 2
    nbox[c[0] : c[0] + bz[0], c[1] : c[1] + bz[1], c[2] : c[2] + bz[2]] = box
    return nbox, c, c + bz


def make_cubic_multiple_boxsize(box, boxsize):
    bz = np.array(box.shape)
    s = np.max(box.shape)
    s += s % 2
    if s % boxsize != 0:
        s += boxsize - (s % boxsize)
    if np.all(box.shape == s):
        return box, np.zeros(3, dtype=np.int64), bz
    nbox = np.zeros((s, s, s))
    c = np.array(nbox.shape) // 2 - bz // 2
    nbox[c[0] : c[0] + bz[0], c[1] : c[1] + bz[1], c[2] : c[2] + bz[2]] = box
    return nbox


def pad_to_shape(grid, shape):
    shape = np.array(shape)
    new_shape = np.max([shape, np.array(grid.shape)], axis=0)
    b = np.zeros(new_shape)
    c = np.array(b.shape) // 2 - np.array(grid.shape) // 2
    assert np.sum(c < 0) == 0
    s = grid.shape
    b[c[0] : c[0] + s[0], c[1] : c[1] + s[1], c[2] : c[2] + s[2]] = grid
    if not np.all(np.equal(b.shape, shape)):
        c_ = np.array(b.shape) // 2 - np.array(shape) // 2
        assert np.sum(c < 0) == 0
        s = shape
        b = b[c_[0] : c_[0] + s[0], c_[1] : c_[1] + s[1], c_[2] : c_[2] + s[2]]
        c -= c_

    return b, c


def rescale_fourier(box, out_sz):
    if out_sz % 2 != 0:
        raise Exception("Bad output size")
    if box.shape[0] != box.shape[1] or box.shape[1] != (box.shape[2] - 1) * 2:
        raise Exception("Input must be cubic")

    ibox = np.fft.ifftshift(box, axes=(0, 1))
    obox = np.zeros((out_sz, out_sz, out_sz // 2 + 1), dtype=box.dtype)

    si = np.array(ibox.shape) // 2
    so = np.array(obox.shape) // 2

    if so[0] < si[0]:
        obox = ibox[
            si[0] - so[0] : si[0] + so[0],
            si[1] - so[1] : si[1] + so[1],
            : obox.shape[2],
        ]
    elif so[0] > si[0]:
        obox[
            so[0] - si[0] : so[0] + si[0],
            so[1] - si[1] : so[1] + si[1],
            : ibox.shape[2],
        ] = ibox
    else:
        obox = ibox

    obox = np.fft.ifftshift(obox, axes=(0, 1))

    return obox


def rescale_real(box, out_sz):
    if out_sz != box.shape[0]:
        f = np.fft.rfftn(box)
        f = rescale_fourier(f, out_sz)
        box = np.fft.irfftn(f)

    return box


def normalize_voxel_size(density, in_voxel_sz, target_voxel_size=1.0):
    (iz, iy, ix) = np.shape(density)

    assert iz % 2 == 0 and iy % 2 == 0 and ix % 2 == 0
    assert ix == iy == iz

    in_sz = ix
    out_sz = int(round(in_sz * in_voxel_sz / target_voxel_size))
    if out_sz % 2 != 0:
        vs1 = in_voxel_sz * in_sz / (out_sz + 1)
        vs2 = in_voxel_sz * in_sz / (out_sz - 1)
        if np.abs(vs1 - target_voxel_size) < np.abs(vs2 - target_voxel_size):
            out_sz += 1
        else:
            out_sz -= 1

    out_voxel_sz = in_voxel_sz * in_sz / out_sz
    density = rescale_real(density, out_sz)

    return density, out_voxel_sz


def set_voxel_size(density, in_voxel_sz, target_out_voxel_sz=1):
    (iz, iy, ix) = np.shape(density)

    assert iz == iy == ix
    assert iz % 2 == 0 and iy % 2 == 0 and ix % 2 == 0
    assert ix == iy == iz

    in_sz = ix
    out_sz = int(round(in_sz * in_voxel_sz / target_out_voxel_sz))
    if out_sz % 2 != 0:
        vs1 = in_voxel_sz * in_sz / (out_sz + 1)
        vs2 = in_voxel_sz * in_sz / (out_sz - 1)
        if np.abs(vs1 - target_out_voxel_sz) < np.abs(vs2 - target_out_voxel_sz):
            out_sz += 1
        else:
            out_sz -= 1

    out_voxel_sz = in_voxel_sz * in_sz / out_sz
    density = rescale_real(density, out_sz)

    return density, out_voxel_sz


def resample_new_grid(
    grid, old_origin, old_voxel_size, new_origin, new_voxel_size, new_shape
):
    old_shape = grid.shape
    z = np.linspace(
        old_origin[2], old_origin[2] + (old_shape[0] - 1) * old_voxel_size, old_shape[0]
    )
    y = np.linspace(
        old_origin[1], old_origin[1] + (old_shape[1] - 1) * old_voxel_size, old_shape[1]
    )
    x = np.linspace(
        old_origin[0], old_origin[0] + (old_shape[2] - 1) * old_voxel_size, old_shape[2]
    )
    interpolator = scipy.interpolate.RegularGridInterpolator(
        (z, y, x), grid, method="nearest", bounds_error=False, fill_value=0.0
    )

    new_coo = np.zeros((np.prod(new_shape), 3))
    Z, Y, X = np.meshgrid(
        np.linspace(
            new_origin[2],
            new_origin[2] + (new_shape[0] - 1) * new_voxel_size,
            new_shape[0],
        ),
        np.linspace(
            new_origin[1],
            new_origin[1] + (new_shape[1] - 1) * new_voxel_size,
            new_shape[1],
        ),
        np.linspace(
            new_origin[0],
            new_origin[0] + (new_shape[2] - 1) * new_voxel_size,
            new_shape[2],
        ),
        indexing="ij",
    )
    new_coo[:, 0] = Z.flatten()
    new_coo[:, 1] = Y.flatten()
    new_coo[:, 2] = X.flatten()

    new_grid = interpolator(new_coo)
    new_grid = new_grid.reshape(new_shape)
    return new_grid


def coordinate_rot90(coordinates, i):
    alpha, beta, gamma = [np.pi * x / 2 for x in grid_rotations_to_rotation_mat[i]]
    rot_matrix = get_rot_matrix(alpha, beta, gamma, device=coordinates.device)
    return torch.einsum("ij,...j->...i", rot_matrix, coordinates)


def get_bounds_for_threshold(density, threshold=0.0):
    """Finds the bounding box encapsulating volume segment above threshold"""
    xy = np.all(density < threshold, axis=0)
    c = [[], [], []]
    c[0] = ~np.all(xy, axis=0)
    c[1] = ~np.all(xy, axis=1)
    c[2] = ~np.all(np.all(density <= threshold, axis=2), axis=1)

    h = np.zeros(3)
    l = np.zeros(3)
    (h[2], h[1], h[0]) = np.shape(density)

    for i in range(3):
        for j in range(len(c[i])):
            if c[i][j]:
                l[i] = j
                break
        for j in reversed(range(len(c[0]))):
            if c[i][j]:
                h[i] = j
                break

    return l.astype(int), h.astype(int)


def get_fsc_ft(map1_ft, map2_ft, voxel_size=0):
    assert np.any(map1_ft.shape == map2_ft.shape)
    (fiz, fiy, fix) = map1_ft.shape

    R = get_spectral_indices(map1_ft)

    fsc = np.zeros(fix)
    res = np.zeros(fix)

    c = (len(fsc) - 1) * 2 * voxel_size

    if voxel_size > 0:
        for i in np.arange(1, len(fsc)):
            j = i == R
            s1 = map1_ft[j]
            s2 = map2_ft[j]
            norm = np.sqrt(
                np.sum(np.square(np.abs(s1))) * np.sum(np.square(np.abs(s2)))
            )
            fsc[i] = np.real(np.sum(s1 * np.conj(s2))) / (norm + 1e-12)
            res[i] = c / i
    else:
        for i in np.arange(1, len(fsc)):
            j = i == R
            s1 = map1_ft[j]
            s2 = map2_ft[j]
            norm = np.sqrt(
                np.sum(np.square(np.abs(s1))) * np.sum(np.square(np.abs(s2)))
            )
            fsc[i] = np.real(np.sum(s1 * np.conj(s2))) / (norm + 1e-12)

    res[0] = 1e9
    fsc[0] = 1

    if voxel_size > 0:
        return res, fsc
    else:
        return fsc


def get_fsc(map1, map2, voxel_size=0):
    assert np.any(map1.shape == map2.shape)

    map1_ft = np.fft.rfftn(map1)
    map2_ft = np.fft.rfftn(map2)

    return get_fsc_ft(map1_ft, map2_ft, voxel_size)


def res_from_fsc(fsc, res, threshold=0.5):
    """
    Calculates the resolution (res) at the FSC (fsc) threshold.
    """
    assert len(fsc) == len(res)
    i = np.argmax(fsc < 0.5)
    if i > 0:
        return res[i - 1]
    else:
        return res[-1]


def plot_grid_rgyr(grid):
    max_value = np.max(grid)
    if max_value <= 0:
        raise RuntimeError("All values equal/less than zero")

    idx_offset = np.array(grid.shape).astype(float) / 2.0

    N1 = 50
    x = np.linspace(0, max_value / 2, N1)
    rgyr = np.zeros(N1)

    for i in range(N1):
        mask = grid > x[i]
        idx = np.array(np.where(mask)).astype(float)
        idx -= idx_offset[:, None]
        rgyr[i] = np.sum(idx ** 2) / np.sum(mask)

    return x, rgyr


def plot_grid_sparsity(grid):
    max_value = np.max(grid)
    if max_value <= 0:
        raise RuntimeError("All values equal/less than zero")

    idx_offset = np.array(grid.shape).astype(float) / 2.0

    N1 = 200
    x = np.linspace(0, max_value / 2, N1)
    sparsity = np.zeros(N1)

    for i in range(N1):
        mask = grid > x[i]

        shift = mask.astype(int)
        shift[:-1, :, :] += mask[1:, :, :]
        shift[:, :-1, :] += mask[:, 1:, :]
        shift[:, :, :-1] += mask[:, :, 1:]

        sparsity[i] = np.sum(shift < 3)

    return x, sparsity


def running_mean(x, N):
    avg = np.copy(x)
    count = np.ones(len(x))  # Ones, becouse we start of with the original array

    for i in range(N):
        avg[1 + i : -2 - i] += x[0 : -3 - 2 * i] + x[2 + 2 * i : -1]
        count[1 + i : -2 - i] += 2

    avg /= count
    return avg


def circular_mask(bz):
    Z, Y, X = np.meshgrid(
        np.linspace(-bz // 2, bz // 2 - 1, bz),
        np.linspace(-bz // 2, bz // 2 - 1, bz),
        np.linspace(-bz // 2, bz // 2 - 1, bz),
        indexing="ij",
    )
    return (X ** 2 + Y ** 2 + Z ** 2 < bz * bz / 4).astype(float)


def grid_rot90(m, i=None, has_batch=False):
    j = 1 if has_batch else 0
    if i is None:
        i = np.random.randint(24)
    if i == 0:
        return m
    if i == 1:
        return np.rot90(m, 1, (0 + j, 2 + j))
    if i == 2:
        return np.rot90(m, 2, (0 + j, 2 + j))
    if i == 3:
        return np.rot90(m, 3, (0 + j, 2 + j))
    if i == 4:
        return np.rot90(m, 1, (1 + j, 2 + j))
    if i == 5:
        return np.rot90(m, 1, (2 + j, 1 + j))
    if i == 6:
        return np.rot90(m, 1, (0 + j, 1 + j))
    if i == 7:
        return np.rot90(np.rot90(m, 1, (0 + j, 1 + j)), 1, (0 + j, 2 + j))
    if i == 8:
        return np.rot90(np.rot90(m, 1, (0 + j, 1 + j)), 2, (0 + j, 2 + j))
    if i == 9:
        return np.rot90(np.rot90(m, 1, (0 + j, 1 + j)), 3, (0 + j, 2 + j))
    if i == 10:
        return np.rot90(np.rot90(m, 1, (0 + j, 1 + j)), 1, (1 + j, 2 + j))

    if i == 11:
        return np.rot90(np.rot90(m, 1, (0 + j, 1 + j)), 1, (2 + j, 1 + j))
    if i == 12:
        return np.rot90(m, 2, (0 + j, 1 + j))
    if i == 13:
        return np.rot90(np.rot90(m, 2, (0 + j, 1 + j)), 1, (0 + j, 2 + j))
    if i == 14:
        return np.rot90(np.rot90(m, 2, (0 + j, 1 + j)), 2, (0 + j, 2 + j))
    if i == 15:
        return np.rot90(np.rot90(m, 2, (0 + j, 1 + j)), 3, (0 + j, 2 + j))
    if i == 16:
        return np.rot90(np.rot90(m, 2, (0 + j, 1 + j)), 1, (1 + j, 2 + j))
    if i == 17:
        return np.rot90(np.rot90(m, 2, (0 + j, 1 + j)), 1, (2 + j, 1 + j))
    if i == 18:
        return np.rot90(m, 3, (0 + j, 1 + j))
    if i == 19:
        return np.rot90(np.rot90(m, 3, (0 + j, 1 + j)), 1, (0 + j, 2 + j))
    if i == 20:
        return np.rot90(np.rot90(m, 3, (0 + j, 1 + j)), 2, (0 + j, 2 + j))
    if i == 21:
        return np.rot90(np.rot90(m, 3, (0 + j, 1 + j)), 3, (0 + j, 2 + j))
    if i == 22:
        return np.rot90(np.rot90(m, 3, (0 + j, 1 + j)), 1, (1 + j, 2 + j))
    if i == 23:
        return np.rot90(np.rot90(m, 3, (0 + j, 1 + j)), 1, (2 + j, 1 + j))
    return m


def slice_by_boxsize_3d(x, box_size):
    assert len(x.shape) >= 3
    assert x.shape[0] == x.shape[1] == x.shape[2]
    assert x.shape[0] % box_size == 0

    slice_product = product(
        *[
            [
                np.s_[box_size * k : box_size * (k + 1)]
                for k in range(x.shape[0] // box_size)
            ]
            for _ in range(3)
        ]
    )
    slice_product = list(map(lambda x: np.s_[x[0], x[1], x[2]], slice_product))

    return slice_product


def tuple_slice_by_boxsize_3d(x, box_size):
    assert len(x.shape) >= 3
    assert x.shape[0] == x.shape[1] == x.shape[2]
    assert x.shape[0] % box_size == 0

    slice_product = product(
        *[
            [(box_size * k, box_size * (k + 1)) for k in range(x.shape[0] // box_size)]
            for _ in range(3)
        ]
    )
    slice_product = list(map(lambda x: (x[0], x[1], x[2]), slice_product))

    return slice_product


def get_lattice_meshgrid_np(shape, no_shift=False):
    linspace = np.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
    )
    mesh = np.stack(
        np.meshgrid(linspace, linspace, linspace, indexing="ij"),
        axis=-1,
    )
    return mesh


def get_lattice_meshgrid(shape, no_shift=False, device="cpu"):
    linspace = torch.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
        device=device,
    )
    mesh = torch.stack(
        torch.meshgrid(linspace, linspace, linspace, indexing="ij"),
        dim=-1,
    )
    return mesh


def sample_from_distribution(x, y):
    cdf = np.cumsum(y / np.sum(y))
    y_selected = np.random.uniform()  # Between zero and one
    x_selected = np.interp(y_selected, cdf, x)
    return x_selected


def coloured_noise_spectrum(
    spectral_amplitude, lres_noise_power=0.8, hres_noise_power=0.8
):
    f = np.arange(len(spectral_amplitude)).astype(float) / float(
        len(spectral_amplitude)
    )
    noise_spectral_amplitude_fraction = lres_noise_power + f * (
        hres_noise_power - lres_noise_power
    )
    noise_spectral_amplitude = noise_spectral_amplitude_fraction * spectral_amplitude
    return noise_spectral_amplitude, lres_noise_power, hres_noise_power


def get_neighbouring_points(i, j, k):
    return [
        (i - 1, j, k),
        (i + 1, j, k),
        (i, j - 1, k),
        (i, j + 1, k),
        (i, j, k - 1),
        (i, j, k + 1),
    ]


def cosine_similarity_np(vec1, vec2):
    return np.einsum("...x,...x->...", vec1, vec2) / (
        np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1) + 1e-6
    )


def cosine_similarity(vec1, vec2):
    with torch.no_grad():
        vector_norms = (
            torch.linalg.norm(vec1, dim=-1) * torch.linalg.norm(vec2, dim=-1) + 1e-6
        )
    return torch.einsum("...x,...x->...", vec1, vec2) / vector_norms


def convert_to_coordinates(
    output_grid: torch.Tensor, threshold: float = 0.5, apply_sigmoid=False
) -> List[torch.Tensor]:
    assert len(output_grid.shape) == 5 and output_grid.shape[1] >= 4
    mesh = get_lattice_meshgrid(
        output_grid.shape[-1], no_shift=True, device=output_grid.device
    ).flip(-1)
    if apply_sigmoid:
        output_grid = torch.sigmoid(output_grid)

    batch_coordinates = []
    for output in output_grid:
        # Shape of output: cxyz
        mask = output[0] > threshold
        batch_coordinates.append(mesh[mask] + output[..., mask][1:].t() - 0.5)
    return batch_coordinates


def slice_grid_numpy(global_grid, positions, box_size=8):
    grid = []
    positions = np.clip(
        np.around(positions), box_size // 2, global_grid.shape[0] - box_size // 2 - 1
    ).astype(int)
    for p in positions:
        grid.append(
            global_grid[
                p[2] - box_size // 2 : p[2] + box_size // 2,
                p[1] - box_size // 2 : p[1] + box_size // 2,
                p[0] - box_size // 2 : p[0] + box_size // 2,
            ]
        )
    return np.stack(grid, axis=0)


def slice_grid(global_grid, positions, box_size=8):
    grid = []
    positions = torch.clamp(
        torch.round(positions), box_size // 2, global_grid.shape[0] - box_size // 2 - 1
    ).long()
    for p in positions:
        grid.append(
            global_grid[
                ...,
                p[2] - box_size // 2 : p[2] + box_size // 2,
                p[1] - box_size // 2 : p[1] + box_size // 2,
                p[0] - box_size // 2 : p[0] + box_size // 2,
            ]
        )
    return torch.stack(grid, dim=0)


def weighted_average_of_ca_positions(network_logits, shape, threshold=0.7):
    lattice = torch.tensor(
        np.flip(get_lattice_meshgrid_np(shape, no_shift=True), -1),
        dtype=network_logits.dtype,
        device=network_logits.device,
    )
    lattice = lattice.reshape(3, shape, shape, shape)
    network_logits = network_logits.reshape(1, shape, shape, shape)
    mul = network_logits * lattice

    logit_sums = (
        F.avg_pool3d(network_logits, kernel_size=2, stride=1) * 8
    )  # Sum pooling
    weighted_average_position = F.avg_pool3d(mul, kernel_size=2, stride=1) / logit_sums
    return weighted_average_position[:, logit_sums > threshold], logit_sums


def remove_dust_from_volume(volume):
    x = np.copy(volume)
    med = np.median(x)
    q = np.quantile(x, q=0.92)

    xt = torch.Tensor(x)
    yt = torch.zeros_like(xt)
    yt[xt > q] = 1

    k = torch.ones(1, 1, 3, 3, 3)
    yt = yt.reshape(1, 1, *yt.shape)
    k_yt = F.conv3d(yt, k, stride=1, padding=1)

    k_yt[k_yt < 13] = 0
    k_yt[k_yt != 0] = 1

    xt = xt - (xt - float(med)) * (1 - k_yt) * (yt)
    new_x = xt[0][0].numpy()
    return new_x


def clean_map(volume):
    x = np.copy(volume)
    x = remove_dust_from_volume(x)
    x -= np.median(x)
    std = np.std(x[x > 0])
    x[(x < std) * (x > 0)] = 0
    return x


def voxelize_coordinates(grid: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    coordinates_round = torch.round(coordinates).long()
    coordinates_round = coordinates_round[
        (coordinates_round > 0) * (coordinates_round < grid.shape[-1])
    ].reshape(-1, 3)
    for p in coordinates_round:
        grid[p[..., 2], p[..., 1], p[..., 0]] = 1
    return grid


def voxelize_coordinates_numpy(
    grid: np.ndarray, coordinates: np.ndarray, clip_output=False
) -> np.ndarray:
    coordinates_round = np.around(coordinates).astype(np.int64)
    coordinates_round = coordinates_round[
        (coordinates_round > 0).all(axis=-1)
        * (coordinates_round < grid.shape[-1]).all(axis=-1)
    ].reshape(-1, 3)
    for p in coordinates_round:
        grid[p[..., 2], p[..., 1], p[..., 0]] += 1
    if clip_output:
        grid = np.clip(grid, 0, 1)
    return grid


def voxelize_coordinate_indices(
    grid: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    coordinates_round = torch.round(coordinates).long()
    coordinate_indices = torch.arange(1, len(coordinates) + 1, dtype=torch.long)

    mask = (coordinates_round >= 0).all(dim=-1) * (
        coordinates_round <= grid.shape[-1]
    ).all(dim=-1)
    coordinate_indices = coordinate_indices[mask]
    coordinates_round = coordinates_round[mask].reshape(-1, 3)

    for (p, v) in zip(coordinates_round, coordinate_indices):
        grid[p[..., 2], p[..., 1], p[..., 0]] = v

    return grid


def make_solvent_mask(grid: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    sim_lp = gaussian_filter(grid, 1)
    sim_mask_hard = np.zeros(grid.shape)
    sim_mask_hard[sim_lp > 0.1] = 1.0
    sim_mask_smooth = gaussian_filter(sim_mask_hard, 0.5)
    return sim_mask_hard, sim_mask_smooth


def mrc_log_norm(grid: ArrayLike) -> ArrayLike:
    x = np.copy(grid)
    x -= np.median(x)
    x = np.maximum(x, 0)
    q = np.quantile(x[x > 0], q=0.9998)
    x = (x / q) * (np.e - 1)
    return np.log(x + 1)


def grid_sampler_unnormalize(coord, size, align_corners=False):
    if align_corners:
        return ((coord + 1) / 2) * (size - 1)
    else:
        return ((coord + 1) * size - 1) / 2


def grid_sampler_normalize(coord, size, align_corners=False):
    if align_corners:
        return (2 / (size - 1)) * coord - 1
    else:
        return ((2 * coord + 1) / size) - 1


def get_sscpm(a, b):
    """
    Get skew symmetric cross product matrix for a cross b
    """
    v = torch.cross(a, b, dim=-1)
    v_x = torch.zeros(*v.shape[:-1], 3, 3).to(a.device)
    # [   0, -v_3,  v_2]
    # [ v_3,    0, -v_1]
    # [-v_2,  v_1,    0]
    v_x[..., 0, 1] = -v[..., 2]
    v_x[..., 1, 0] = v[..., 2]
    v_x[..., 0, 2] = v[..., 1]
    v_x[..., 2, 0] = -v[..., 1]
    v_x[..., 2, 1] = v[..., 0]
    v_x[..., 1, 2] = -v[..., 0]
    return v_x


def get_z_to_w_rotation_matrix(w):
    """
    Special case of get_a_to_b_rotation matrix for when you are converting from
    the Z axis to some vector w. Algorithm comes from
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    _w = F.normalize(w, p=2, dim=-1)
    # (1, 0, 0) cross _w
    v2 = -_w[..., 2]
    v3 = _w[..., 1]
    # (1, 0, 0) dot _w
    c = _w[..., 0]
    # The result of I + v_x + v_x_2 / (1 + c)
    # [   1 - (v_2^2 + v_3 ^ 2) / (1 + c),                   -v_3,                    v_2]
    # [                               v_3,    1 - v_3^2 / (1 + c),    v_2 * v_3 / (1 + c)]
    # [                              -v_2,    v_2 * v_3 / (1 + c),    1 - v_2^2 / (1 + c)]
    R = torch.zeros(*w.shape[:-1], 3, 3).to(w.device)
    v2_2, v3_2 = ((v2 ** 2) / (1 + c)), ((v3 ** 2) / (1 + c))
    v2_v3 = v2 * v3 / (1 + c)
    R[..., 0, 0] = 1 - (v2_2 + v3_2)
    R[..., 0, 1] = -v3
    R[..., 0, 2] = v2
    R[..., 1, 0] = v3
    R[..., 1, 1] = 1 - v3_2
    R[..., 1, 2] = v2_v3
    R[..., 2, 0] = -v2
    R[..., 2, 1] = v2_v3
    R[..., 2, 2] = 1 - v2_2
    return R


def get_a_to_b_rotation_matrix(a, b):
    """
    Algorithm comes from
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    a, b = F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1)
    c = torch.einsum("...i,...i->...", a, b)[..., None, None]
    v_x = get_sscpm(a, b)
    v_x_2 = v_x @ v_x
    I = torch.eye(3).to(a.device)
    R = I + v_x + v_x_2 / (1 + c)
    return R


def sample_rectangle(
    grid, rotation_matrices, shifts, rectangle_shape=(10, 1, 1), align_corners=True
):
    assert len(grid.shape) == 5
    assert grid.shape[2] == grid.shape[3] == grid.shape[4]
    bz, cz, sz = grid.shape[0], grid.shape[1], grid.shape[2]
    scale_mult = (
        (torch.Tensor([rectangle_shape[2], rectangle_shape[1], rectangle_shape[0]]) - 1)
        / (sz - 1)
    ).to(grid.device)
    rotation_matrices = rotation_matrices * scale_mult[None, ..., None]
    shifts = (
        grid_sampler_normalize(shifts, sz, align_corners=align_corners)
        + scale_mult[None]
    )
    affine_matrix = get_affine(rotation_matrices, shifts)
    cc = F.affine_grid(
        affine_matrix,
        (
            bz,
            cz,
        )
        + rectangle_shape,
        align_corners=align_corners,
    )
    return F.grid_sample(grid.detach(), cc, align_corners=align_corners)


def sample_rectangle_along_vector(
    grid,
    vectors,
    origin_points,
    rectangle_shape=(10, 2, 2),
    marginalization_dims=None,
):
    rotation_matrices = get_z_to_w_rotation_matrix(vectors)
    rectangle = sample_rectangle(
        grid, rotation_matrices, origin_points, rectangle_shape=rectangle_shape
    )
    if marginalization_dims is not None:
        rectangle = rectangle.sum(dim=marginalization_dims)
    return rectangle


def sample_centered_rectangle(
    grid,
    rotation_matrices,
    shifts,
    rectangle_length=10,
    rectangle_width=3,
    align_corners=True,
):
    assert len(grid.shape) == 5
    assert grid.shape[2] == grid.shape[3] == grid.shape[4]
    bz, cz, sz = grid.shape[0], grid.shape[1], grid.shape[2]
    scale_mult = (
        (
            torch.Tensor(
                [
                    rectangle_width,
                    rectangle_width,
                    rectangle_length,
                ]
            )
            - 1
        )
        / (sz - 1)
    ).to(grid.device)
    rotation_matrices = rotation_matrices * scale_mult[None, ..., None]
    center_shift_vector = torch.Tensor(
        [[0, rectangle_width // 2, rectangle_width // 2]]
    ).to(shifts.device)
    shifts = (
        grid_sampler_normalize(
            shifts - center_shift_vector, sz, align_corners=align_corners
        )
        + scale_mult[None]
    )
    affine_matrix = get_affine(rotation_matrices, shifts)
    cc = F.affine_grid(
        affine_matrix,
        (
            bz,
            cz,
        )
        + (rectangle_length, rectangle_width, rectangle_width),
        align_corners=align_corners,
    )
    return F.grid_sample(grid.detach(), cc, align_corners=align_corners)


def sample_centered_cube(
    grid,
    rotation_matrices,
    shifts,
    cube_side=10,
    align_corners=True,
):
    assert len(grid.shape) == 5
    assert grid.shape[2] == grid.shape[3] == grid.shape[4]
    bz, cz, sz = grid.shape[0], grid.shape[1], grid.shape[2]
    scale_mult = (
        (
            torch.Tensor(
                [
                    cube_side,
                    cube_side,
                    cube_side,
                ]
            )
            - 1
        )
        / (sz - 1)
    ).to(grid.device)
    rotation_matrices = rotation_matrices * scale_mult[None, ..., None]
    center_shift_vector = torch.Tensor(
        [[cube_side // 2, cube_side // 2, cube_side // 2]]
    ).to(shifts.device)
    shifts = (
        grid_sampler_normalize(
            shifts - center_shift_vector, sz, align_corners=align_corners
        )
        + scale_mult[None]
    )
    affine_matrix = get_affine(rotation_matrices, shifts)
    cc = F.affine_grid(
        affine_matrix,
        (
            bz,
            cz,
        )
        + 3 * (cube_side,),
        align_corners=align_corners,
    )
    return F.grid_sample(grid.detach(), cc, align_corners=align_corners)


def sample_centered_rectangle_along_vector(
    batch_grids,
    batch_vectors,
    batch_origin_points,
    rectangle_length=10,
    rectangle_width=3,
    marginalization_dims=None,
):
    if not isinstance(batch_grids, list):
        batch_grids = [batch_grids]
        batch_vectors = [batch_vectors]
        batch_origin_points = [batch_origin_points]
    output = []
    for (grid, vectors, origin_points) in zip(
        batch_grids, batch_vectors, batch_origin_points
    ):
        rotation_matrices = get_z_to_w_rotation_matrix(vectors)
        rectangle = sample_centered_rectangle(
            grid,
            rotation_matrices.to(grid.device),
            origin_points.to(grid.device),
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
        )
        if marginalization_dims is not None:
            rectangle = rectangle.sum(dim=marginalization_dims)
        output.append(rectangle)
    output = torch.cat(output, dim=0)
    return output


def sample_centered_rectangle_rot_matrix(
    batch_grids,
    batch_rot_matrices,
    batch_origin_points,
    rectangle_length=10,
    rectangle_width=3,
    marginalization_dims=None,
):
    if not isinstance(batch_grids, list):
        batch_grids = [batch_grids]
        batch_rot_matrices = [batch_rot_matrices]
        batch_origin_points = [batch_origin_points]
    output = []
    for (grid, rotation_matrices, origin_points) in zip(
        batch_grids, batch_rot_matrices, batch_origin_points
    ):
        rectangle = sample_centered_rectangle(
            grid,
            rotation_matrices.to(grid.device),
            origin_points.to(grid.device),
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
        )
        if marginalization_dims is not None:
            rectangle = rectangle.sum(dim=marginalization_dims)
        output.append(rectangle)
    output = torch.cat(output, dim=0)
    return output


def sample_centered_cube_rot_matrix(
    batch_grids,
    batch_rot_matrices,
    batch_origin_points,
    cube_side=10,
    marginalization_dims=None,
):
    if not isinstance(batch_grids, list):
        batch_grids = [batch_grids]
        batch_rot_matrices = [batch_rot_matrices]
        batch_origin_points = [batch_origin_points]
    output = []
    for (grid, rotation_matrices, origin_points) in zip(
        batch_grids, batch_rot_matrices, batch_origin_points
    ):
        cube = sample_centered_cube(
            grid,
            rotation_matrices.to(grid.device),
            origin_points.to(grid.device),
            cube_side=cube_side,
        )
        if marginalization_dims is not None:
            cube = cube.sum(dim=marginalization_dims)
        output.append(cube)
    output = torch.cat(output, dim=0)
    return output


def mask_grid_by_grid(grid_fn, masking_grid_fn, output_fn, threshold=0.01):
    mrc_data = load_mrc(grid_fn, False)
    mask_data = load_mrc(masking_grid_fn, False)

    new_grid = np.zeros_like(mrc_data.grid)
    new_grid[mask_data.grid > threshold] = mrc_data.grid[mask_data.grid > threshold]

    save_mrc(new_grid, mrc_data.voxel_size, mrc_data.global_origin, output_fn)


def multiply_grid_by_grid(grid_fn, masking_grid_fn, output_fn):
    mrc_data = load_mrc(grid_fn, False)
    mask_data = load_mrc(masking_grid_fn, False)

    new_grid = np.zeros_like(mrc_data.grid)
    new_grid[:] = mrc_data.grid * mask_data.grid

    save_mrc(new_grid, mrc_data.voxel_size, mrc_data.global_origin, output_fn)


def apply_bfactor_to_map(
    grid: np.ndarray, voxel_size: float, bfactor: float
) -> np.ndarray:
    grid_ft = np.fft.rfftn(np.fft.fftshift(grid))
    spectral_radius = get_fourier_shells(grid_ft)
    ori_size = grid.shape[0]

    res = spectral_radius / (ori_size * voxel_size)
    scale_spectrum = np.exp(-bfactor / 4 * np.square(res))
    grid_ft *= scale_spectrum

    grid = np.fft.ifftshift(np.fft.irfftn(grid_ft))
    return grid


def get_spherical_mask(grid: np.ndarray) -> np.ndarray:
    ls = np.linspace(-grid.shape[0] // 2, grid.shape[0] // 2, grid.shape[0])
    r = np.stack(np.meshgrid(ls, ls, ls, indexing="ij"), -1)
    r = np.linalg.norm(r, ord=2, axis=-1)
    mask = np.zeros_like(grid, dtype=bool)
    mask[r < (grid.shape[0] / 2 + 1)] = True
    return mask


def apply_lowpass_filter_to_map(
    grid: np.ndarray,
    voxel_size: float,
    lowpass_ang: float,
    filter_edge_width: int = 2,
    use_cosine_kernel: bool = True,
) -> np.ndarray:
    grid_ft = np.fft.rfftn(np.fft.fftshift(grid))
    spectral_radius = get_fourier_shells(grid_ft)
    ori_size = grid.shape[0]

    ires_filter = round((ori_size * voxel_size) / lowpass_ang)
    filter_edge_halfwidth = filter_edge_width // 2

    edge_low = max(0, (ires_filter - filter_edge_halfwidth) / ori_size)
    edge_high = min(grid_ft.shape[0], (ires_filter + filter_edge_halfwidth) / ori_size)
    edge_width = edge_high - edge_low

    res = spectral_radius / ori_size
    scale_spectrum = np.zeros_like(res)
    scale_spectrum[res < edge_low] = 1

    if use_cosine_kernel:
        scale_spectrum[(res >= edge_low) & (res <= edge_high)] = 0.5 + 0.5 * np.cos(
            np.pi
            * (res[(res >= edge_low) & (res <= edge_high)] - edge_low)
            / edge_width
        )

    grid_ft *= scale_spectrum
    grid = np.fft.ifftshift(np.fft.irfftn(grid_ft))
    return grid


def extend_edge(binary_mask, edge, kernel, ramp):
    smooth_mask = np.copy(binary_mask)
    prev_mask = np.copy(binary_mask).astype(bool)
    for i in range(edge):
        mask = convolve(prev_mask.astype(np.float32), kernel) > 0
        skin = mask & ~prev_mask
        prev_mask = mask
        smooth_mask[skin] = ramp[i]
    return smooth_mask


def get_mask_from_grid(
    grid: np.ndarray,
    ini_threshold: float = 0.01,
    extend_inimask: int = 3,
    width_soft_edge: int = 3,
) -> np.ndarray:
    binary_mask = np.zeros_like(grid, dtype=np.float32)
    binary_mask[grid > ini_threshold] = 1

    kernel = np.zeros((3, 3, 3))
    kernel[:, 1, 1] = 1
    kernel[1, :, 1] = 1
    kernel[1, 1, :] = 1

    if extend_inimask > 0:
        binary_mask = extend_edge(
            binary_mask,
            extend_inimask,
            kernel,
            np.ones((extend_inimask,)),
        )

    if width_soft_edge > 0:
        binary_mask = extend_edge(
            binary_mask,
            width_soft_edge,
            kernel,
            ramp=np.cos(np.linspace(0, np.pi, width_soft_edge)) * 0.5 + 0.5,
        )

    return binary_mask


def get_auto_mask(grid: np.ndarray, voxel_size: float) -> np.ndarray:
    lowpass_grid = apply_lowpass_filter_to_map(grid, voxel_size, 15)
    s_mask = get_spherical_mask(lowpass_grid)
    lowpass_grid[~s_mask] = 0

    threshold = np.quantile(lowpass_grid[lowpass_grid > 0], q=0.97)
    extend_mask_value = max(5, round(0.01 * grid.shape[0]) + 1)
    mask_grid = get_mask_from_grid(
        lowpass_grid,
        threshold,
        extend_inimask=extend_mask_value,
        width_soft_edge=extend_mask_value,
    )
    return mask_grid


def generate_auto_mask(input_file, output_file):
    mrc_data = load_mrc(input_file)
    mask_grid = get_auto_mask(mrc_data.grid, mrc_data.voxel_size)
    save_mrc(mask_grid, mrc_data.voxel_size, mrc_data.global_origin, output_file)


def make_model_angelo_grid(grid, voxel_size, global_origin, target_voxel_size=1.5):
    grid, shift, _ = make_cubic(grid)
    global_origin[0] -= shift[0] * voxel_size
    global_origin[1] -= shift[1] * voxel_size
    global_origin[2] -= shift[2] * voxel_size

    grid, voxel_size = normalize_voxel_size(
        grid, voxel_size, target_voxel_size=target_voxel_size
    )
    return MRCObject(grid, voxel_size, global_origin)


def crop_center_along_z(grid: np.ndarray, num_voxels: int) -> np.ndarray:
    down_voxels = num_voxels // 2
    up_voxels = num_voxels - down_voxels
    center = grid.shape[0] // 2
    new_grid = np.zeros_like(grid)
    new_grid[center - down_voxels : center + up_voxels, :, :] = grid[
        center - down_voxels : center + up_voxels, :, :
    ]
    return new_grid


def get_local_std(grid: torch.Tensor, kernel_size: int = 10) -> torch.Tensor:
    assert len(grid.shape) == 5
    grid_mean = grid.clone()
    grid_squared = grid.square()
    kernel = torch.exp(-torch.linspace(-1.5, 1.5, 2 * kernel_size + 1).square()).to(
        grid.device
    )
    kernel = kernel[None, None, :, None, None] / kernel.sum()
    for i in range(3):
        grid_mean = einops.rearrange(grid_mean, "b c x y z -> b c z x y")
        grid_mean = F.conv3d(grid_mean, kernel, padding="same")
        grid_squared = einops.rearrange(grid_squared, "b c x y z -> b c z x y")
        grid_squared = F.conv3d(grid_squared, kernel, padding="same")
    return grid_squared.sub_(grid_mean.square_()).relu_().sqrt_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mrc_fn", type=str)
    args = parser.parse_args()

    grid, voxel_size, _ = load_mrc(args.mrc_fn)

    x, sparsity = plot_grid_sparsity(grid)
    d_sparsity = sparsity[1:] - sparsity[:-1]
    dd_sparsity = d_sparsity[1:] - d_sparsity[:-1]
    plt.plot(x, sparsity / np.mean(sparsity), "k")

    plt.show()
