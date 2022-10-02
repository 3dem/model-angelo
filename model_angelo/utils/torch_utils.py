import glob
import importlib.util
import os
import shutil
import stat
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model_angelo.models.common_modules import SpatialAvg, SpatialMax
from model_angelo.utils.misc_utils import flatten_dict, unflatten_dict


def expand_as(x, y):
    n = len(x.shape)
    m = len(y.shape)

    assert m > n

    ones = (1,) * (m - n)

    return x.reshape(*x.shape, *ones)


def checkpoint_save(step_num, log_dir, accelerator, prefix="", **kwargs):
    for v in kwargs.values():
        if isinstance(v, torch.nn.Module):
            v.eval()

    path = os.path.join(log_dir, prefix + f"chkpt_{step_num}.torch")

    kwargs["step_num"] = step_num

    accelerator.save(
        dict(
            [
                (k, v.state_dict())
                if (
                    isinstance(v, torch.nn.Module)
                    or isinstance(v, torch.optim.Optimizer)
                    or isinstance(v, torch.optim.lr_scheduler._LRScheduler)
                )
                else (k, v)
                for (k, v) in kwargs.items()
            ]
        ),
        path,
    )


def init_optimizer_with_gradients(model: nn.Module, optimizer: torch.optim.Optimizer):
    for p in model.parameters():
        p.grad = 1e-32 * torch.ones_like(p, device=p.device)
    optimizer.step()
    optimizer.zero_grad()


def checkpoint_load_latest(
    log_dir: str, device: torch.DeviceObjType, match_model: bool = True, **kwargs
) -> int:
    checkpoints = glob.glob(os.path.join(log_dir, "chkpt_*"))
    if len(checkpoints) == 0:
        return 0
    checkpoints = [(x, int(x.split("_")[-1].split(".")[0])) for x in checkpoints]
    checkpoints = sorted(checkpoints, key=lambda x: x[1])

    checkpoint_to_load, step_num = checkpoints[-1]

    state_dicts = torch.load(checkpoint_to_load, map_location=device)

    if match_model:
        warnings.warn(
            "In checkpoint_load_latest, match_model is set to True. "
            "This means that loading weights is only proceeding on the basis of best efforts. "
            "Please ensure this results in what you need."
        )
    for (k, v) in kwargs.items():
        if isinstance(v, nn.Module):
            v.load_state_dict(state_dicts[k], strict=not match_model)
        elif hasattr(v, "load_state_dict") and not match_model:
            v.load_state_dict(state_dicts[k])

    return step_num


def load_state_dict_to_match_model(model: nn.Module, state_dict: OrderedDict) -> None:
    model_state_dict = model.state_dict()
    updated_state_dict = OrderedDict()

    for key, parameter in state_dict.items():
        if key in model_state_dict:
            if parameter.shape == model_state_dict[key].shape:
                updated_state_dict[key] = parameter

    updated_state_dict.update(model_state_dict)
    model.load_state_dict(updated_state_dict)


def load_state_dict_to_match_optimizer(
    optimizer: torch.optim.Optimizer, state_dict: OrderedDict
) -> None:
    optimizer_state_dict = flatten_dict(optimizer.state_dict())
    state_dict = flatten_dict(state_dict)
    updated_state_dict = OrderedDict()

    for key, parameter in state_dict.items():
        if key in optimizer_state_dict:
            if hasattr(parameter, "shape"):
                if parameter.shape == optimizer_state_dict[key].shape:
                    updated_state_dict[key] = parameter
            else:
                updated_state_dict[key] = parameter

    updated_state_dict.update(optimizer_state_dict)
    optimizer.load_state_dict(unflatten_dict(updated_state_dict))


def delete_data(log_dir):
    checkpoints = glob.glob(os.path.join(log_dir, "chkpt_*"))
    for checkpoint in checkpoints:
        os.remove(checkpoint)

    if os.path.isfile(os.path.join(log_dir, "train.log")):
        os.remove(os.path.join(log_dir, "train.log"))

    shutil.rmtree(os.path.join(log_dir, "tensorboard"))
    shutil.rmtree(os.path.join(log_dir, "volumes"))


def apply_function_to_state_dict(state_dict, lambda_fn):
    return OrderedDict([(lambda_fn(k), v) for (k, v) in state_dict.items()])


def sgd_functional(
    grad, t, grad_last_step=None, b_last_step=None, mu=0.9, tau=0, nesterov=False
):
    assert mu > 0
    if t > 1:
        b_t = mu * b_last_step + (1 - tau) * grad
    else:
        b_t = grad
    if nesterov:
        grad = grad_last_step + mu * b_t
    else:
        grad = b_t
    return grad, b_t


def get_activation_function(string):
    acts = {
        "sigmoid": torch.sigmoid,
        "relu": torch.nn.ReLU(),
        "gelu": torch.nn.GELU(),
        "tanh": torch.tanh,
        "swish": lambda x: x * torch.sigmoid(x),
        "sin": torch.sin,
    }
    return acts[string]


def get_activation_class(string):
    acts = {
        "sigmoid": torch.nn.Sigmoid,
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }
    return acts[string]


def get_normalization_class(string):
    norms = {
        "batch": torch.nn.BatchNorm3d,
        "instance": torch.nn.InstanceNorm3d,
        "none": torch.nn.Identity,
    }
    return norms[string]


def get_pooling_cls(string):
    pooling = {
        "avg": torch.nn.AvgPool3d,
        "max": torch.nn.MaxPool3d,
    }
    return pooling[string]


def get_spatial_cls(string):
    spatial = {
        "avg": SpatialAvg,
        "max": SpatialMax,
    }
    return spatial[string]


def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def freeze_network_except_last_layer(network: nn.Module):
    for name, p in network.named_parameters():
        if "last_layers" not in name:
            p.requires_grad = False


def get_last_layers_parameters(network: nn.Module) -> List[nn.parameter.Parameter]:
    output = []
    for name, p in network.named_parameters():
        if "last_layers" in name:
            output.append(p)

    return output


def no_weight_decay_groups(network: nn.Module) -> List[Dict]:
    weight_decay = []
    no_weight_decay = []

    for n, p in network.named_parameters():
        if (n.endswith(".bias")) or ("bn" in n) or ("layer_norm" in n):
            no_weight_decay.append(p)
        else:
            weight_decay.append(p)

    return [
        {"params": weight_decay},
        {
            "params": no_weight_decay,
            "weight_decay": 0,
        },
    ]


def sum_pool3d(x, kernel_size=1, stride=1, padding=0):
    return (kernel_size ** 3) * F.avg_pool3d(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )


def inverse_sigmoid(x, offset=0.05):
    return torch.log(x + offset) - torch.log(1 + offset - x)


@torch.no_grad()
def accuracy(out, targets, topk=1):
    if topk == 1:
        _, pred = torch.max(out, 1)
        acc = torch.mean(torch.eq(pred, targets).float())
    else:
        _, pred = out.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
        acc = correct[:topk].reshape(-1).float().sum(0) / out.size(0)
    return acc


@torch.no_grad()
def binary_accuracy(out, targets, threshold=0.5):
    pred = torch.where(out.reshape(-1, 1) < threshold, 0, 1)
    acc = torch.eq(pred, targets.reshape(-1, 1)).float().mean()
    return acc


@torch.no_grad()
def binary_accuracy_report(out, targets, threshold=0.5):
    pred, targets = torch.where(out.reshape(-1, 1) < threshold, 0, 1), torch.where(
        targets.reshape(-1, 1) < threshold, 0, 1
    )
    acc = torch.eq(pred, targets).float().mean().item()
    precision = torch.eq(pred[pred == 1], targets[pred == 1]).float().mean().item()
    recall = torch.eq(pred[targets == 1], targets[targets == 1]).float().mean().item()
    return acc, precision, recall


def get_batch_slices(
    num_total: int,
    batch_size: int,
) -> List[List[int]]:
    if num_total <= batch_size:
        return [list(range(num_total))]

    num_batches = num_total // batch_size
    batches = [
        list(range(i * batch_size, (i + 1) * batch_size)) for i in range(num_batches)
    ]
    if num_total % batch_size > 0:
        batches += [list(range(num_batches * batch_size, num_total))]
    return batches


def get_model_from_file(file_path: str) -> nn.Module:
    spec = importlib.util.spec_from_file_location("network", file_path)
    network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(network)
    return network.Model()


def get_batches_to_idx(idx_to_batches: torch.Tensor) -> List[torch.Tensor]:
    assert len(idx_to_batches.shape) == 1
    max_batch_num = idx_to_batches.max().item() + 1
    idxs = torch.arange(0, len(idx_to_batches), dtype=int, device=idx_to_batches.device)
    return [idxs[idx_to_batches == i] for i in range(max_batch_num)]


def linear_warmup_exponential_decay(
    num_warmup_steps=2000,
    decay_rate=0.9,
    decay_ratio=30000,
    min_lr=0.0,
):
    def learning_rate_fn(step):
        if step < num_warmup_steps:
            lr_update = step / num_warmup_steps
        else:
            lr_update = decay_rate ** ((step - num_warmup_steps) // decay_ratio)
        return max(min_lr, lr_update)

    return learning_rate_fn


def pad_sequences(sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    masks = [torch.ones(*s.shape[:-1], device=s.device) for s in sequences]
    padded_sequence = pad_sequence(sequences, batch_first=True)
    padded_masks = pad_sequence(masks, batch_first=True)
    return padded_sequence, padded_masks


def padded_sequence_softmax(
    padded_sequence_values: torch.Tensor,
    padded_mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    padded_softmax = torch.softmax(padded_sequence_values, dim=dim)
    padded_softmax = padded_softmax * padded_mask  # Mask out padded values
    padded_softmax = (
        padded_softmax / (padded_softmax.sum(dim=dim, keepdim=True) + eps).detach()
    )  # Renormalize
    return padded_softmax


def assert_all_params_have_grads(module):
    for n, p in module.named_parameters():
        assert p.grad is not None, f"The parameter {n} has no gradient"
        assert (p.grad != 0).any(), f"The parameter {n} has zero gradients"


def check_grad_health(module):
    grads = np.array(
        [p.grad.square().mean().sqrt().item() for (_, p) in module.named_parameters()],
        dtype=np.float32,
    )
    idx_to_name = np.array([n for (n, _) in module.named_parameters()], dtype=object)
    mean, std = np.mean(grads), np.std(grads)
    grads_normalized = (grads - mean) / std

    report = {}
    report["mean"] = mean
    report["std"] = std

    smallest_10 = np.argpartition(grads, kth=10)[:10]

    abnormal_grads = grads_normalized > 3
    for n, g in zip(idx_to_name[abnormal_grads], grads[abnormal_grads]):
        report[n] = g
    for n, g in zip(idx_to_name[smallest_10], grads[smallest_10]):
        report[n] = g

    return report


def get_module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def mean_over_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    maskf = mask.float()
    return x.mul(maskf).sum().div_(maskf.sum())


def stable_distance(
    x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-6
) -> torch.Tensor:
    return x.sub(y).square_().sum(dim=dim).add_(eps).sqrt_()


def masked_stable_distance_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert (
        x.shape == y.shape
    ), f"Shape of x and y must be the same. Got: x: {x.shape}, y: {y.shape}"
    assert len(mask.shape) == len(x.shape) - 1, (
        f"Number of dimensions for mask and x don't match. "
        f"Got mask: {mask.shape}, x: {x.shape}"
    )
    assert mask.shape == x.shape[:-1], (
        f"Shape of mask and x is not compatible. "
        f"Got mask: {mask.shape}, x: {x.shape}"
    )
    dist = stable_distance(x=x, y=y, dim=dim, eps=eps)
    loss = dist.mul(mask).sum().div(mask.sum())
    return loss


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def shared_cat(args, dim=0, is_torch=True) -> Union[torch.Tensor, np.ndarray]:
    if is_torch:
        return torch.cat(args, dim=dim)
    else:
        return np.concatenate(args, axis=dim)


def one_hot(index: int, num_classes: int, device: str = "cpu") -> torch.Tensor:
    return F.one_hot(torch.LongTensor([index]).to(device=device), num_classes)[0]


def is_ndarray(x) -> bool:
    return isinstance(x, np.ndarray)


def download_and_install_model(bundle_name: str) -> str:
    dest = os.path.join(
        torch.hub.get_dir(),
        "checkpoints",
        "model_angelo",
        bundle_name
    )
    if os.path.isfile(os.path.join(dest, "success.txt")):
        return dest

    print(f"Setting up bundle with name: {bundle_name} for the first time.")
    import zipfile
    os.makedirs(os.path.split(dest)[0], exist_ok=True)
    torch.hub.download_url_to_file(
        f"ftp://ftp.mrc-lmb.cam.ac.uk/pub/scheres/modelangelo/{bundle_name}.zip",
        dest + ".zip"
    )

    with zipfile.ZipFile(dest + ".zip", "r") as zip_object:
        zip_object.extractall(path=os.path.split(dest)[0])

    os.remove(dest + ".zip")

    with open(os.path.join(dest, "success.txt"), "w") as f:
        f.write("Successfully downloaded model")

    print(f"Bundle {bundle_name} successfully installed.")

    return dest


def check_permissions_exceed(file_name: str, permissions) -> bool:
    file_permissions = oct(os.stat(file_name).st_mode)
    permissions_str = oct(permissions)

    if len(file_permissions) < len(permissions_str) or min(len(file_permissions), len(permissions_str)) < 3:
        return False

    file_permissions, permissions_str = file_permissions[-3:], permissions_str[-3:]
    for fp, p in zip(file_permissions, permissions_str):
        if int(fp) < int(p):
            return False
    return True


def download_and_install_esm_model(esm_model_name: str) -> str:
    permissions = stat.S_IROTH | stat.S_IXOTH | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
    dest_model = os.path.join(
        torch.hub.get_dir(),
        "checkpoints",
        esm_model_name + ".pt"
    )
    dest_regr = os.path.join(
        torch.hub.get_dir(),
        "checkpoints",
        esm_model_name + "-contact-regression.pt"
    )
    model_url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{esm_model_name}.pt"
    regr_url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{esm_model_name}-contact-regression.pt"
    if not os.path.isfile(dest_model):
        print(f"Setting up language model bundle with name: {esm_model_name} for the first time.")
        torch.hub.download_url_to_file(model_url, dest_model)
    if not os.path.isfile(dest_regr):
        torch.hub.download_url_to_file(regr_url, dest_regr)
    if not check_permissions_exceed(dest_model, permissions):
        os.chmod(dest_model, permissions)
    if not check_permissions_exceed(dest_regr, permissions):
        os.chmod(dest_regr, permissions)
    return dest_model


def get_device_name(device_name: str) -> str:
    if device_name is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name == "cpu":
        return "cpu"
    if device_name.startswith("cuda:"):
        return device_name
    if device_name.isnumeric():
        return f"cuda:{device_name}"
    else:
        raise RuntimeError(
            f"Device name: {device_name} not recognized. "
            f"Either do not set, set to cpu, or give a number"
        )


class ShapeError(Exception):
    pass
