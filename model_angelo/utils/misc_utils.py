import os
import pickle
import sys
from typing import List
import psutil


def batch_iterator(iterator, batch_size):
    if len(iterator) <= batch_size:
        return [iterator]

    output = []
    i = 0

    while (len(iterator) - i) > batch_size:
        output.append(iterator[i : i + batch_size])
        i += batch_size

    output.append(iterator[i:])
    return output


def make_empty_dirs(log_dir):
    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir, "coordinates"))
    os.makedirs(os.path.join(log_dir, "summary"))


def accelerator_print(string, accelerator):
    if accelerator.is_main_process:
        print(string)


def flatten_dict(dictionary: dict, level: List = []) -> dict:
    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten_dict(val, level + [str(key)]))
        else:
            tmp_dict[".".join(level + [str(key)])] = val
    return tmp_dict


def unflatten_dict(dictionary: dict, to_int: bool = True) -> dict:
    result_dict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        if to_int:
            parts = [p if not p.isnumeric() else int(p) for p in parts]
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result_dict


def setup_logger(log_path: str):
    from loguru import logger

    try:
        logger.remove(handler_id=0)  # Remove pre-configured sink to sys.stderror
    except ValueError:
        pass

    logger.add(
        log_path,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        enqueue=True,
        diagnose=True,
    )
    return logger


def pickle_dump(obj: object, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file_path: str) -> object:
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def assertion_check(if_statement: bool, failure_message: str = ""):
    assert if_statement, failure_message


class FileHandle:
    def __init__(self, print_fn):
        self.print_fn = print_fn

    def write(self, *args, **kwargs):
        self.print_fn(*args, **kwargs)

    def flush(self, *args, **kwargs):
        pass


def filter_useless_warnings():
    import warnings

    warnings.filterwarnings("ignore", ".*nn\.functional\.upsample is deprecated.*")
    warnings.filterwarnings("ignore", ".*none of the inputs have requires_grad.*")
    warnings.filterwarnings("ignore", ".*with given element none.*")
    warnings.filterwarnings("ignore", ".*invalid value encountered in true\_divide.*")


def get_esm_model(esm_model_name):
    import esm

    return getattr(esm.pretrained, esm_model_name)()


class Args(object):  # Generic container for arguments
    def __init__(self, kwarg_dict):
        for (k, v) in kwarg_dict.items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)


def is_relion_abort(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, "RELION_JOB_ABORT_NOW"))


def write_relion_job_exit_status(
    directory: str, status: str, pipeline_control: str = "",
):
    if pipeline_control != "":
        open(os.path.join(directory, f"RELION_JOB_EXIT_{status}"), "a").close()
    elif status == "FAILURE":
        sys.exit(1)


def abort_if_relion_abort(directory: str):
    if is_relion_abort(directory):
        write_relion_job_exit_status(directory, "ABORTED")
        print("Aborting now...")
        sys.exit(1)

def check_available_memory():
    mem = psutil.virtual_memory().available >> 30
    assertion_check(
        mem > 10,
        "Not enough memory available. Please allocate at least 10GB of memory."
    )
