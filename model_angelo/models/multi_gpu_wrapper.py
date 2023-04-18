import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from typing import List
from collections import namedtuple
import os

from model_angelo.utils.misc_utils import filter_useless_warnings
from model_angelo.utils.torch_utils import get_model_from_file, compile_if_possible

# Currently hard-coded, can change later
if os.environ.get("MASTER_ADDR") is None:
    os.environ["MASTER_ADDR"] = "localhost"
if os.environ.get("MASTER_PORT") is None:
    os.environ["MASTER_PORT"] = "29500"


InferenceData = namedtuple(
    "InferenceData",
    ["data", "status"]
)


def is_iterable(object) -> bool:
    try:
        iter(object)
        return True
    except:
        return False


def send_dict_to_device(dictionary, device: str):
    for key in dictionary:
        if torch.is_tensor(dictionary[key]):
            dictionary[key] = dictionary[key].to(device)
        elif is_iterable(dictionary[key]):
            dictionary[key] = [x.to(device) for x in dictionary[key]]
    return dictionary


def cast_dict_to_half(dictionary):
    for key in dictionary:
        if torch.is_tensor(dictionary[key]):
            if dictionary[key].dtype == torch.float32:
                dictionary[key] = dictionary[key].to(torch.float16)
        elif is_iterable(dictionary[key]):
            dictionary[key] = [x.to(torch.float16) for x in dictionary[key] if x.dtype == torch.float32]
    return dictionary


def cast_dict_to_full(dictionary):
    for key in dictionary:
        if torch.is_tensor(dictionary[key]):
            if dictionary[key].dtype == torch.float16:
                dictionary[key] = dictionary[key].to(torch.float32)
        elif is_iterable(dictionary[key]):
            dictionary[key] = [x.to(torch.float32) for x in dictionary[key] if x.dtype == torch.float16]
    return dictionary

def init_model(model_definition_path: str, state_dict_path: str, device: str) -> nn.Module:
    model = get_model_from_file(model_definition_path).eval()
    checkpoint = torch.load(state_dict_path, map_location="cpu")
    if "model" not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint["model"])
    # model = compile_if_possible(model)
    model.to(device)
    return model

def run_inference(
        rank_id: int,
        model_definition_path: str,
        state_dict_path: str,
        devices: List[str],
        world_size: int,
        input_queues: List[mp.Queue],
        output_queues: List[mp.Queue],
        dtype: torch.dtype = torch.float32,
):
    device = devices[rank_id]
    input_queue = input_queues[rank_id]
    output_queue = output_queues[rank_id]
    model = init_model(model_definition_path, state_dict_path, device)

    dist.init_process_group("gloo", rank=rank_id, world_size=world_size)
    filter_useless_warnings()

    while True:
        with torch.no_grad():
            try:
                inference_data = input_queue.get()
                if inference_data.status != 1:
                    break
                with torch.cuda.amp.autocast(dtype=dtype):
                    output = model(**inference_data.data)
                output = output.to("cpu").to(torch.float32)
                output_queue.put(output)
            except Exception as e:
                output_queue.put(None)
                raise e


class MultiGPUWrapper(nn.Module):
    def __init__(
            self,
            model_definition_path: str,
            state_dict_path: str,
            devices: List[str],
            fp16: bool = False
    ):
        super().__init__()
        self.proc_ctx = None
        self.input_queues = []
        self.output_queues = []
        self.world_size = len(devices)
        self.devices = devices
        self.dtype = torch.float32 if not fp16 else torch.float16

        if self.world_size > 1:
            torch.multiprocessing.set_start_method('spawn', force=True)

            self.input_queues, self.output_queues = [], []
            for _ in range(self.world_size):
                self.input_queues.append(mp.Queue())
                self.output_queues.append(mp.Queue())

            self.proc_ctx = mp.spawn(
                run_inference,
                args=(
                    model_definition_path,
                    state_dict_path,
                    devices,
                    self.world_size,
                    self.input_queues,
                    self.output_queues,
                    self.dtype
                ),
                nprocs=self.world_size,
                join=False,
            )
        else:
            self.model = init_model(model_definition_path, state_dict_path, devices[0])

    def forward(self, data_list: List) -> List:
        output_list = []
        for i, data in enumerate(data_list):
            device = self.devices[i]
            if self.dtype == torch.float16:
                data = cast_dict_to_half(data)
            if self.world_size > 1:
                input_queue = self.input_queues[i]
                input_queue.put(
                    InferenceData(data=send_dict_to_device(data, device), status=1)
                )
            else:
                with torch.cuda.amp.autocast(dtype=self.dtype), torch.no_grad():
                    output_list.append(
                        self.model(**send_dict_to_device(data, device)).to("cpu").to(torch.float32)
                    )
        if self.world_size > 1:
            for output_queue, _ in zip(self.output_queues, data_list):
                output_list.append(output_queue.get())
        return output_list


    def __del__(self):
        if self.world_size > 1:
            for input_queue in self.input_queues:
                try:
                    input_queue.put(InferenceData(data=None, status=0))
                except:
                    pass
            for input_queue, output_queue in zip(self.input_queues, self.output_queues):
                input_queue.close()
                input_queue.join_thread()
                output_queue.close()
                output_queue.join_thread()
            self.proc_ctx.join()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.__del__()

