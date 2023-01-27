import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from typing import List
from collections import namedtuple
import os


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


def run_inference(
        model: nn.Module,
        device: str,
        rank_id: int,
        world_size: int,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        dtype: torch.dtype = torch.float32,
):
    dist.init_process_group("gloo", rank=rank_id, world_size=world_size)
    model.eval()
    model.to(device)
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
                return


class MultiGPUWrapper(nn.Module):
    def __init__(self, model: nn.Module, devices: List[str], fp16: bool = False):
        super().__init__()
        self.processes = []
        self.input_queues = []
        self.output_queues = []
        self.world_size = len(devices)
        self.devices = devices
        self.dtype = torch.float32 if not fp16 else torch.float16

        if self.world_size > 1:
            model.share_memory()
        
            for rank_id, device in enumerate(devices):
                self.input_queues.append(mp.Queue())
                self.output_queues.append(mp.Queue())
                self.processes.append(
                    mp.Process(target=run_inference, args=(model, device, rank_id, self.world_size, self.input_queues[-1], self.output_queues[-1], self.dtype))
                )
                self.processes[-1].start()
        else:
            self.model = model.to(self.devices[0])

    def forward(self, data_list: List) -> List:
        output_list = []
        for i, data in enumerate(data_list):
            device = self.devices[i]
            if self.dtype == torch.float16:
                data = cast_dict_to_half(data)
            data = send_dict_to_device(data, device)
            if self.world_size > 1:
                input_queue = self.input_queues[i]
                input_queue.put(InferenceData(data=data, status=1))
            else:
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    output_list.append(self.model(**data).to("cpu").to(torch.float32))
        if self.world_size > 1:
            for output_queue, _ in zip(self.output_queues, data_list):
                output_list.append(output_queue.get())
        return output_list


    def __del__(self):
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
        for p in self.processes:
            p.join()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.__del__()

