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


def run_inference(
        model: nn.Module,
        device: str,
        rank_id: int,
        world_size: int,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
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
                output = model(inference_data.data)
                output = output.to("cpu")
                output_queue.put(output)
            except:
                output_queue.put(None)
                return


class MultiGPUWrapper(nn.Module):
    def __init__(self, model: nn.Module, devices: List[str]):
        super().__init__()
        self.processes = []
        self.input_queues = []
        self.output_queues = []
        self.world_size = len(devices)
        self.devices = devices

        model.share_memory()
        
        for rank_id, device in enumerate(devices):
            self.input_queues.append(mp.Queue())
            self.output_queues.append(mp.Queue())
            self.processes.append(
                mp.Process(target=run_inference, args=(model, device, rank_id, self.world_size, self.input_queues[-1], self.output_queues[-1]))
            )
            self.processes[-1].start()

    def forward(self, data_list: List) -> List:
        for input_queue, device, data in zip(self.input_queues, self.devices, data_list):
            input_queue.put(
                InferenceData(data=data.to(device), status=1)
            )
        output_list = []
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

