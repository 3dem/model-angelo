import glob
import os
import shutil

import accelerate
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from model_angelo.c_alpha.arguments import parse_arguments
from model_angelo.c_alpha.dataset import CASDataset
from model_angelo.c_alpha.train import train
from model_angelo.utils.misc_utils import make_empty_dirs, setup_logger
from model_angelo.utils.summary_writer_wrapper import SummaryWriterWrapper
from model_angelo.utils.torch_utils import (
    checkpoint_load_latest,
    count_parameters,
    get_model_from_file,
    no_weight_decay_groups,
)

if __name__ == "__main__":
    args = parse_arguments()
    accelerator = accelerate.Accelerator()

    if accelerator.is_main_process:
        if args.dont_load and os.path.isdir(args.log_dir):
            shutil.rmtree(os.path.join(args.log_dir, "summary"), ignore_errors=True)
            for chkpt in glob.glob(os.path.join(args.log_dir, "chkpt_*")):
                os.remove(chkpt)
            os.remove(os.path.join(args.log_dir, "train.log"))
        if not os.path.isdir(args.log_dir):
            make_empty_dirs(args.log_dir)

    logger = setup_logger(os.path.join(args.log_dir, "train.log"))

    dataset_kwargs = {
        "positional_encoding_dim": None
        if args.positional_encoding_dim == 0
        else args.positional_encoding_dim,
        "num_cache_elements": 5,
        "do_data_augmentation": not args.dont_use_data_augmentation,
        "do_bfactor_augmentation": True,
        "global_normalization": args.use_global_normalization,
        "log_path": os.path.join(args.log_dir, "train.log"),
        "max_noise": args.max_noise,
    }

    dataset = CASDataset(
        args.dataset_list, args.box_size, shuffle_idx=True, **dataset_kwargs
    )
    train_data_loader = DataLoader(
        dataset,
        num_workers=2 * args.batch_size,
        batch_size=args.batch_size,
    )

    dataset_kwargs["do_data_augmentation"] = False
    dataset_kwargs["do_bfactor_augmentation"] = False

    valid_dataset = CASDataset(
        args.valid_dataset_list, args.box_size, dataset_length=20, **dataset_kwargs
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        num_workers=args.batch_size,
        batch_size=args.batch_size,
    )

    module = get_model_from_file(os.path.join(args.log_dir, "model.py"))

    if accelerator.is_main_process:
        logger.info(f"Model has {count_parameters(module)} parameters", accelerator)
        logger.info(f"Running training with args: {args}")

    opt = AdamW(
        no_weight_decay_groups(module), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.use_cosine_annealing:
        learning_rate_scheduler = CosineAnnealingLR(
            opt,
            T_max=args.num_steps,
        )
    else:
        learning_rate_scheduler = LambdaLR(
            opt, lr_lambda=lambda step: 0.9 ** (step // 20000)
        )

    step = 0

    if not args.dont_load:
        step = checkpoint_load_latest(
            args.log_dir,
            torch.device("cpu"),
            match_model=args.match_model,
            model=module,
            optimizer=opt,
            # learning_rate_scheduler=learning_rate_scheduler,
        )
        if accelerator.is_main_process:
            logger.info(f"Loaded module from step {step}", accelerator)

    summary_writer = None
    if accelerator.is_main_process:
        summary_writer = SummaryWriterWrapper(
            os.path.join(args.log_dir, "summary"), debug=args.debug
        )

    module, opt, train_data_loader, valid_data_loader = accelerator.prepare(
        module, opt, train_data_loader, valid_data_loader
    )

    train(
        logger=logger,
        step=step,
        module=module,
        accelerator=accelerator,
        train_dataloader=train_data_loader,
        valid_dataloader=valid_data_loader,
        opt=opt,
        learning_rate_schedule=learning_rate_scheduler,
        args=args,
        summary_writer_wrapper=summary_writer,
    )
