import datetime
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from config.config import Config
from hexplane.dataloader import get_test_dataset, get_train_dataset
from hexplane.model import init_model
from hexplane.render.render import evaluation, evaluation_path
from hexplane.render.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def load_compressed_model(compressed_path, model_instance, device):
    """
    Loads the compressed model data into the provided model instance.

    Args:
        compressed_path (str): Path to the compressed file.
        model_instance (HexPlane): Instance of the HexPlane model.
        device (torch.device): Device to load the parameters on.

    Returns:
        HexPlane: Model instance with loaded compressed data.
    """
    model_instance.load_compressed(compressed_path, device)
    return model_instance


def render_test(cfg):
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg

    # Add these lines to define 'near_far' and 'aabb'
    near_far = test_dataset.near_far
    aabb = test_dataset.scene_bbox.to(device)

    if not os.path.exists(cfg.systems.ckpt):
        print("The checkpoint path does not exist!")
        return

    # Initialize model instance
    HexPlane, reso_cur = init_model(cfg, aabb, near_far, device)

    # Load model (compressed or full)
    if cfg.systems.load_compressed:
        compressed_model_path = cfg.systems.ckpt.replace('.pth', '_compressed.pth')
        HexPlane.load_compressed(compressed_model_path, device)
    else:
        HexPlane.load_model(cfg.systems.ckpt, device)

    logfolder = os.path.dirname(cfg.systems.ckpt)

    if cfg.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    if cfg.render_path:
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_path_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )


def reconstruction(cfg):
    if cfg.data.datasampler_type == "rays":
        train_dataset = get_train_dataset(cfg, is_stack=False)
    else:
        train_dataset = get_train_dataset(cfg, is_stack=True)
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg
    near_far = test_dataset.near_far

    # Setup logging directory
    if cfg.systems.add_timestamp:
        logfolder = f'{cfg.systems.basedir}/{cfg.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{cfg.systems.basedir}/{cfg.expname}"

    # Create log directories and save config
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))
    cfg_file = os.path.join(f"{logfolder}", "cfg.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # Initialize model
    aabb = train_dataset.scene_bbox.to(device)
    HexPlane, reso_cur = init_model(cfg, aabb, near_far, device)

    # Initialize optimizers
    grad_vars = HexPlane.get_optparam_groups(cfg.optim)  # <-- Updated Call
    optimizer = torch.optim.Adam(
        [p for g in grad_vars for p in g['params'] if p.requires_grad],
        betas=(cfg.optim.beta1, cfg.optim.beta2)
    )

    # Initialize compression optimizer if needed
    aux_optimizer = None
    if HexPlane.use_codec:
        aux_optimizer = torch.optim.Adam(
            HexPlane.codec_aux_params,
            lr=cfg.model.compression.lr_codec_aux
        )

    # Initialize trainer
    trainer = Trainer(
        HexPlane,
        cfg,
        reso_cur,
        train_dataset, 
        test_dataset,
        summary_writer,
        logfolder,
        device,
        optimizer=optimizer,
        aux_optimizer=aux_optimizer
    )

    trainer.train()

    # Save the compressed HexPlane model
    compressed_save_path = os.path.join(logfolder, f"{cfg.expname}_compressed.th")
    HexPlane.save_compressed(compressed_save_path)

    # torch.save(HexPlane, f"{logfolder}/{cfg.expname}.th")

    # Render training viewpoints.
    if cfg.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render test viewpoints.
    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/{cfg.expname}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render validation viewpoints.
    if cfg.render_path:
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/{cfg.expname}/imgs_path_all/",
            prefix="validation",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )


if __name__ == "__main__":
    # Load config file from base config, yaml and cli.
    base_cfg = OmegaConf.structured(Config())
    cli_cfg = OmegaConf.from_cli()
    base_yaml_path = base_cfg.get("config", None)
    yaml_path = cli_cfg.get("config", None)
    if yaml_path is not None:
        yaml_cfg = OmegaConf.load(yaml_path)
    elif base_yaml_path is not None:
        yaml_cfg = OmegaConf.load(base_yaml_path)
    else:
        yaml_cfg = OmegaConf.create()
    cfg = OmegaConf.merge(base_cfg, yaml_cfg, cli_cfg)  # merge configs

    # Fix Random Seed for Reproducibility.
    random.seed(cfg.systems.seed)
    np.random.seed(cfg.systems.seed)
    torch.manual_seed(cfg.systems.seed)
    torch.cuda.manual_seed(cfg.systems.seed)

    if cfg.render_only and (cfg.render_test or cfg.render_path):
        # Inference only.
        render_test(cfg)
    else:
        # Reconstruction and Inference.
        reconstruction(cfg)
