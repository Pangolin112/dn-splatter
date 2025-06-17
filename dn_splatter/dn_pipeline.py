import json
import os
import random
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import numpy as np
import open3d as o3d
import torch
import trimesh
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch.cuda.amp.grad_scaler import GradScaler

from dn_splatter.data.mushroom_utils.eval_faro import depth_eval_faro
from dn_splatter.dn_model import DNSplatterModelConfig
from dn_splatter.metrics import PDMetrics
from dn_splatter.utils import camera_utils
from dn_splatter.utils.utils import gs_render_dataset_images, ns_render_dataset_images
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    DDP,
    Model,
    VanillaPipeline,
    VanillaPipelineConfig,
    dist,
)
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE

# imports for secret view editing
import yaml
from types import SimpleNamespace
from torchvision.utils import save_image

from dn_splatter.ip2p_ptd import IP2P_PTD

from dn_splatter.ip2p_depth import InstructPix2Pix_depth

from nerfstudio.engine.callbacks import TrainingCallbackAttributes


@dataclass
class DNSplatterPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DNSplatterPipeline)
    datamanager: DataManagerConfig = field(default_factory=lambda: DataManagerConfig())
    model: ModelConfig = field(default_factory=lambda: DNSplatterModelConfig())
    experiment_name: str = "experiment"
    """Experiment name for saving metrics and rendered images to disk"""
    skip_point_metrics: bool = True
    """Skip evaluating point cloud metrics"""
    num_pd_points: int = 1_000_000
    """Total number of points to extract from train/eval renders for pointcloud reconstruction"""
    save_train_images: bool = False
    """saving train images to disc"""
    gs_steps: int = 2500
    """how many GS steps between dataset updates"""


class DNSplatterPipeline(VanillaPipeline):
    """Pipeline for convenient eval metrics across model types"""

    def __init__(
        self,
        config: DNSplatterPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)

        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz"
            in self.datamanager.train_dataparser_outputs.metadata  # type: ignore
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata[
                "points3D_xyz"
            ]  # type: ignore
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata[
                "points3D_rgb"
            ]  # type: ignore
            if "points3D_normals" in self.datamanager.train_dataparser_outputs.metadata:
                normals = self.datamanager.train_dataparser_outputs.metadata[
                    "points3D_normals"
                ]  # type: ignore
                seed_pts = (pts, pts_rgb, normals)
            else:
                seed_pts = (pts, pts_rgb)

        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])

        self.pd_metrics = PDMetrics()

        ######################################################
        # Secret view updating                              
        ######################################################
        # prepare for secret view editing
        # which image index we are editing
        self.curr_edit_idx = 0
        # whether we are doing regular GS updates or editing images
        self.makeSquentialEdits = False
        
        # whether we are doing the first sequential edits
        self.first_SequentialEdit = True

        config_path = 'config/config.yaml'
        with open(config_path, 'r') as file:
            cfg_dict = yaml.safe_load(file)
        self.config_secret = SimpleNamespace(**cfg_dict)

        self.config_secret.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        self.ip2p_ptd = IP2P_PTD(
            self.dtype, 
            self.config_secret.device, 
            prompt=self.config_secret.prompt_2, 
            t_dec=self.config_secret.t_dec, 
            image_guidance_scale=self.config_secret.image_guidance_scale_ip2p_ptd, 
            async_ahead_steps=self.config_secret.async_ahead_steps
        )

        self.text_embeddings_ip2p = self.ip2p_ptd.text_embeddings_ip2p

        self.ip2p_depth = InstructPix2Pix_depth(
            self.dtype, 
            self.config_secret.device, 
            self.config_secret.render_size, 
            self.config_secret.conditioning_scale
        )

        # secret data preparation
        secret_view_idx = self.config_secret.secret_view_idx
        self.camera_secret, self.data_secret = self.datamanager.next_train_idx(secret_view_idx)
        self.original_image_secret = self.datamanager.original_cached_train[secret_view_idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        self.depth_image_secret = self.datamanager.original_cached_train[secret_view_idx]["depth"] # [bs, h, w]
        ######################################################
    
    # add callback function to fetch the components from other parts of the training process.
    def get_training_callbacks(self, attrs: TrainingCallbackAttributes):
        # stash a reference to the Trainer
        self.trainer = attrs.trainer
        # now return whatever callbacks the base class wants
        return super().get_training_callbacks(attrs)
    

    # comment this function for 1st stage updating, __update all images at once__then randomly update 1 image__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / "images"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)
    
    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSquentialEdits = True

    #     if (not self.makeSquentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(allowed)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image_{self.config_secret.image_guidance_scale_ip2p}_edit_rate_{self.config_secret.edit_rate}.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % 20 == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image_{self.config_secret.image_guidance_scale_ip2p_ptd}_edit_rate_{self.config_secret.edit_rate}.png')
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             # also update the secret view w/o editing (important)
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #         # edited_image = self.ip2p.edit_image(
    #         #             self.text_embedding.to(self.ip2p_device),
    #         #             rendered_image.to(self.ip2p_device),
    #         #             original_image.to(self.ip2p_device),
    #         #             guidance_scale=self.config.guidance_scale,
    #         #             image_guidance_scale=self.config.image_guidance_scale,
    #         #             diffusion_steps=self.config.diffusion_steps,
    #         #             lower_bound=self.config.lower_bound,
    #         #             upper_bound=self.config.upper_bound,
    #         #         )
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image_{self.config_secret.image_guidance_scale_ip2p_ptd}.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.config_secret.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image_{self.config_secret.image_guidance_scale_ip2p_ptd}.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSquentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSquentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    # comment this function for 1st stage updating, __update 1 image every edit rate__
    def get_train_loss_dict(self, step: int):
        ######################################################
        # Secret view updating                              
        ######################################################
        ckpt_dir = self.trainer.checkpoint_dir
        image_dir = ckpt_dir / "images"
        if not image_dir.exists():
            image_dir.mkdir(parents=True, exist_ok=True)

        # print(step) # start from 30000

        # the implementation of randomly selecting an index to edit instead of update all images at once
        # generate the indexes for non-secret view editing
        all_indices = np.arange(len(self.datamanager.cached_train))
        allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

        if step % self.config_secret.edit_rate == 0:
            #----------------non-secret view editing----------------
            # randomly select an index to edit
            idx = random.choice(allowed)
            camera, data = self.datamanager.next_train_idx(idx)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)

            original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
            rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

            depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

            edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
                self.text_embeddings_ip2p.to(self.config_secret.device),
                rendered_image.to(self.dtype),
                original_image.to(self.config_secret.device).to(self.dtype),
                False, # is depth tensor
                depth_image,
                guidance_scale=self.config_secret.guidance_scale,
                image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
                diffusion_steps=self.config_secret.t_dec,
                lower_bound=self.config_secret.lower_bound,
                upper_bound=self.config_secret.upper_bound,
            )

            # resize to original image size (often not necessary)
            if (edited_image.size() != rendered_image.size()):
                edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

            # write edited image to dataloader
            edited_image = edited_image.to(original_image.dtype)
            self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
            data["image"] = edited_image.squeeze().permute(1,2,0)

            # save edited non-secret image
            if step % 50 == 0:
                image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
                save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image_{self.config_secret.image_guidance_scale_ip2p}_edit_rate_{self.config_secret.edit_rate}.png')

            loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

            #----------------secret view editing----------------
            if step % 20 == 0:
                model_outputs_secret = self.model(self.camera_secret)
                metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
                loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
                rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
                    image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
                    image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
                    depth=self.depth_image_secret,
                    lower_bound=self.config_secret.lower_bound,
                    upper_bound=self.config_secret.upper_bound
                )

                # resize to original image size (often not necessary)
                if (edited_image_secret.size() != rendered_image_secret.size()):
                    edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

                # write edited image to dataloader
                edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
                self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
                self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

                for k, v in metrics_dict_secret.items():
                    metrics_dict[f"secret_{k}"] = v
                for k, v in loss_dict_secret.items():
                    loss_dict[f"secret_{k}"] = v
                
                # save edited secret image
                if step % 50 == 0:
                    image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
                    save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image_{self.config_secret.image_guidance_scale_ip2p_ptd}_edit_rate_{self.config_secret.edit_rate}.png')
        else:
            # non-editing steps loss computing
            camera, data = self.datamanager.next_train(step)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)
            loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

            # also update the secret view w/o editing (important)
            model_outputs_secret = self.model(self.camera_secret)
            metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
            loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

            for k, v in metrics_dict_secret.items():
                metrics_dict[f"secret_{k}"] = v
            for k, v in loss_dict_secret.items():
                loss_dict[f"secret_{k}"] = v

        return model_outputs, loss_dict, metrics_dict
        ######################################################


    # comment this function for 1st stage updating, __update all images at once__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / "images"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)
    
    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0: # update also for the first step
    #         self.makeSquentialEdits = True

    #     if (not self.makeSquentialEdits):
    #         # update the non-secret views w/o editing
    #         camera, data = self.datamanager.next_train(step)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         if step % 500 == 0:
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             save_image((rendered_image).clamp(0, 1), image_dir / f'{step}_non_secret_image_{self.config_secret.image_guidance_scale_ip2p_ptd}.png')

    #         # also update the secret view w/o editing
    #         model_outputs_secret = self.model(self.camera_secret)
    #         metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #         loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #         for k, v in metrics_dict_secret.items():
    #             metrics_dict[f"secret_{k}"] = v
    #         for k, v in loss_dict_secret.items():
    #             loss_dict[f"secret_{k}"] = v

    #         if step % 500 == 0:
    #             # save the secret view image
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             image_save_secret = torch.cat([rendered_image_secret, self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image_{self.config_secret.image_guidance_scale_ip2p_ptd}.png')

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #         # edited_image = self.ip2p.edit_image(
    #         #             self.text_embedding.to(self.ip2p_device),
    #         #             rendered_image.to(self.ip2p_device),
    #         #             original_image.to(self.ip2p_device),
    #         #             guidance_scale=self.config.guidance_scale,
    #         #             image_guidance_scale=self.config.image_guidance_scale,
    #         #             diffusion_steps=self.config.diffusion_steps,
    #         #             lower_bound=self.config.lower_bound,
    #         #             upper_bound=self.config.upper_bound,
    #         #         )
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image_{self.config_secret.image_guidance_scale_ip2p_ptd}.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.config_secret.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image_{self.config_secret.image_guidance_scale_ip2p_ptd}.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSquentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSquentialEdits = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    @profiler.time_function
    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        assert isinstance(
            self.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager),
        )
        num_eval = len(self.datamanager.fixed_indices_eval_dataloader)
        num_train = len(self.datamanager.train_dataset)  # type: ignore
        all_images = num_train + num_eval

        if not self.config.skip_point_metrics:
            pixels_per_frame = int(
                self.datamanager.train_dataset.cameras[0].width
                * self.datamanager.train_dataset.cameras[0].height
            )
            samples_per_frame = (self.config.num_pd_points + all_images) // (all_images)

        if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
            # init mushroom eval lists
            metrics_dict_with_list = []
            metrics_dict_within_list = []
            points_with = []
            points_within = []
            colors_with = []
            colors_within = []
        else:
            # eval lists for other dataparsers
            metrics_dict_list = []
            points_eval = []
            colors_eval = []
        points_train = []
        colors_train = []

        # # compute eval metrics and generate eval point clouds
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[green]Evaluating all eval images...", total=num_eval
            )

            cameras = self.datamanager.eval_dataset.cameras  # type: ignore
            for image_idx, batch in enumerate(
                self.datamanager.cached_eval  # Undistorted images
            ):  # type: ignore
                camera = cameras[image_idx : image_idx + 1].to("cpu")
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, _ = self.model.get_image_metrics_and_images(
                    outputs, batch
                )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (
                    num_rays / (time() - inner_start)
                ).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (
                    metrics_dict["num_rays_per_sec"] / (height * width)
                ).item()

                # get point cloud from each frame
                if "depth" in outputs and not self.config.skip_point_metrics:
                    depth = outputs["depth"]
                    rgb = outputs["rgb"]
                    indices = random.sample(range(pixels_per_frame), samples_per_frame)
                    c2w = torch.concatenate(
                        [
                            camera.camera_to_worlds,
                            torch.tensor([[[0, 0, 0, 1]]]).to(self.device),
                        ],
                        dim=1,
                    )
                    c2w = torch.matmul(
                        c2w,
                        torch.from_numpy(camera_utils.OPENGL_TO_OPENCV)
                        .float()
                        .to(depth.device),
                    )
                    fx, fy, cx, cy, img_size = (
                        camera.fx.item(),
                        camera.fy.item(),
                        camera.cx.item(),
                        camera.cy.item(),
                        (camera.width.item(), camera.height.item()),
                    )
                    if self._model.__class__.__name__ not in [
                        "DNSplatterModel",
                        "SplatfactoModel",
                    ]:
                        depth = depth / outputs["directions_norm"]

                    points, colors = camera_utils.get_colored_points_from_depth(
                        depths=depth,
                        rgbs=rgb,
                        c2w=c2w,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        img_size=img_size,
                        mask=indices,
                    )
                    points, colors = (
                        points.detach().cpu().numpy(),
                        colors.detach().cpu().numpy(),
                    )
                if (
                    self.datamanager.dataparser.__class__.__name__
                    == "MushroomDataParser"
                ):
                    seq_name = self.datamanager.eval_dataset.image_filenames[
                        batch["image_idx"]
                    ]
                    if "long_capture" in seq_name.parts[-3]:
                        metrics_dict_within_list.append(metrics_dict)
                        if not self.config.skip_point_metrics:
                            points_within.append(points)
                            colors_within.append(colors)
                    else:
                        metrics_dict_with_list.append(metrics_dict)
                        if not self.config.skip_point_metrics:
                            points_with.append(points)
                            colors_with.append(colors)
                else:
                    metrics_dict_list.append(metrics_dict)
                    if not self.config.skip_point_metrics:
                        points_eval.append(points)
                        colors_eval.append(colors)
                progress.advance(task)

        # save pointcloud from training images
        pd_metrics = {}
        if not self.config.skip_point_metrics:
            train_dataset = self.datamanager.train_dataset
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "[green]Extracting point cloud from train images...",
                    total=num_train,
                )
                for image_idx, _ in enumerate(train_dataset):
                    camera = train_dataset.cameras[image_idx : image_idx + 1].to(
                        self._model.device
                    )
                    outputs = self.model.get_outputs_for_camera(camera=camera)
                    rgb, depth = outputs["rgb"], outputs["depth"]
                    indices = random.sample(range(pixels_per_frame), samples_per_frame)
                    c2w = torch.concatenate(
                        [
                            camera.camera_to_worlds,
                            torch.tensor([[[0, 0, 0, 1]]]).to(self.device),
                        ],
                        dim=1,
                    )
                    c2w = torch.matmul(
                        c2w,
                        torch.from_numpy(camera_utils.OPENGL_TO_OPENCV)
                        .float()
                        .to(depth.device),
                    )
                    fx, fy, cx, cy, img_size = (
                        camera.fx.item(),
                        camera.fy.item(),
                        camera.cx.item(),
                        camera.cy.item(),
                        (camera.width.item(), camera.height.item()),
                    )
                    if self._model.__class__.__name__ not in [
                        "DNSplatterModel",
                        "SplatfactoModel",
                    ]:
                        depth = depth / outputs["directions_norm"]

                    points, colors = camera_utils.get_colored_points_from_depth(
                        depths=depth,
                        rgbs=rgb,
                        c2w=c2w,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        img_size=img_size,
                        mask=indices,
                    )
                    points, colors = (
                        points.detach().cpu().numpy(),
                        colors.detach().cpu().numpy(),
                    )
                    points_train.append(points)
                    colors_train.append(colors)
                    progress.advance(task)

            CONSOLE.print("[bold green]Computing point cloud metrics")
            pd_output_path = f"/{output_path}/final_renders"
            os.makedirs(os.getcwd() + f"{pd_output_path}", exist_ok=True)
            if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
                # load reference pcd for pointcloud comparison
                dataset_path = self.datamanager.dataparser_config.data
                ref_pcd_path = f"{dataset_path}/gt_pd.ply"
                if not os.path.exists(ref_pcd_path):
                    from dn_splatter.data.download_scripts.mushroom_download import (
                        download_mushroom,
                    )

                    download_mushroom(room_name=dataset_path.parts[-1], sequence="faro")
                ref_pcd = o3d.io.read_point_cloud(ref_pcd_path)
                transform_path = (
                    f"{dataset_path}/icp_{self.datamanager.dataparser_config.mode}.json"
                )
                initial_transformation = np.array(
                    json.load(open(transform_path))["gt_transformation"]
                ).reshape(4, 4)

                points_all = points_within + points_train
                colors_all = colors_within + colors_train
                points_all = np.vstack(points_all)
                colors_all = np.vstack(colors_all)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_all)
                pcd.colors = o3d.utility.Vector3dVector(colors_all)
                if self._model.__class__.__name__ not in [
                    "DNSplatterModel",
                    "SplatfactoModel",
                ]:
                    scale = self.datamanager.dataparser.scale_factor
                    transformation_matrix = self.datamanager.dataparser.transform_matrix
                    transformation_matrix = torch.cat(
                        [transformation_matrix, torch.tensor([0, 0, 0, 1]).unsqueeze(0)]
                    )
                    inverse_transformation = np.linalg.inv(transformation_matrix)

                    points = np.array(pcd.points) / scale
                    points = (
                        points @ inverse_transformation[:3, :3]
                        + inverse_transformation[:3, 3:4].T
                    )
                    pcd.points = o3d.utility.Vector3dVector(points)

                pcd = pcd.transform(initial_transformation)
                if output_path is not None:
                    o3d.io.write_point_cloud(
                        os.getcwd() + f"{pd_output_path}/pointcloud_within.ply", pcd
                    )

                acc, comp = self.pd_metrics(pcd, ref_pcd)
                pd_metrics.update(
                    {
                        "within_pd_acc": float(acc.item()),
                        "within_pd_comp": float(comp.item()),
                    }
                )

                points_all = points_with + points_train
                colors_all = colors_with + colors_train
                points_all = np.vstack(points_all)
                colors_all = np.vstack(colors_all)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_all)
                pcd.colors = o3d.utility.Vector3dVector(colors_all)
                if self._model.__class__.__name__ not in [
                    "DNSplatterModel",
                    "SplatfactoModel",
                ]:
                    scale = self.datamanager.dataparser.scale_factor
                    transformation_matrix = self.datamanager.dataparser.transform_matrix
                    transformation_matrix = torch.cat(
                        [transformation_matrix, torch.tensor([0, 0, 0, 1]).unsqueeze(0)]
                    )
                    inverse_transformation = np.linalg.inv(transformation_matrix)

                    points = np.array(pcd.points) / scale
                    points = (
                        points @ inverse_transformation[:3, :3]
                        + inverse_transformation[:3, 3:4].T
                    )
                    pcd.points = o3d.utility.Vector3dVector(points)

                pcd = pcd.transform(initial_transformation)
                if output_path is not None:
                    o3d.io.write_point_cloud(
                        os.getcwd() + f"{pd_output_path}/pointcloud_with.ply", pcd
                    )

                acc, comp = self.pd_metrics(pcd, ref_pcd)
                pd_metrics.update(
                    {
                        "with_pd_acc": float(acc.item()),
                        "with_pd_comp": float(comp.item()),
                    }
                )

            elif self.datamanager.dataparser.__class__.__name__ == "ReplicaDataparser":
                ref_pcd_path = self.config.datamanager.dataparser.data / (
                    self.config.datamanager.dataparser.sequence + "_mesh.ply"
                )  # load raplica mesh
                ref_mesh = trimesh.load_mesh(str(ref_pcd_path)).as_open3d
                ref_pcd = ref_mesh.sample_points_uniformly(
                    number_of_points=self.config.num_pd_points
                )
                points_all = points_eval + points_train
                colors_all = colors_eval + colors_train
                points_all = np.vstack(points_all)
                colors_all = np.vstack(colors_all)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_all)
                pcd.colors = o3d.utility.Vector3dVector(colors_all)

                if self._model.__class__.__name__ not in [
                    "DNSplatterModel",
                    "SplatfactoModel",
                ]:
                    scale = self.datamanager.dataparser.scale_factor
                    transformation_matrix = self.datamanager.dataparser.transform_matrix
                    transformation_matrix = torch.cat(
                        [transformation_matrix, torch.tensor([0, 0, 0, 1]).unsqueeze(0)]
                    )
                    inverse_transformation = np.linalg.inv(transformation_matrix)

                    points = np.array(pcd.points) / scale
                    points = (
                        points @ inverse_transformation[:3, :3]
                        + inverse_transformation[:3, 3:4].T
                    )
                    pcd.points = o3d.utility.Vector3dVector(points)

                if output_path is not None:
                    o3d.io.write_point_cloud(
                        os.getcwd() + f"{pd_output_path}/pointcloud.ply", pcd
                    )
                acc, comp = self.pd_metrics(pcd, ref_pcd)
                pd_metrics = {
                    "pd_acc": float(acc.item()),
                    "pd_comp": float(comp.item()),
                }
        # average the metrics list
        metrics_dict = {}

        if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
            for key in metrics_dict_within_list[0].keys():
                if get_std:
                    key_std, key_mean = torch.std_mean(
                        torch.tensor(
                            [
                                metrics_dict[key]
                                for metrics_dict in metrics_dict_within_list
                            ]
                        )
                    )
                    metrics_dict["within_" + key] = float(key_mean)
                    metrics_dict[f"within_{key}_std"] = float(key_std)
                else:
                    metrics_dict["within_" + key] = float(
                        torch.mean(
                            torch.tensor(
                                [
                                    metrics_dict["within_" + key]
                                    for metrics_dict in metrics_dict_within_list
                                ]
                            )
                        )
                    )
            for key in metrics_dict_with_list[0].keys():
                if get_std:
                    key_std, key_mean = torch.std_mean(
                        torch.tensor(
                            [
                                metrics_dict[key]
                                for metrics_dict in metrics_dict_with_list
                            ]
                        )
                    )
                    metrics_dict["with_" + key] = float(key_mean)
                    metrics_dict[f"with_{key}_std"] = float(key_std)
                else:
                    metrics_dict["with_" + key] = float(
                        torch.mean(
                            torch.tensor(
                                [
                                    metrics_dict[key]
                                    for metrics_dict in metrics_dict_with_list
                                ]
                            )
                        )
                    )
        else:
            for key in metrics_dict_list[0].keys():
                if get_std:
                    key_std, key_mean = torch.std_mean(
                        torch.tensor(
                            [metrics_dict[key] for metrics_dict in metrics_dict_list]
                        )
                    )
                    metrics_dict[key] = float(key_mean)
                    metrics_dict[f"{key}_std"] = float(key_std)
                else:
                    metrics_dict[key] = float(
                        torch.mean(
                            torch.tensor(
                                [
                                    metrics_dict[key]
                                    for metrics_dict in metrics_dict_list
                                ]
                            )
                        )
                    )
        metrics_dict.update(pd_metrics)
        self.train()

        # render images
        if output_path is not None:
            # render gs model images
            CONSOLE.print("[bold green]Rendering output images")
            if self._model.__class__.__name__ in ["DNSplatterModel", "SplatfactoModel"]:
                render_output_path = f"/{output_path}/final_renders"
                train_cache = self.datamanager.cached_train
                eval_cache = self.datamanager.cached_eval
                train_dataset = self.datamanager.train_dataset
                eval_dataset = self.datamanager.eval_dataset
                model = self._model
                gs_render_dataset_images(
                    train_cache=train_cache,
                    eval_cache=eval_cache,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    model=model,
                    render_output_path=render_output_path,
                    mushroom=(
                        True
                        if self.datamanager.dataparser.__class__.__name__
                        == "MushroomDataParser"
                        else False
                    ),
                    save_train_images=self.config.save_train_images,
                )
            else:
                # render other models
                print("Rendering for ", self._model.__class__.__name__)
                render_output_path = f"/{output_path}/final_renders"
                train_dataset = self.datamanager.train_dataset
                eval_dataset = self.datamanager.eval_dataset
                model = self._model
                train_dataloader = FixedIndicesEvalDataloader(
                    input_dataset=train_dataset,
                    device=self.datamanager.device,
                    num_workers=self.datamanager.world_size * 4,
                )
                eval_dataloader = FixedIndicesEvalDataloader(
                    input_dataset=eval_dataset,
                    device=self.datamanager.device,
                    num_workers=self.datamanager.world_size * 4,
                )
                ns_render_dataset_images(
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    model=model,
                    render_output_path=render_output_path,
                    mushroom=(
                        True
                        if self.datamanager.dataparser.__class__.__name__
                        == "MushroomDataParser"
                        else False
                    ),
                    save_train_images=self.config.save_train_images,
                )

        # compare rendered depth with faro depth for mushroom dataset
        if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
            if output_path is not None:
                faro_depth_path = (
                    self.datamanager.dataparser_config.data
                    / self.datamanager.dataparser_config.mode
                )
                faro_metrics = depth_eval_faro(output_path, faro_depth_path)
                metrics_dict.update(faro_metrics)

        return metrics_dict


    