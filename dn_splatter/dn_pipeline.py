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
from PIL import Image
import torchvision.transforms as transforms
import lpips
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt

from dn_splatter.ip2p_ptd import IP2P_PTD

from dn_splatter.ip2p_depth import InstructPix2Pix_depth

from nerfstudio.engine.callbacks import TrainingCallbackAttributes

from dn_splatter.utils.secret_utils import generate_ves_poses_opengl

from dn_splatter.utils.pie_utils import opencv_seamless_clone

from dn_splatter.utils.edge_loss_utils import SobelFilter, SobelEdgeLoss

import copy

# seva imports
from seva.data_io import get_parser
from seva.eval import (
    IS_TORCH_NIGHTLY,
    create_transforms_simple,
    infer_prior_stats,
    run_one_scene,
)
from seva.geometry import (
    get_default_intrinsics,
    get_preset_pose_fov,
)
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model

# lseg
from dn_splatter.utils.lseg_utils import lseg_module_init


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
        self.makeSequentialEdits = False
        
        # whether we are doing the first sequential edition
        self.first_SequentialEdit = True

        # whether we are at the first step
        self.first_step = True

        self.secret_loss_weight: float = 5.0

        config_path = 'config/config.yaml'
        with open(config_path, 'r') as file:
            cfg_dict = yaml.safe_load(file)
        self.config_secret = SimpleNamespace(**cfg_dict)

        self.config_secret.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        self.ip2p_ptd = IP2P_PTD(
            self.dtype, 
            self.config_secret.device, 
            conditioning_scale = self.config_secret.conditioning_scale,
            prompt=self.config_secret.prompt_2, 
            a_prompt=self.config_secret.a_prompt,
            n_prompt=self.config_secret.n_prompt,
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

        # refenece image for lpips computing
        self.ref_image = Image.open(self.ip2p_ptd.ref_img_path).convert('RGB').resize((self.config_secret.render_size, self.config_secret.render_size))
        # Convert reference image to tensor and process it the same way
        ref_image_tensor = torch.from_numpy(np.array(self.ref_image)).float() / 255.0  # Convert to [0, 1] range
        ref_image_tensor = ref_image_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
        self.ref_image_tensor = (ref_image_tensor * 2 - 1).clamp(-1, 1).to(self.config_secret.device)  # Convert to [-1, 1] range

        # 'vgg', 'alex', 'squeeze', lpips loss
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(self.config_secret.device)

        # edge loss 
        self.edge_loss_fn = SobelEdgeLoss(loss_type='l1', ksize=3, use_grayscale=self.config_secret.use_grayscale)

        # secret data preparation
        secret_view_idx = self.config_secret.secret_view_idx
        self.camera_secret, self.data_secret = self.datamanager.next_train_idx(secret_view_idx)
        self.original_image_secret = self.datamanager.original_cached_train[secret_view_idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        self.depth_image_secret = self.datamanager.original_cached_train[secret_view_idx]["depth"] # [bs, h, w]
        # original secret edges
        self.original_secret_edges = SobelFilter(ksize=3, use_grayscale=self.config_secret.use_grayscale)(self.original_image_secret)
        
        # lseg model
        self.lseg_model = lseg_module_init()

        # second secret view preparation
        secret_view_idx_2 = self.config_secret.secret_view_idx_2
        self.camera_secret_2, self.data_secret_2 = self.datamanager.next_train_idx(secret_view_idx_2)
        self.original_image_secret_2 = self.datamanager.original_cached_train[secret_view_idx_2]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        self.depth_image_secret_2 = self.datamanager.original_cached_train[secret_view_idx_2]["depth"] # [bs, h, w]

        self.first_iter = True

        c2w_secret = np.concatenate(
            [
                self.camera_secret.camera_to_worlds.cpu().numpy()[0],
                np.array([[0, 0, 0, 1]], dtype=np.float32)  # Add the last row for homogeneous coordinates
            ],
            0,
        )

        # generate ves cameras for validating the seva's ability
        self.ves_c2ws = generate_ves_poses_opengl(
            c2w_secret, # TODO fix the bug of 3 * 4 to 4 * 4!
            angle_limit_degrees=self.config_secret.angle_limits[0]
        )

        # Define VES view indices - these will be appended after existing training views
        self.num_ves_views = len(self.ves_c2ws)  # Should be 9 based on your code
        original_train_size = len(self.datamanager.cached_train)
        self.ves_view_indices = list(range(original_train_size, original_train_size + self.num_ves_views))
        
        # ############### Extend cached_train to accommodate VES views ###############
        # Create placeholder entries for VES views
        # for i in range(self.num_ves_views):
        #     # Create a placeholder data entry with the same structure as existing entries
        #     # You may need to adjust this based on your actual data structure
        #     placeholder_entry = {
        #         "image": torch.zeros_like(self.datamanager.cached_train[0]["image"]),  # Placeholder image
        #         "idx": original_train_size + i,
        #         "is_ves_view": True,  # Flag to identify VES views
        #     }
            
        #     # Add other required fields from your data structure but set depth/normal to None for VES views
        #     if "depth" in self.datamanager.cached_train[0]:
        #         placeholder_entry["depth"] = None  # Set to None instead of zeros
            
        #     if "sensor_depth" in self.datamanager.cached_train[0]:
        #         placeholder_entry["sensor_depth"] = None
                
        #     if "mono_depth" in self.datamanager.cached_train[0]:
        #         placeholder_entry["mono_depth"] = None
                
        #     if "normal" in self.datamanager.cached_train[0]:
        #         placeholder_entry["normal"] = None
                
        #     if "confidence" in self.datamanager.cached_train[0]:
        #         placeholder_entry["confidence"] = None
            
        #     # Add any other fields that exist in your cached_train entries
        #     for key in self.datamanager.cached_train[0].keys():
        #         if key not in placeholder_entry:
        #             if key in ["depth", "sensor_depth", "mono_depth", "normal", "confidence"]:
        #                 placeholder_entry[key] = None
        #             elif isinstance(self.datamanager.cached_train[0][key], torch.Tensor):
        #                 placeholder_entry[key] = torch.zeros_like(self.datamanager.cached_train[0][key])
        #             else:
        #                 placeholder_entry[key] = self.datamanager.cached_train[0][key]  # Copy non-tensor values
            
        #     self.datamanager.cached_train.append(placeholder_entry)
        
        # # Also extend original_cached_train if it exists
        # if hasattr(self.datamanager, 'original_cached_train'):
        #     for i in range(self.num_ves_views):
        #         placeholder_entry = {
        #             "image": torch.zeros_like(self.datamanager.original_cached_train[0]["image"]),
        #             "idx": original_train_size + i,
        #             "is_ves_view": True,
        #         }
                
        #         if "depth" in self.datamanager.original_cached_train[0]:
        #             placeholder_entry["depth"] = None
                
        #         for key in self.datamanager.original_cached_train[0].keys():
        #             if key not in placeholder_entry:
        #                 if key in ["depth", "sensor_depth", "mono_depth", "normal", "confidence"]:
        #                     placeholder_entry[key] = None
        #                 elif isinstance(self.datamanager.original_cached_train[0][key], torch.Tensor):
        #                     placeholder_entry[key] = torch.zeros_like(self.datamanager.original_cached_train[0][key])
        #                 else:
        #                     placeholder_entry[key] = self.datamanager.original_cached_train[0][key]
                
        #         self.datamanager.original_cached_train.append(placeholder_entry)

        # ############### generate ves cameras ###############
        # self.ves_cameras = []
        # for ves_c2w in self.ves_c2ws:
        #     # need to create a new copy for each camera, or all the cameras will refer to the same object
        #     camera_secret_copy = copy.deepcopy(self.camera_secret)
        #     ves_c2w_tensor = torch.tensor(ves_c2w, dtype=torch.float32, device=self.config_secret.device)
        #     camera_secret_copy.camera_to_worlds = ves_c2w_tensor[:3, :4].unsqueeze(0)
        #     self.ves_cameras.append(camera_secret_copy)

        # # seva c2ws input
        # self.task = "img2trajvid_s-prob"

        # # convert from OpenGL to OpenCV camera format
        # self.seva_c2ws = np.stack(self.ves_c2ws, axis=0) @ np.diag([1, -1, -1, 1])

        # DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees by default
        # self.num_frames = 9
        # fovs = np.full((self.num_frames,), DEFAULT_FOV_RAD)
        # aspect_ratio = 1.0
        # Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)  # unormalized
        # Ks[:, :2] *= (
        #     torch.tensor([self.config_secret.render_size, self.config_secret.render_size]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
        # )  # normalized
        # self.Ks = Ks.numpy()

        # # model loading
        # if IS_TORCH_NIGHTLY:
        #     COMPILE = True
        #     os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
        #     os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        # else:
        #     COMPILE = False
        # version=1.1
        # pretrained_model_name_or_path="stabilityai/stable-virtual-camera"
        # weight_name="model.safetensors"
        # self.MODEL = SGMWrapper(
        #     load_model(
        #         model_version=version,
        #         pretrained_model_name_or_path=pretrained_model_name_or_path,
        #         weight_name=weight_name,
        #         device="cpu",
        #         verbose=True,
        #     ).eval()
        # ).to(self.config_secret.device)

        # if COMPILE:
        #     MODEL = torch.compile(MODEL, dynamic=False)

        # self.AE = AutoEncoder(chunk_size=1).to(self.config_secret.device)
        # self.CONDITIONER = CLIPConditioner().to(self.config_secret.device)
        # self.DENOISER = DiscreteDenoiser(num_idx=1000, device=self.config_secret.device)

        # if COMPILE:
        #     self.CONDITIONER = torch.compile(self.CONDITIONER, dynamic=False)
        #     self.AE = torch.compile(self.AE, dynamic=False)

        # self.seed = 23

        # options = {
        #     'chunk_strategy': 'interp', 
        #     'video_save_fps': 30.0, 
        #     'beta_linear_start': 5e-06, 
        #     'log_snr_shift': 2.4, 
        #     'guider_types': 1, 
        #     'cfg': (4.0, 2.0), 
        #     'camera_scale': 0.1, 
        #     'num_steps': 20, 
        #     'cfg_min': 1.2, 
        #     'encoding_t': 1, 
        #     'decoding_t': 1, 
        #     'replace_or_include_input': True, 
        #     'traj_prior': 'stabilization', 
        #     'guider': (1, 2), 
        #     'num_targets': 8
        # }

        # self.VERSION_DICT = {
        #     'H': 512, 
        #     'W': 512, 
        #     'T': 21, 
        #     'C': 4, 
        #     'f': 8,
        #     "options": options,
        # }

        # self.num_inputs = 1
        # self.num_targets = self.num_frames - 1
        # self.input_indices = [0]
        # num_anchors = infer_prior_stats(
        #     self.VERSION_DICT["T"],
        #     self.num_inputs,
        #     num_total_frames=self.num_targets,
        #     version_dict=self.VERSION_DICT,
        # )
        # self.anchor_indices = np.linspace(1, self.num_targets, num_anchors).tolist()

        # self.anchor_c2ws = self.seva_c2ws[[round(ind) for ind in self.anchor_indices]]
        # self.anchor_Ks = self.Ks[[round(ind) for ind in self.anchor_indices]]

        # self.anchor_c2ws = torch.tensor(self.anchor_c2ws[:, :3]).float()
        # self.anchor_Ks = torch.tensor(self.anchor_Ks).float()

        # self.seva_c2ws = torch.tensor(self.seva_c2ws[:, :3]).float()
        # self.Ks = torch.tensor(self.Ks).float()

        
        ######################################################
    
    # add callback function to fetch the components from other parts of the training process.
    def get_training_callbacks(self, attrs: TrainingCallbackAttributes):
        # stash a reference to the Trainer
        self.trainer = attrs.trainer
        # now return whatever callbacks the base class wants
        return super().get_training_callbacks(attrs)

    def _axis_angle_to_rotation_matrix(self, axis_angle):
        """Convert axis-angle to rotation matrix using matrix exponential."""
        # Create skew-symmetric matrix
        K = torch.zeros(3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
        K[0, 1] = -axis_angle[2]
        K[0, 2] = axis_angle[1]
        K[1, 0] = axis_angle[2]
        K[1, 2] = -axis_angle[0]
        K[2, 0] = -axis_angle[1]
        K[2, 1] = axis_angle[0]
        
        # Use matrix exponential (more stable for gradients)
        R = torch.matrix_exp(K)
        return R

    # 2nd stage: secret + non-secret loss + masked fighting (ref, original) loss
    # uncomment the main_loss = 0.0 on line 659 in dn_model.py before training
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_loss_fighting_ref_original_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view every secret_edit_rate steps
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         model_outputs_secret = self.model(self.camera_secret)
    #         metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #         loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1, lpips, regularization

    #         # compute masked lpips value
    #         mask_np = self.ip2p_ptd.mask
    #         # Convert mask to tensor and ensure it's the right shape/device
    #         mask_tensor = torch.from_numpy(mask_np).float()
    #         if len(mask_tensor.shape) == 2:
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #         if mask_tensor.shape[0] == 1:
    #             mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #         mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #         mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #         # Prepare model output
    #         model_rgb_secret = (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #         # Apply mask to both images
    #         masked_model_rgb = model_rgb_secret * mask_tensor
    #         masked_ref_image = self.ref_image_tensor * mask_tensor

    #         # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #         ref_loss = self.lpips_loss_fn(
    #             masked_model_rgb,
    #             masked_ref_image
    #         ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #         ref_l1_loss = torch.nn.functional.l1_loss(
    #             masked_model_rgb,
    #             masked_ref_image
    #         )
    #         loss_dict_secret["main_loss"] += self.config_secret.ref_loss_weight * ref_loss + ref_l1_loss

    #         # edge loss
    #         # rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #         # edge_loss = self.edge_loss_fn(
    #         #     rendered_image_secret.to(self.config_secret.device), 
    #         #     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #         #     self.original_secret_edges.to(self.config_secret.device),
    #         #     image_dir,
    #         #     step
    #         # )
    #         # loss_dict_secret["main_loss"] += edge_loss

    #         if step % 100 == 0:
    #             image_save_secret = torch.cat([model_outputs_secret["rgb"].detach().permute(2, 0, 1).unsqueeze(0), ((masked_ref_image + 1) / 2).to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         # put the secret metrics and loss into the main dict
    #         for k, v in metrics_dict_secret.items():
    #             metrics_dict[f"secret_{k}"] = v
    #         for k, v in loss_dict_secret.items():
    #             loss_dict[f"secret_{k}"] = v

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    ############################################################################################################
    # start: 2nd stage: secret + non-secret loss + masked fighting (ref, original) loss + lseg loss
    ############################################################################################################
    def get_train_loss_dict(self, step: int):
        base_dir = self.trainer.base_dir
        image_dir = base_dir / f"images_only_secret_loss_fighting_ref_original_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
        if not image_dir.exists():
            image_dir.mkdir(parents=True, exist_ok=True)

        # non-editing steps loss computing
        camera, data = self.datamanager.next_train(step)
        model_outputs = self.model(camera)
        metrics_dict = self.model.get_metrics_dict(model_outputs, data)
        loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

        # update the secret view every secret_edit_rate steps
        if step % self.config_secret.secret_edit_rate == 0:
            model_outputs_secret = self.model(self.camera_secret)
            metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
            loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1, lpips, regularization

            # compute masked lpips value
            mask_np = self.ip2p_ptd.mask
            # Convert mask to tensor and ensure it's the right shape/device
            mask_tensor = torch.from_numpy(mask_np).float()
            if len(mask_tensor.shape) == 2:
                mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
            if mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
            mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
            mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

            # Prepare model output
            model_rgb_secret = (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

            # Apply mask to both images
            masked_model_rgb = model_rgb_secret * mask_tensor
            masked_ref_image = self.ref_image_tensor * mask_tensor

            # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
            ref_loss = self.lpips_loss_fn(
                masked_model_rgb,
                masked_ref_image
            ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
            ref_l1_loss = torch.nn.functional.l1_loss(
                masked_model_rgb,
                masked_ref_image
            )
            # loss_dict_secret["main_loss"] += self.config_secret.ref_loss_weight * ref_loss + ref_l1_loss
            metrics_dict["ref_loss"] = self.config_secret.ref_loss_weight * ref_loss + ref_l1_loss
            loss_dict["ref_loss"] = self.config_secret.ref_loss_weight * ref_loss + ref_l1_loss

            # edge loss
            # rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
            # edge_loss = self.edge_loss_fn(
            #     rendered_image_secret.to(self.config_secret.device), 
            #     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
            #     self.original_secret_edges.to(self.config_secret.device),
            #     image_dir,
            #     step
            # )
            # loss_dict_secret["main_loss"] += edge_loss

            # lseg loss
            rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
            # [1, 512, 256, 256]
            rendered_image_sem_feature = self.lseg_model.get_image_features(rendered_image_secret.to(self.config_secret.device))
            
            with torch.no_grad():
                original_image_sem_feature = self.lseg_model.get_image_features(self.original_image_secret.to(self.config_secret.device))

            lseg_loss = torch.nn.functional.l1_loss(rendered_image_sem_feature, original_image_sem_feature)

            # loss_dict_secret["main_loss"] += lseg_loss
            metrics_dict["lseg_loss"] = lseg_loss * 10.0
            loss_dict["lseg_loss"] = lseg_loss * 10.0

            if step % 100 == 0:
                image_save_secret = torch.cat([model_outputs_secret["rgb"].detach().permute(2, 0, 1).unsqueeze(0), ((masked_ref_image + 1) / 2).to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
                save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

            # put the secret metrics and loss into the main dict
            for k, v in metrics_dict_secret.items():
                metrics_dict[f"secret_{k}"] = v
            for k, v in loss_dict_secret.items():
                loss_dict[f"secret_{k}"] = v

        return model_outputs, loss_dict, metrics_dict
    ############################################################################################################
    # end: 2nd stage: secret + non-secret loss + masked fighting (ref, original) loss + lseg loss
    ############################################################################################################


    # 2nd stage: only masked fighting (ref, original) loss
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_loss_fighting_ref_original_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1 + lpips
    #     loss_dict_secret["main_loss"] = 0.0

    #     # compute masked lpips value
    #     mask_np = self.ip2p_ptd.mask
    #     # Convert mask to tensor and ensure it's the right shape/device
    #     mask_tensor = torch.from_numpy(mask_np).float()
    #     if len(mask_tensor.shape) == 2:
    #         mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #     if mask_tensor.shape[0] == 1:
    #         mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #     mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #     mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #     # Prepare model output
    #     model_rgb_secret = (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #     original_image_secret = (self.datamanager.original_cached_train[self.config_secret.secret_view_idx]["image"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1).to(self.ref_image_tensor.device)

    #     # Apply mask to both images
    #     masked_model_rgb = model_rgb_secret * mask_tensor
    #     masked_ref_image = self.ref_image_tensor * mask_tensor
    #     masked_original_image = original_image_secret * mask_tensor

    #     # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #     ref_loss = self.lpips_loss_fn(
    #         masked_model_rgb,
    #         masked_ref_image
    #     ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #     original_loss = self.lpips_loss_fn(
    #         masked_model_rgb,
    #         masked_original_image
    #     ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
        
    #     loss_dict_secret["main_loss"] += self.config_secret.ref_loss_weight * ref_loss + original_loss

    #     if step % 100 == 0:
    #         image_save_secret = torch.cat([model_outputs_secret["rgb"].detach().permute(2, 0, 1).unsqueeze(0), ((masked_ref_image + 1) / 2).to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #     return model_outputs_secret, loss_dict_secret, metrics_dict_secret

    ######################################################
    # start: 3rd stage: only editing with ip2p
    ######################################################
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_3rd_stage_only_editing_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # replace the original dataset with current rendering
    #     if self.first_step:
    #         self.first_step = False
        
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             rendered_image = model_outputs["rgb"].detach()

    #             self.datamanager.original_cached_train[idx]["image"] = rendered_image
    #             self.datamanager.cached_train[idx]["image"] = rendered_image
    #             data["image"] = rendered_image

    #         print("dataset replacement complete!")

    #     # start editing
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         all_indices = np.arange(len(self.datamanager.cached_train))

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(all_indices)
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             if idx == self.config_secret.secret_view_idx:
    #                 image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)
                
    #             ############################ for edge loss ########################
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1 + lpips

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step + 1
    #             )
    #             loss_dict_secret["main_loss"] += edge_loss * 0.1

    #             # put the secret metrics and loss into the main dict
    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
    #             ############################ for edge loss ########################

    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #             self.text_embeddings_ip2p.to(self.config_secret.device),
    #             rendered_image.to(self.dtype),
    #             original_image.to(self.config_secret.device).to(self.dtype),
    #             False, # is depth tensor
    #             depth_image,
    #             guidance_scale=self.config_secret.guidance_scale,
    #             image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #             diffusion_steps=self.config_secret.t_dec,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound,
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image.size() != rendered_image.size()):
    #             edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image = edited_image.to(original_image.dtype)
    #         self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #         data["image"] = edited_image.squeeze().permute(1,2,0)

    #         # save edited non-secret image
    #         if step % 25 == 0:
    #             image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')
            
    #         if idx == self.config_secret.secret_view_idx:
    #             image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
    ######################################################
    # end: 3rd stage: only editing with ip2p
    ######################################################

    ######################################################
    # start: 3rd stage: original IGS2GS
    ######################################################
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_3rd_stage_only_editing_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # replace the original dataset with current rendering
    #     if self.first_step:
    #         self.first_step = False
        
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             rendered_image = model_outputs["rgb"].detach()

    #             self.datamanager.original_cached_train[idx]["image"] = rendered_image
    #             self.datamanager.cached_train[idx]["image"] = rendered_image
    #             data["image"] = rendered_image

    #         print("dataset replacement complete!")

    #     # start editing
    #     if (step % self.config.gs_steps) == 0:
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         camera, data = self.datamanager.next_train(step)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #             self.text_embeddings_ip2p.to(self.config_secret.device),
    #             rendered_image.to(self.dtype),
    #             original_image.to(self.config_secret.device).to(self.dtype),
    #             False, # is depth tensor
    #             depth_image,
    #             guidance_scale=self.config_secret.guidance_scale,
    #             image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #             diffusion_steps=self.config_secret.t_dec,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound,
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image.size() != rendered_image.size()):
    #             edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image = edited_image.to(original_image.dtype)
    #         self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #         data["image"] = edited_image.squeeze().permute(1,2,0)

    #         # save edited non-secret image
    #         if step % 25 == 0:
    #             image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')
            
    #         if idx == self.config_secret.secret_view_idx:
    #             image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False

    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     return model_outputs, loss_dict, metrics_dict
    ######################################################
    # end: 3rd stage: only editing with ip2p
    ######################################################
    
    # __camera_pose_offset_updating__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_camera_pose_offset_updating"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     if self.first_step:
    #         self.first_step = False
    #         # find the best secret view that align with the reference image
    #         current_secret_idx = 0
    #         current_score = float("inf")
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             # compute masked lpips value
    #             mask_np = self.ip2p_ptd.mask
    #             # Convert mask to tensor and ensure it's the right shape/device
    #             mask_tensor = torch.from_numpy(mask_np).float()
    #             if len(mask_tensor.shape) == 2:
    #                 mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #             if mask_tensor.shape[0] == 1:
    #                 mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #             mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #             # Prepare model output
    #             model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #             # Apply mask to both images
    #             masked_model_rgb = model_rgb * mask_tensor
    #             masked_ref_image = self.ref_image_tensor * mask_tensor

    #             # Compute masked LPIPS score
    #             lpips_score = self.lpips_loss_fn(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             l1_score = torch.nn.functional.l1_loss(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             # score = l1_score
    #             score = lpips_score

    #             # # unmasked lpips score
    #             # lpips_score = self.lpips_loss_fn(
    #             #     (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #             #     self.ref_image_tensor
    #             # ).item()

    #             if score < current_score:
    #                 current_score = score
    #                 current_secret_idx = idx

    #         self.secret_view_idx = current_secret_idx
    #         camera, data = self.datamanager.next_train_idx(self.secret_view_idx)
    #         model_outputs = self.model(camera)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         save_image(rendered_image, image_dir / f"best_secret_view_{self.secret_view_idx}_score_{current_score}.png")

    #         CONSOLE.print(f"Best secret view index: {self.secret_view_idx} with score: {current_score}")

    #         # start pose offset updating
    #         # Get the secret view camera and initialize pose offset parameters
    #         camera_secret, data_secret = self.datamanager.next_train_idx(self.secret_view_idx)

    #         original_pose_backup = camera_secret.camera_to_worlds.clone()
            
    #         # Initialize camera pose offset parameters (6DOF: translation + rotation)
    #         if not hasattr(self, 'camera_pose_offset'):
    #             # Translation offset (x, y, z)
    #             self.translation_offset = torch.zeros(3, device=camera_secret.camera_to_worlds.device, requires_grad=True)
    #             # Rotation offset (axis-angle representation)
    #             self.rotation_offset = torch.zeros(3, device=camera_secret.camera_to_worlds.device, requires_grad=True)
                
    #             # Optimizer for camera pose offset
    #             self.pose_optimizer = torch.optim.Adam([self.translation_offset, self.rotation_offset], lr=float(self.config_secret.pose_learning_rate))

    #         # Before the pose optimization loop, store the gradient state and disable model gradients
    #         model_param_grad_states = {}
    #         for name, param in self.model.named_parameters():
    #             model_param_grad_states[name] = param.requires_grad
    #             param.requires_grad = False

    #         # Camera pose optimization loop
    #         num_pose_iterations = self.config_secret.num_pose_iterations
            
    #         for pose_iter in range(num_pose_iterations):
    #             self.pose_optimizer.zero_grad()
                
    #             # Create rotation matrix from axis-angle representation
    #             rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                
    #             # Apply rotation offset: R_new = R_offset @ R_original
    #             original_rotation = original_pose_backup[0, :3, :3]
    #             new_rotation = rotation_matrix @ original_rotation
                
    #             # Apply translation offset: t_new = t_original + t_offset
    #             original_translation = original_pose_backup[0, :3, 3]
    #             new_translation = original_translation + self.translation_offset
                
    #             # Construct new camera-to-world matrix
    #             new_c2w = original_pose_backup.clone()
    #             new_c2w[0, :3, :3] = new_rotation
    #             new_c2w[0, :3, 3] = new_translation
                
    #             camera_secret.camera_to_worlds = new_c2w
                
    #             # Render with updated camera pose
    #             with torch.enable_grad():
    #                 model_outputs = self.model(camera_secret)
                    
    #                 # Compute LPIPS loss with mask
    #                 mask_np = self.ip2p_ptd.mask
    #                 mask_tensor = torch.from_numpy(mask_np).float()
    #                 if len(mask_tensor.shape) == 2:
    #                     mask_tensor = mask_tensor.unsqueeze(0)
    #                 if mask_tensor.shape[0] == 1:
    #                     mask_tensor = mask_tensor.repeat(3, 1, 1)
    #                 mask_tensor = mask_tensor.unsqueeze(0)
    #                 mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #                 model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1) # don't detach here, or the gradients won't exist
    #                 masked_model_rgb = model_rgb * mask_tensor
    #                 masked_ref_image = self.ref_image_tensor * mask_tensor
                    
    #                 lpips_loss = self.lpips_loss_fn(masked_model_rgb, masked_ref_image)
    #                 l1_loss = torch.nn.functional.l1_loss(masked_model_rgb, masked_ref_image)
                    
    #                 # Add regularization to prevent large offsets
    #                 translation_reg = torch.norm(self.translation_offset) * float(self.config_secret.translation_reg_weight)
    #                 rotation_reg = torch.norm(self.rotation_offset) * float(self.config_secret.rotation_reg_weight)
    #                 # total_loss = lpips_loss + translation_reg + rotation_reg
    #                 total_loss = lpips_loss
    #                 # total_loss = l1_loss
                    
    #                 # Backward pass and optimization step
    #                 total_loss.backward()
    #                 self.pose_optimizer.step()
                
    #             # Optional: clamp offsets to reasonable ranges
    #             with torch.no_grad():
    #                 self.translation_offset.clamp_(-self.config_secret.max_translation_offset, 
    #                                             self.config_secret.max_translation_offset)
    #                 self.rotation_offset.clamp_(-self.config_secret.max_rotation_offset, 
    #                                         self.config_secret.max_rotation_offset)
                
    #             if pose_iter % 50 == 0:
    #                 with torch.no_grad():
    #                     CONSOLE.print(
    #                         # f"Translation gradient norm: {self.translation_offset.grad.norm().item()}",
    #                         # f"Rotation gradient norm: {self.rotation_offset.grad.norm().item()}",
    #                         f"Pose iter {pose_iter}: total loss = {total_loss.item():.6f}, "
    #                         f"Trans offset norm = {torch.norm(self.translation_offset).item():.6f}, "
    #                         f"Rot offset norm = {torch.norm(self.rotation_offset).item():.6f}"
    #                     )

    #                     rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                    
    #                     new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #                     new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                        
    #                     new_c2w = original_pose_backup.clone()
    #                     new_c2w[0, :3, :3] = new_rotation
    #                     new_c2w[0, :3, 3] = new_translation
                        
    #                     camera_secret.camera_to_worlds = new_c2w
    #                     optimized_camera = camera_secret
                        
    #                     final_outputs = self.model(optimized_camera)
    #                     rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                        
    #                     # Compute final LPIPS score
    #                     model_rgb = (final_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)
    #                     masked_model_rgb = model_rgb * mask_tensor
                        
    #                     save_image(rendered_image, 
    #                             image_dir / f"optimized_secret_view_{self.secret_view_idx}_step_{pose_iter}_loss_{total_loss.item():.6f}.png")
            
    #         # After optimization, restore model parameter gradient states
    #         for name, param in self.model.named_parameters():
    #             param.requires_grad = model_param_grad_states[name]

    #         # # Final rendering with optimized pose
    #         # if step % 10 == 0:
    #         #     with torch.no_grad():
    #         #         rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                    
    #         #         new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #         #         new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                    
    #         #         new_c2w = original_pose_backup.clone()
    #         #         new_c2w[0, :3, :3] = new_rotation
    #         #         new_c2w[0, :3, 3] = new_translation
                    
    #         #         camera_secret.camera_to_worlds = new_c2w
    #         #         optimized_camera = camera_secret
                    
    #         #         final_outputs = self.model(optimized_camera)
    #         #         rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                    
    #         #         # Compute final LPIPS score
    #         #         model_rgb = (final_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)
    #         #         masked_model_rgb = model_rgb * mask_tensor
    #         #         final_lpips = self.lpips_loss_fn(masked_model_rgb, masked_ref_image).item()
                    
    #         #         save_image(rendered_image, 
    #         #                 image_dir / f"optimized_secret_view_{self.secret_view_idx}_step_{step}_lpips_{final_lpips:.6f}.png")
                    
    #         #         CONSOLE.print(f"Final optimized LPIPS score: {final_lpips:.6f}")
                    
    #         #         # Store optimized camera for use in training
    #         #         self.camera_secret = optimized_camera
    #         #         self.data_secret = data_secret       

    #         # secret data preparation
    #         with torch.no_grad():
    #             rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                
    #             new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #             new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                
    #             new_c2w = original_pose_backup.clone()
    #             new_c2w[0, :3, :3] = new_rotation
    #             new_c2w[0, :3, 3] = new_translation
                
    #             camera_secret.camera_to_worlds = new_c2w
    #             optimized_camera = camera_secret
                
    #             final_outputs = self.model(optimized_camera)
    #             rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)  
                
    #             # secret data preparation
    #             self.camera_secret, self.data_secret = optimized_camera, data_secret
    #             self.original_image_secret = rendered_image
    #             self.depth_image_secret = self.datamanager.original_cached_train[self.secret_view_idx]["depth"] # [bs, h, w]
    #             # original secret edges
    #             self.original_secret_edges = SobelFilter(ksize=3, use_grayscale=self.config_secret.use_grayscale)(self.original_image_secret)
            
    #     return model_outputs, loss_dict, metrics_dict
    ######################################################

    # only secret
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # # only secret loss
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_loss_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     torch.cuda.empty_cache()

    #     return model_outputs_secret, loss_dict_secret, metrics_dict_secret
        ######################################################


    # secret + non-secret loss + masked fighting (ref, original) loss
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_loss_fighting_ref_original_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view every secret_edit_rate steps
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         model_outputs_secret = self.model(self.camera_secret)
    #         metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #         loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #         # compute masked lpips value
    #         mask_np = self.ip2p_ptd.mask
    #         # Convert mask to tensor and ensure it's the right shape/device
    #         mask_tensor = torch.from_numpy(mask_np).float()
    #         if len(mask_tensor.shape) == 2:
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #         if mask_tensor.shape[0] == 1:
    #             mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #         mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #         mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #         # Prepare model output
    #         model_rgb_secret = (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #         # Apply mask to both images
    #         masked_model_rgb = model_rgb_secret * mask_tensor
    #         masked_ref_image = self.ref_image_tensor * mask_tensor

    #         # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #         ref_loss = self.lpips_loss_fn(
    #             masked_model_rgb,
    #             masked_ref_image
    #         ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #         loss_dict_secret["main_loss"] += self.config_secret.ref_loss_weight * ref_loss

    #         if step % 100 == 0:
    #             image_save_secret = torch.cat([model_outputs_secret["rgb"].detach().permute(2, 0, 1).unsqueeze(0), ((masked_ref_image + 1) / 2).to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         # put the secret metrics and loss into the main dict
    #         for k, v in metrics_dict_secret.items():
    #             metrics_dict[f"secret_{k}"] = v
    #         for k, v in loss_dict_secret.items():
    #             loss_dict[f"secret_{k}"] = v

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    # only secret + fighting loss (ref， PTD)
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_fighting_loss"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     # a content loss of ref image added to the original rgb (L1 + lpips loss)
    #     ref_loss_weight = 0.2
    #     ref_loss = self.lpips_loss_fn(
    #         (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #         self.ref_image_tensor
    #     ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #     loss_dict["main_loss"] += ref_loss_weight * ref_loss

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # pie + only secret
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_pie_only_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         # ###########################################
    #         # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #         # edited_image_secret is B, C, H, W in [0, 1]
    #         edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #         edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #         # Convert image_original to numpy
    #         if hasattr(self.ip2p_ptd.image_original, 'cpu'):
    #             # It's a PyTorch tensor
    #             image_original_np = self.ip2p_ptd.image_original.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             image_original_np = (image_original_np * 255).astype(np.uint8)
    #         elif hasattr(self.ip2p_ptd.image_original, 'save'):
    #             # It's a PIL Image
    #             image_original_np = np.array(self.ip2p_ptd.image_original)
    #             # PIL images are usually already in [0, 255] uint8 format
    #             if image_original_np.dtype != np.uint8:
    #                 image_original_np = image_original_np.astype(np.uint8)
    #         else:
    #             # If it's already numpy, just use it
    #             image_original_np = self.ip2p_ptd.image_original

    #         # Convert mask to numpy
    #         if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #             # It's a PyTorch tensor
    #             mask_tensor = self.ip2p_ptd.mask
    #             if mask_tensor.dim() == 4:
    #                 mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #             elif mask_tensor.dim() == 3:
    #                 mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #             else:
    #                 mask_np = mask_tensor.cpu().numpy()
    #             mask_np = (mask_np * 255).astype(np.uint8)
    #         elif hasattr(self.ip2p_ptd.mask, 'save'):
    #             # It's a PIL Image
    #             mask_np = np.array(self.ip2p_ptd.mask)
    #             # Convert to grayscale if needed
    #             if mask_np.ndim == 3:
    #                 mask_np = mask_np[:, :, 0]  # Take first channel
    #             # Ensure it's uint8
    #             if mask_np.dtype != np.uint8:
    #                 if mask_np.max() <= 1.0:
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 else:
    #                     mask_np = mask_np.astype(np.uint8)
    #         else:
    #             # If it's already numpy, just use it
    #             mask_np = self.ip2p_ptd.mask

    #         # Call the original opencv_seamless_clone function with numpy arrays
    #         result_np = opencv_seamless_clone(edited_image_np, image_original_np, mask_np)

    #         # Convert the result back to PyTorch tensor format
    #         # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #         edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #         edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #         edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #         # ###########################################

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # pie + only_secret + edge loss
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_pie_only_secret_edge_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     # compute edge loss
    #     # rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #     rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #     edge_loss = self.edge_loss_fn(
    #         rendered_image_secret.to(self.config_secret.device), 
    #         self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #         self.original_secret_edges.to(self.config_secret.device),
    #         # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #         image_dir,
    #         step
    #     )
    #     loss_dict["main_loss"] += edge_loss

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')
            
    #         if step % 200 == 0:
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         # ###########################################
    #         # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #         # edited_image_secret is B, C, H, W in [0, 1]
    #         edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #         edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #         # Convert image_original to numpy
    #         if hasattr(self.original_image_secret, 'cpu'):
    #             # It's a PyTorch tensor
    #             image_original_np = self.original_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             image_original_np = (image_original_np * 255).astype(np.uint8)
    #         elif hasattr(self.original_image_secret, 'save'):
    #             # It's a PIL Image
    #             image_original_np = np.array(self.original_image_secret)
    #             # PIL images are usually already in [0, 255] uint8 format
    #             if image_original_np.dtype != np.uint8:
    #                 image_original_np = image_original_np.astype(np.uint8)
    #         else:
    #             # If it's already numpy, just use it
    #             image_original_np = self.original_image_secret

    #         # Convert mask to numpy
    #         if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #             # It's a PyTorch tensor
    #             mask_tensor = self.ip2p_ptd.mask
    #             if mask_tensor.dim() == 4:
    #                 mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #             elif mask_tensor.dim() == 3:
    #                 mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #             else:
    #                 mask_np = mask_tensor.cpu().numpy()
    #             mask_np = (mask_np * 255).astype(np.uint8)
    #         elif hasattr(self.ip2p_ptd.mask, 'save'):
    #             # It's a PIL Image
    #             mask_np = np.array(self.ip2p_ptd.mask)
    #             # Convert to grayscale if needed
    #             if mask_np.ndim == 3:
    #                 mask_np = mask_np[:, :, 0]  # Take first channel
    #             # Ensure it's uint8
    #             if mask_np.dtype != np.uint8:
    #                 if mask_np.max() <= 1.0:
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 else:
    #                     mask_np = mask_np.astype(np.uint8)
    #         else:
    #             # If it's already numpy, just use it
    #             mask_np = self.ip2p_ptd.mask

    #         # Call the original opencv_seamless_clone function with numpy arrays
    #         result_np = opencv_seamless_clone(edited_image_np, image_original_np, mask_np)

    #         # Convert the result back to PyTorch tensor format
    #         # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #         edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #         edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #         edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #         # ###########################################

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         if step % 200 == 0:
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # remove close views + only secret
    # def _is_view_close_to_secret(self, camera):
    #     """
    #     Check if the current camera view is close to the secret view.
    #     You can customize this logic based on your specific criteria.
    #     """
    #     # Get camera poses
    #     current_pose = camera.camera_to_worlds[0]  # Assuming batch size 1
    #     secret_pose = self.camera_secret.camera_to_worlds[0]
        
    #     # Calculate distance between camera positions
    #     position_distance = torch.norm(current_pose[:3, 3] - secret_pose[:3, 3])
        
    #     # Calculate angular difference between camera orientations
    #     # Using the rotation matrices (first 3x3 of the poses)
    #     current_rotation = current_pose[:3, :3]
    #     secret_rotation = secret_pose[:3, :3]
        
    #     # Calculate rotation difference using trace of R1^T * R2
    #     rotation_diff = torch.trace(torch.matmul(current_rotation.T, secret_rotation))
    #     # Convert to angle: cos(angle) = (trace(R) - 1) / 2
    #     angle_diff = torch.acos(torch.clamp((rotation_diff - 1) / 2, -1, 1))

    #     # print("position_distance: ", position_distance, "angle_diff: ", angle_diff)
        
    #     # Define thresholds (you may need to adjust these based on your scene scale)
    #     position_threshold = 1.0  # Adjust based on your scene scale
    #     angle_threshold = 0.5  # Radians (0.2 about 11.5 degrees, 0.5, 60 degrees)
        
    #     # Check if view is close based on both position and orientation
    #     is_close = (position_distance < position_threshold) and (angle_diff < angle_threshold)
        
    #     return is_close

    # def _is_rgb_loss(self, loss_key):
    #     """
    #     Determine if a loss key corresponds to an RGB-related loss.
    #     Based on your loss_dict structure: {'main_loss': ..., 'scale_reg': ...}
        
    #     Simple approach: Only filter main_loss for close views
    #     """
    #     # Only filter out main_loss for close views, keep everything else
    #     return loss_key == 'main_loss'
    
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
        
    #     # Check if current view is close to secret view
    #     is_close_to_secret = self._is_view_close_to_secret(camera)
        
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)
        
    #     # Filter out RGB loss for views close to secret view
    #     if is_close_to_secret:
    #         # Remove RGB-related losses while keeping depth and normal losses
    #         filtered_loss_dict = {}
    #         for key, value in loss_dict.items():
    #             # Keep all losses except RGB-related ones
    #             if not self._is_rgb_loss(key):
    #                 filtered_loss_dict[key] = value
    #             else:
    #                 # Optionally log that we're skipping this loss
    #                 print(f"Skipping RGB loss '{key}' for view close to secret view")
    #         loss_dict = filtered_loss_dict

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # seva + only secret
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_seva_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0 or self.first_iter:
    #         self.first_iter = False

    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         # generate ves views
    #         rgb_list = []
    #         for ves_camera in self.ves_cameras:
    #             model_outputs_ves = self.model(ves_camera)
    #             rendered_image_ves = model_outputs_ves["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2) # [1, 3, H, W]
    #             rgb_list.append(rendered_image_ves.cpu().squeeze())

    #         row1 = torch.cat([rgb_list[8], rgb_list[7], rgb_list[6]], dim=2)
    #         row2 = torch.cat([rgb_list[5], rgb_list[0], rgb_list[4]], dim=2)
    #         row3 = torch.cat([rgb_list[3], rgb_list[2], rgb_list[1]], dim=2)  # concat along W

    #         # Now stack the three rows along H to get a single [3, 3H, 3W] image
    #         img = torch.cat([row1, row2, row3], dim=1)  # concat along H

    #         save_image(img.clamp(0, 1), image_dir / f'{step}_ves_image.png')

    #         # save secret images
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         # seva results
    #         all_imgs_path = [str(image_dir / f'{step}_secret_image.png')] + [None] * self.num_targets

    #         print(all_imgs_path)

    #         # Create image conditioning.
    #         image_cond = {
    #             "img": all_imgs_path,
    #             "input_indices": self.input_indices,
    #             "prior_indices": self.anchor_indices,
    #         }
    #         # Create camera conditioning.
    #         camera_cond = {
    #             "c2w": self.seva_c2ws.clone(),
    #             "K": self.Ks.clone(),
    #             "input_indices": list(range(self.num_inputs + self.num_targets)),
    #         }

    #         # run_one_scene -> transform_img_and_K modifies VERSION_DICT["H"] and VERSION_DICT["W"] in-place.
    #         video_path_generator = run_one_scene(
    #             self.task,
    #             self.VERSION_DICT,  # H, W maybe updated in run_one_scene
    #             model=self.MODEL,
    #             ae=self.AE,
    #             conditioner=self.CONDITIONER,
    #             denoiser=self.DENOISER,
    #             image_cond=image_cond,
    #             camera_cond=camera_cond,
    #             save_path=image_dir / f'{step}_seva',
    #             use_traj_prior=True,
    #             traj_prior_Ks=self.anchor_Ks,
    #             traj_prior_c2ws=self.anchor_c2ws,
    #             seed=self.seed,
    #         )
    #         for _ in video_path_generator:
    #             pass

    #         # load seva images
    #         # images in 00x.png 's format under image_dir / samples-rgb / f'{step}_seva' folder
    #         self.rgb_list_seva = []
    #         for i in range(self.num_targets + 1):
    #             image_path = image_dir / f'{step}_seva/samples-rgb/00{i}.png'
    #             image = Image.open(image_path).convert('RGB')
    #             transform = transforms.ToTensor()  # Converts PIL to [C, H, W] and [0, 1]
    #             rgb_tensor = transform(image)
    #             self.rgb_list_seva.append(rgb_tensor)

    #         row1 = torch.cat([self.rgb_list_seva[8], self.rgb_list_seva[7], self.rgb_list_seva[6]], dim=2)
    #         row2 = torch.cat([self.rgb_list_seva[5], self.rgb_list_seva[0], self.rgb_list_seva[4]], dim=2)
    #         row3 = torch.cat([self.rgb_list_seva[3], self.rgb_list_seva[2], self.rgb_list_seva[1]], dim=2)  # concat along W

    #         # Now stack the three rows along H to get a single [3, 3H, 3W] image
    #         img = torch.cat([row1, row2, row3], dim=1)  # concat along H

    #         save_image(img.clamp(0, 1), image_dir / f'{step}_ves_seva_image.png')

    #         # Add SEVA images to dataloader
    #         # Update cached_train and create data entries for each SEVA view
    #         for i, (seva_image, ves_camera) in enumerate(zip(self.rgb_list_seva, self.ves_cameras)):
    #             # Convert from [C, H, W] to [H, W, C] for dataloader format
    #             seva_image_hwc = seva_image.permute(1, 2, 0).to(self.config_secret.device).to(self.original_image_secret.dtype)
                
    #             # Get the corresponding VES view index from predefined indices
    #             view_idx = self.ves_view_indices[i]
                
    #             # Update cached_train with SEVA image
    #             self.datamanager.cached_train[view_idx]["image"] = seva_image_hwc
                    
    #     # also update seva views in normal steps
    #     for i, (seva_image, ves_camera) in enumerate(zip(self.rgb_list_seva, self.ves_cameras)):
    #         # Convert from [C, H, W] to [H, W, C] for dataloader format
    #         seva_image_hwc = seva_image.permute(1, 2, 0).to(self.config_secret.device).to(self.original_image_secret.dtype)
            
    #         # Get the corresponding VES view index from predefined indices
    #         view_idx = self.ves_view_indices[i]

    #         # Create data dict for this SEVA view for loss computation
    #         data_seva = {
    #             "image": seva_image_hwc,
    #             "idx": view_idx,
    #             "is_ves_view": True,  # Flag to identify VES views
    #         }

    #         # Get model outputs for this SEVA view
    #         model_outputs_seva = self.model(ves_camera)
            
    #         # Compute metrics and loss for this SEVA view
    #         # Note: We're only computing image-based losses, not depth/normal losses
    #         metrics_dict_seva = self.model.get_metrics_dict(model_outputs_seva, data_seva)
            
    #         # Create a custom loss dict that only includes image-based losses
    #         loss_dict_seva = {}
            
    #         # loss for seva views
    #         # Only compute image-based losses for VES views
    #         if "rgb_loss" in self.model.get_loss_dict(model_outputs_seva, data_seva, metrics_dict_seva):
    #             # Get the full loss dict first
    #             full_loss_dict = self.model.get_loss_dict(model_outputs_seva, data_seva, metrics_dict_seva)
                
    #             # Filter to only include image-based losses (skip depth/normal losses)
    #             for k, v in full_loss_dict.items():
    #                 if any(term in k.lower() for term in ["rgb", "image", "psnr", "ssim", "lpips"]):
    #                     loss_dict_seva[k] = v
    #         else:
    #             # If the model doesn't separate losses, compute a L1 + lpips loss
    #             rgb_pred = model_outputs_seva["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2) # [1, 3, H, W], [0 ,1]
    #             rgb_gt = data_seva["image"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             # loss_dict_seva["rgb_loss"] = torch.nn.functional.mse_loss(rgb_pred, rgb_gt)
    #             loss_dict_seva["rgb_loss"] = torch.nn.functional.l1_loss(rgb_pred, rgb_gt) + 0.1 * self.lpips_loss_fn(2 * rgb_pred - 1, 2 *  rgb_gt - 1) # make them normalized to [-1, 1]
            
    #         # Add to main dicts with unique keys
    #         for k, v in metrics_dict_seva.items():
    #             metrics_dict[f"seva_view_{i}_{k}"] = v
    #         for k, v in loss_dict_seva.items():
    #             loss_dict[f"seva_view_{i}_{k}"] = v

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # comment this function for 1st stage updating, __IGS2GS + IN2N__pie__
    # we can use only IN2N when the number of images in the dataset is small, since IGS2GS + IN2N has longer training time but same results with IN2N in this case.
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_secret_image_cond_{self.config_secret.image_guidance_scale_ip2p_ptd}_async_{self.config_secret.async_ahead_steps}_contrast_{self.ip2p_ptd.contrast}_non_secret_{self.config_secret.image_guidance_scale_ip2p}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 # Convert mask to numpy
    #                 if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                     # It's a PyTorch tensor
    #                     mask_tensor = self.ip2p_ptd.mask
    #                     if mask_tensor.dim() == 4:
    #                         mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                     elif mask_tensor.dim() == 3:
    #                         mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                     else:
    #                         mask_np = mask_tensor.cpu().numpy()
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                     # It's a PIL Image
    #                     mask_np = np.array(self.ip2p_ptd.mask)
    #                     # Convert to grayscale if needed
    #                     if mask_np.ndim == 3:
    #                         mask_np = mask_np[:, :, 0]  # Take first channel
    #                     # Ensure it's uint8
    #                     if mask_np.dtype != np.uint8:
    #                         if mask_np.max() <= 1.0:
    #                             mask_np = (mask_np * 255).astype(np.uint8)
    #                         else:
    #                             mask_np = mask_np.astype(np.uint8)
    #                 else:
    #                     # If it's already numpy, just use it
    #                     mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

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
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

                

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    # __IGS2GS_IN2N_pie_edge__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_edge_secret_image_cond_{self.config_secret.image_guidance_scale_ip2p_ptd}_async_{self.config_secret.async_ahead_steps}_contrast_{self.ip2p_ptd.contrast}_non_secret_{self.config_secret.image_guidance_scale_ip2p}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
                    
    #                 # edge loss
    #                 rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #                 edge_loss = self.edge_loss_fn(
    #                     rendered_image_secret.to(self.config_secret.device), 
    #                     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                     self.original_secret_edges.to(self.config_secret.device),
    #                     # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                     image_dir,
    #                     step
    #                 )
    #                 loss_dict["main_loss"] += edge_loss
                    
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 # Convert mask to numpy
    #                 if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                     # It's a PyTorch tensor
    #                     mask_tensor = self.ip2p_ptd.mask
    #                     if mask_tensor.dim() == 4:
    #                         mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                     elif mask_tensor.dim() == 3:
    #                         mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                     else:
    #                         mask_np = mask_tensor.cpu().numpy()
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                     # It's a PIL Image
    #                     mask_np = np.array(self.ip2p_ptd.mask)
    #                     # Convert to grayscale if needed
    #                     if mask_np.ndim == 3:
    #                         mask_np = mask_np[:, :, 0]  # Take first channel
    #                     # Ensure it's uint8
    #                     if mask_np.dtype != np.uint8:
    #                         if mask_np.max() <= 1.0:
    #                             mask_np = (mask_np * 255).astype(np.uint8)
    #                         else:
    #                             mask_np = mask_np.astype(np.uint8)
    #                 else:
    #                     # If it's already numpy, just use it
    #                     mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
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

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step
    #             )
    #             loss_dict["main_loss"] += edge_loss

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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

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
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             # print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        #####################################################

    # __IGS2GS_IN2N_pie_edge_camera_pose_offset_updating__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_edge_camera_pose_offset_updating"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     if self.first_step:
    #         self.first_step = False
    #         # find the best secret view that align with the reference image
    #         current_secret_idx = 0
    #         current_score = float("inf")

    #         # Lists to store scores for all views
    #         all_lpips_scores = []
    #         all_l1_scores = []
    #         view_indices = []
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             # compute masked lpips value
    #             mask_np = self.ip2p_ptd.mask
    #             # Convert mask to tensor and ensure it's the right shape/device
    #             mask_tensor = torch.from_numpy(mask_np).float()
    #             if len(mask_tensor.shape) == 2:
    #                 mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #             if mask_tensor.shape[0] == 1:
    #                 mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #             mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #             # Prepare model output
    #             model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #             # Apply mask to both images
    #             masked_model_rgb = model_rgb * mask_tensor
    #             masked_ref_image = self.ref_image_tensor * mask_tensor

    #             # Compute masked LPIPS score
    #             lpips_score = self.lpips_loss_fn(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             l1_score = torch.nn.functional.l1_loss(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             # Store scores for plotting
    #             all_lpips_scores.append(lpips_score)
    #             all_l1_scores.append(l1_score)
    #             view_indices.append(idx)

    #             score = l1_score
    #             # score = lpips_score

    #             if score < current_score:
    #                 current_score = score
    #                 current_secret_idx = idx

    #         # Create and save score curves
    #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    #         # Plot LPIPS scores
    #         ax1.plot(view_indices, all_lpips_scores, 'b-', linewidth=2, label='LPIPS Score')
    #         ax1.axvline(x=current_secret_idx, color='r', linestyle='--', linewidth=2, 
    #                 label=f'Best View (idx={current_secret_idx})')
    #         ax1.scatter([current_secret_idx], [current_score], color='red', s=100, zorder=5)
    #         ax1.set_xlabel('View Index')
    #         ax1.set_ylabel('LPIPS Score')
    #         ax1.set_title('LPIPS Scores Across All Views')
    #         ax1.grid(True, alpha=0.3)
    #         ax1.legend()

    #         # Plot L1 scores
    #         ax2.plot(view_indices, all_l1_scores, 'g-', linewidth=2, label='L1 Score')
    #         best_l1_idx = view_indices[np.argmin(all_l1_scores)]
    #         best_l1_score = min(all_l1_scores)
    #         ax2.axvline(x=best_l1_idx, color='orange', linestyle='--', linewidth=2, 
    #                 label=f'Best L1 View (idx={best_l1_idx})')
    #         ax2.scatter([best_l1_idx], [best_l1_score], color='orange', s=100, zorder=5)
    #         ax2.set_xlabel('View Index')
    #         ax2.set_ylabel('L1 Score')
    #         ax2.set_title('L1 Scores Across All Views')
    #         ax2.grid(True, alpha=0.3)
    #         ax2.legend()

    #         plt.tight_layout()
    #         plt.savefig(image_dir / 'score_curves_comparison.png', dpi=300, bbox_inches='tight')
    #         plt.close()

    #         self.secret_view_idx = current_secret_idx
    #         # self.secret_view_idx = self.config_secret.secret_view_idx
    #         camera, data = self.datamanager.next_train_idx(self.secret_view_idx)
    #         model_outputs = self.model(camera)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         save_image(rendered_image, image_dir / f"best_secret_view_{self.secret_view_idx}_score_{current_score}.png")

    #         CONSOLE.print(f"Best secret view index: {self.secret_view_idx} with score: {current_score}")

    #         # start pose offset updating
    #         # Get the secret view camera and initialize pose offset parameters
    #         camera_secret, data_secret = self.datamanager.next_train_idx(self.secret_view_idx)

    #         original_pose_backup = camera_secret.camera_to_worlds.clone()
            
    #         # Initialize camera pose offset parameters (6DOF: translation + rotation)
    #         if not hasattr(self, 'camera_pose_offset'):
    #             # Translation offset (x, y, z)
    #             self.translation_offset = torch.zeros(3, device=camera_secret.camera_to_worlds.device, requires_grad=True)
    #             # Rotation offset (axis-angle representation)
    #             self.rotation_offset = torch.zeros(3, device=camera_secret.camera_to_worlds.device, requires_grad=True)
                
    #             # Optimizer for camera pose offset
    #             self.pose_optimizer = torch.optim.Adam([self.translation_offset, self.rotation_offset], lr=float(self.config_secret.pose_learning_rate))

    #         # # Before the pose optimization loop, store the gradient state and disable model gradients
    #         # model_param_grad_states = {}
    #         # for name, param in self.model.named_parameters():
    #         #     model_param_grad_states[name] = param.requires_grad
    #         #     param.requires_grad = False

    #         # Camera pose optimization loop
    #         num_pose_iterations = self.config_secret.num_pose_iterations
            
    #         for pose_iter in range(num_pose_iterations):
    #             self.pose_optimizer.zero_grad()
                
    #             # Create rotation matrix from axis-angle representation
    #             rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                
    #             # Apply rotation offset: R_new = R_offset @ R_original
    #             original_rotation = original_pose_backup[0, :3, :3]
    #             new_rotation = rotation_matrix @ original_rotation
                
    #             # Apply translation offset: t_new = t_original + t_offset
    #             original_translation = original_pose_backup[0, :3, 3]
    #             new_translation = original_translation + self.translation_offset
                
    #             # Construct new camera-to-world matrix
    #             new_c2w = original_pose_backup.clone()
    #             new_c2w[0, :3, :3] = new_rotation
    #             new_c2w[0, :3, 3] = new_translation
                
    #             camera_secret.camera_to_worlds = new_c2w
                
    #             # Render with updated camera pose
    #             with torch.enable_grad():
    #                 model_outputs = self.model(camera_secret)
                    
    #                 # Compute LPIPS loss with mask
    #                 mask_np = self.ip2p_ptd.mask
    #                 mask_tensor = torch.from_numpy(mask_np).float()
    #                 if len(mask_tensor.shape) == 2:
    #                     mask_tensor = mask_tensor.unsqueeze(0)
    #                 if mask_tensor.shape[0] == 1:
    #                     mask_tensor = mask_tensor.repeat(3, 1, 1)
    #                 mask_tensor = mask_tensor.unsqueeze(0)
    #                 mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #                 model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1) # don't detach here, or the gradients won't exist
    #                 masked_model_rgb = model_rgb * mask_tensor
    #                 masked_ref_image = self.ref_image_tensor * mask_tensor
                    
    #                 lpips_loss = self.lpips_loss_fn(masked_model_rgb, masked_ref_image)
    #                 l1_loss = torch.nn.functional.l1_loss(masked_model_rgb, masked_ref_image)
                    
    #                 # Add regularization to prevent large offsets
    #                 translation_reg = torch.norm(self.translation_offset) * float(self.config_secret.translation_reg_weight)
    #                 rotation_reg = torch.norm(self.rotation_offset) * float(self.config_secret.rotation_reg_weight)
    #                 total_loss = lpips_loss + translation_reg + rotation_reg
    #                 # total_loss = lpips_loss
    #                 # total_loss = l1_loss  + translation_reg + rotation_reg
                    
    #                 # Backward pass and optimization step
    #                 total_loss.backward()
    #                 self.pose_optimizer.step()
                
    #             # Optional: clamp offsets to reasonable ranges
    #             with torch.no_grad():
    #                 self.translation_offset.clamp_(-self.config_secret.max_translation_offset, 
    #                                             self.config_secret.max_translation_offset)
    #                 self.rotation_offset.clamp_(-self.config_secret.max_rotation_offset, 
    #                                         self.config_secret.max_rotation_offset)
                
    #             if pose_iter % 200 == 0 or pose_iter == num_pose_iterations - 1:
    #                 with torch.no_grad():
    #                     CONSOLE.print(
    #                         # f"Translation gradient norm: {self.translation_offset.grad.norm().item()}",
    #                         # f"Rotation gradient norm: {self.rotation_offset.grad.norm().item()}",
    #                         f"Pose iter {pose_iter}: total loss = {total_loss.item():.6f}, "
    #                         f"Trans offset norm = {torch.norm(self.translation_offset).item():.6f}, "
    #                         f"Rot offset norm = {torch.norm(self.rotation_offset).item():.6f}"
    #                     )

    #                     rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                    
    #                     new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #                     new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                        
    #                     new_c2w = original_pose_backup.clone()
    #                     new_c2w[0, :3, :3] = new_rotation
    #                     new_c2w[0, :3, 3] = new_translation
                        
    #                     camera_secret.camera_to_worlds = new_c2w
    #                     optimized_camera = camera_secret
                        
    #                     final_outputs = self.model(optimized_camera)
    #                     rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                        
    #                     # Compute final LPIPS score
    #                     model_rgb = (final_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)
    #                     masked_model_rgb = model_rgb * mask_tensor
                        
    #                     save_image(rendered_image, 
    #                             image_dir / f"optimized_secret_view_{self.secret_view_idx}_step_{pose_iter}_loss_{total_loss.item():.6f}.png")
            
    #         # # After optimization, restore model parameter gradient states
    #         # for name, param in self.model.named_parameters():
    #         #     param.requires_grad = model_param_grad_states[name]    

    #         # secret data preparation
    #         with torch.no_grad():
    #             rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                
    #             new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #             new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                
    #             new_c2w = original_pose_backup.clone()
    #             new_c2w[0, :3, :3] = new_rotation
    #             new_c2w[0, :3, 3] = new_translation
                
    #             camera_secret.camera_to_worlds = new_c2w
    #             optimized_camera = camera_secret
                
    #             final_outputs = self.model(optimized_camera)
    #             rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)  
                
    #             # secret data preparation
    #             self.camera_secret, self.data_secret = optimized_camera, data_secret
    #             self.original_image_secret = rendered_image # [bs, c, h, w]
    #             self.depth_image_secret = final_outputs["depth"].detach().permute(2, 0, 1) # [bs, h, w]
    #             # original secret edges
    #             self.original_secret_edges = SobelFilter(ksize=3, use_grayscale=self.config_secret.use_grayscale)(self.original_image_secret)

    #         torch.cuda.empty_cache()

    #     # start editing
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #     # if (self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.secret_view_idx]

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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
                    
    #                 # edge loss
    #                 rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #                 edge_loss = self.edge_loss_fn(
    #                     rendered_image_secret.to(self.config_secret.device), 
    #                     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                     self.original_secret_edges.to(self.config_secret.device),
    #                     # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                     image_dir,
    #                     step
    #                 )
    #                 loss_dict["main_loss"] += edge_loss

    #                 # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #                 ref_loss = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #                 loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss
                    
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
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

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step
    #             )
    #             loss_dict["main_loss"] += edge_loss

    #             # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #             ref_loss = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #             loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss

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
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.secret_view_idx:
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             # print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
    ######################################################

    # __IGS2GS_IN2N_pie_edge_ref_best__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_edge_ref__best_secret_image_cond_{self.config_secret.image_guidance_scale_ip2p_ptd}_async_{self.config_secret.async_ahead_steps}_contrast_{self.ip2p_ptd.contrast}_non_secret_{self.config_secret.image_guidance_scale_ip2p}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if self.first_step:
    #         self.first_step = False
    #         # find the best secret view that align with the reference image
    #         current_secret_idx = 0
    #         current_lpips_score = float("inf")
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             # compute masked lpips value
    #             mask_np = self.ip2p_ptd.mask
    #             # Convert mask to tensor and ensure it's the right shape/device
    #             mask_tensor = torch.from_numpy(mask_np).float()
    #             if len(mask_tensor.shape) == 2:
    #                 mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #             if mask_tensor.shape[0] == 1:
    #                 mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #             mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #             # Prepare model output
    #             model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #             # Apply mask to both images
    #             masked_model_rgb = model_rgb * mask_tensor
    #             masked_ref_image = self.ref_image_tensor * mask_tensor

    #             # Compute masked LPIPS score
    #             lpips_score = self.lpips_loss_fn(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             # # unmasked lpips score
    #             # lpips_score = self.lpips_loss_fn(
    #             #     (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #             #     self.ref_image_tensor
    #             # ).item()

    #             if lpips_score < current_lpips_score:
    #                 current_lpips_score = lpips_score
    #                 current_secret_idx = idx

    #         self.secret_view_idx = current_secret_idx
    #         camera, data = self.datamanager.next_train_idx(self.secret_view_idx)
    #         model_outputs = self.model(camera)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         save_image(rendered_image, image_dir / f"best_secret_view_{self.secret_view_idx}_lpips_score_{current_lpips_score}.png")

    #         CONSOLE.print(f"Best secret view index: {self.secret_view_idx} with LPIPS score: {current_lpips_score}")

    #         # secret data preparation
    #         self.camera_secret, self.data_secret = self.datamanager.next_train_idx(self.secret_view_idx)
    #         self.original_image_secret = self.datamanager.original_cached_train[self.secret_view_idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         self.depth_image_secret = self.datamanager.original_cached_train[self.secret_view_idx]["depth"] # [bs, h, w]
    #         # original secret edges
    #         self.original_secret_edges = SobelFilter(ksize=3, use_grayscale=self.config_secret.use_grayscale)(self.original_image_secret)

    #     if (not self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.secret_view_idx]

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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
                    
    #                 # edge loss
    #                 rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #                 edge_loss = self.edge_loss_fn(
    #                     rendered_image_secret.to(self.config_secret.device), 
    #                     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                     self.original_secret_edges.to(self.config_secret.device),
    #                     # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                     image_dir,
    #                     step
    #                 )
    #                 loss_dict["main_loss"] += edge_loss

    #                 # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #                 ref_loss = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #                 loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss
                    
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
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

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step
    #             )
    #             loss_dict["main_loss"] += edge_loss

    #             # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #             ref_loss = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #             loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss

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
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.secret_view_idx:
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             # print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        #####################################################

    # __IGS2GS_IN2N_pie_edge_ref_
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_edge_ref_secret_image_cond_{self.config_secret.image_guidance_scale_ip2p_ptd}_async_{self.config_secret.async_ahead_steps}_contrast_{self.ip2p_ptd.contrast}_non_secret_{self.config_secret.image_guidance_scale_ip2p}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
                    
    #                 # edge loss
    #                 rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #                 edge_loss = self.edge_loss_fn(
    #                     rendered_image_secret.to(self.config_secret.device), 
    #                     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                     self.original_secret_edges.to(self.config_secret.device),
    #                     # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                     image_dir,
    #                     step
    #                 )
    #                 loss_dict["main_loss"] += edge_loss

    #                 # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #                 ref_loss = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #                 loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss
                    
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 # Convert mask to numpy
    #                 if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                     # It's a PyTorch tensor
    #                     mask_tensor = self.ip2p_ptd.mask
    #                     if mask_tensor.dim() == 4:
    #                         mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                     elif mask_tensor.dim() == 3:
    #                         mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                     else:
    #                         mask_np = mask_tensor.cpu().numpy()
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                     # It's a PIL Image
    #                     mask_np = np.array(self.ip2p_ptd.mask)
    #                     # Convert to grayscale if needed
    #                     if mask_np.ndim == 3:
    #                         mask_np = mask_np[:, :, 0]  # Take first channel
    #                     # Ensure it's uint8
    #                     if mask_np.dtype != np.uint8:
    #                         if mask_np.max() <= 1.0:
    #                             mask_np = (mask_np * 255).astype(np.uint8)
    #                         else:
    #                             mask_np = mask_np.astype(np.uint8)
    #                 else:
    #                     # If it's already numpy, just use it
    #                     mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
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

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step
    #             )
    #             loss_dict["main_loss"] += edge_loss

    #             # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #             ref_loss = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #             loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss

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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

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
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             # print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        #####################################################

    # comment this function for 1st stage updating, __IGS2GS + IN2N__seva__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N__seva_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # generate ves views for the first iteration
    #     if self.first_iter:
    #         self.first_iter = False
    #         for i, ves_camera in enumerate(self.ves_cameras):
    #             model_outputs_ves = self.model(ves_camera)
    #             rendered_image_ves = model_outputs_ves["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             save_image((rendered_image_ves).clamp(0, 1), image_dir / f'{step}_ves_image_{i}.png')

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
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
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_render.png')
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

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
    #                 secret_idx=0,
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
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # seva results
    #             all_imgs_path = [str(image_dir / f'{step}_secret_image.png')] + [None] * self.num_targets

    #             print(all_imgs_path)

    #             # Create image conditioning.
    #             image_cond = {
    #                 "img": all_imgs_path,
    #                 "input_indices": self.input_indices,
    #                 "prior_indices": self.anchor_indices,
    #             }
    #             # Create camera conditioning.
    #             camera_cond = {
    #                 "c2w": self.seva_c2ws.clone(),
    #                 "K": self.Ks.clone(),
    #                 "input_indices": list(range(self.num_inputs + self.num_targets)),
    #             }

    #             # run_one_scene -> transform_img_and_K modifies VERSION_DICT["H"] and VERSION_DICT["W"] in-place.
    #             video_path_generator = run_one_scene(
    #                 self.task,
    #                 self.VERSION_DICT,  # H, W maybe updated in run_one_scene
    #                 model=self.MODEL,
    #                 ae=self.AE,
    #                 conditioner=self.CONDITIONER,
    #                 denoiser=self.DENOISER,
    #                 image_cond=image_cond,
    #                 camera_cond=camera_cond,
    #                 save_path=image_dir / f'{step}_seva',
    #                 use_traj_prior=True,
    #                 traj_prior_Ks=self.anchor_Ks,
    #                 traj_prior_c2ws=self.anchor_c2ws,
    #                 seed=self.seed,
    #             )
    #             for _ in video_path_generator:
    #                 pass

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################
    
    # comment this function for 1st stage updating, __IGS2GS + IN2N__2_secrets__
    # we can use only IN2N when the number of images in the dataset is small, since IGS2GS + IN2N has longer training time but same results with IN2N in this case.
    # def get_train_loss_dict(self, step: int):        
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_2_secrets_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # generate ves views for the first iteration
    #     if self.first_iter:
    #         self.first_iter = False
    #         for i, ves_camera in enumerate(self.ves_cameras):
    #             model_outputs_ves = self.model(ves_camera)
    #             rendered_image_ves = model_outputs_ves["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             save_image((rendered_image_ves).clamp(0, 1), image_dir / f'{step}_ves_image_{i}.png')

    #         model_outputs_secret = self.model(self.camera_secret)
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         save_image((rendered_image_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
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
    #                     loss_dict[f"secret_{k}"] = v * self.config_secret.secret_loss_weight

    #                 # also edit the second secret view
    #                 model_outputs_secret_2 = self.model(self.camera_secret_2)
    #                 metrics_dict_secret_2 = self.model.get_metrics_dict(model_outputs_secret_2, self.data_secret_2)
    #                 loss_dict_secret_2 = self.model.get_loss_dict(model_outputs_secret_2, self.data_secret_2, metrics_dict_secret_2)
    #                 rendered_image_secret_2 = model_outputs_secret_2["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret_2, depth_tensor_secret_2 = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret_2.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret_2.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=1,
    #                     depth=self.depth_image_secret_2,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )
    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret_2.size() != rendered_image_secret_2.size()):
    #                     edited_image_secret_2 = torch.nn.functional.interpolate(edited_image_secret_2, size=rendered_image_secret_2.size()[2:], mode='bilinear')
                    
    #                 # write edited image to dataloader
    #                 edited_image_secret_2 = edited_image_secret_2.to(self.original_image_secret_2.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx_2]["image"] = edited_image_secret_2.squeeze().permute(1,2,0)
    #                 self.data_secret_2["image"] = edited_image_secret_2.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret_2.items():
    #                     metrics_dict[f"secret_2_{k}"] = v
    #                 for k, v in loss_dict_secret_2.items():
    #                     loss_dict[f"secret_2_{k}"] = v * self.config_secret.secret_loss_weight
                    
    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')
    #                     image_save_secret_2 = torch.cat([depth_tensor_secret_2, rendered_image_secret_2, edited_image_secret_2.to(self.config_secret.device), self.original_image_secret_2.to(self.config_secret.device)])
    #                     save_image((image_save_secret_2).clamp(0, 1), image_dir / f'{step}_secret_image_2.png')
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
    #                 loss_dict[f"secret_{k}"] = v * self.config_secret.secret_loss_weight

    #             # also update the second secret view w/o editing (important)
    #             model_outputs_secret_2 = self.model(self.camera_secret_2)
    #             metrics_dict_secret_2 = self.model.get_metrics_dict(model_outputs_secret_2, self.data_secret_2)
    #             loss_dict_secret_2 = self.model.get_loss_dict(model_outputs_secret_2, self.data_secret_2, metrics_dict_secret_2)

    #             for k, v in metrics_dict_secret_2.items():
    #                 metrics_dict[f"secret_2_{k}"] = v
    #             for k, v in loss_dict_secret_2.items():
    #                 loss_dict[f"secret_2_{k}"] = v * self.config_secret.secret_loss_weight

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx and idx != self.config_secret.secret_view_idx_2:
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

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
    #                 secret_idx=0,
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
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         if idx == self.config_secret.secret_view_idx_2:
    #             model_outputs_secret_2 = self.model(self.camera_secret_2)
    #             metrics_dict_secret_2 = self.model.get_metrics_dict(model_outputs_secret_2, self.data_secret_2)
    #             loss_dict_secret_2 = self.model.get_loss_dict(model_outputs_secret_2, self.data_secret_2, metrics_dict_secret_2)
    #             rendered_image_secret_2 = model_outputs_secret_2["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret_2, depth_tensor_secret_2 = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret_2.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret_2.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=1,
    #                 depth=self.depth_image_secret_2,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )
    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret_2.size() != rendered_image_secret_2.size()):
    #                 edited_image_secret_2 = torch.nn.functional.interpolate(edited_image_secret_2, size=rendered_image_secret_2.size()[2:], mode='bilinear')
                
    #             # write edited image to dataloader
    #             edited_image_secret_2 = edited_image_secret_2.to(self.original_image_secret_2.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx_2]["image"] = edited_image_secret_2.squeeze().permute(1,2,0)
    #             self.data_secret_2["image"] = edited_image_secret_2.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret_2.items():
    #                 metrics_dict[f"secret_2_{k}"] = v
    #             for k, v in loss_dict_secret_2.items():
    #                 loss_dict[f"secret_2_{k}"] = v
                
    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret_2, rendered_image_secret_2, edited_image_secret_2.to(self.config_secret.device), self.original_image_secret_2.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image_2.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # comment this function for 1st stage updating, __IGS2GS + IN2N__
    # we can use only IN2N when the number of images in the dataset is small, since IGS2GS + IN2N has longer training time but same results with IN2N in this case.
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # generate ves views for the first iteration
    #     # if self.first_iter:
    #     #     self.first_iter = False
    #     #     for i, ves_camera in enumerate(self.ves_cameras):
    #     #         model_outputs_ves = self.model(ves_camera)
    #     #         rendered_image_ves = model_outputs_ves["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #     #         save_image((rendered_image_ves).clamp(0, 1), image_dir / f'{step}_ves_image_{i}.png')

    #     #     model_outputs_secret = self.model(self.camera_secret)
    #     #     rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #     #     save_image((rendered_image_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
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
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')
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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

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
    #                 secret_idx=0,
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
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    # comment this function for 1st stage updating, __IN2N__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IN2N_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # print(step) # start from 30000

    #     # the implementation of randomly selecting an index to edit instead of update all images at once
    #     # generate the indexes for non-secret view editing
    #     all_indices = np.arange(len(self.datamanager.cached_train))
    #     allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

    #     if step % self.config_secret.edit_rate == 0:
    #         #----------------non-secret view editing----------------
    #         # randomly select an index to edit
    #         idx = random.choice(allowed)
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #         edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #             self.text_embeddings_ip2p.to(self.config_secret.device),
    #             rendered_image.to(self.dtype),
    #             original_image.to(self.config_secret.device).to(self.dtype),
    #             False, # is depth tensor
    #             depth_image,
    #             guidance_scale=self.config_secret.guidance_scale,
    #             image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #             diffusion_steps=self.config_secret.t_dec,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound,
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image.size() != rendered_image.size()):
    #             edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image = edited_image.to(original_image.dtype)
    #         self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #         data["image"] = edited_image.squeeze().permute(1,2,0)

    #         # save edited non-secret image
    #         if step % 50 == 0:
    #             image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         #----------------secret view editing----------------
    #         if step % self.config_secret.secret_edit_rate == 0:
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
    #             if step % 50 == 0:
    #                 image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                 save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')
    #                 save_image(edited_image_secret.to(self.config_secret.device).clamp(0, 1), image_dir / f'{step}_secret_image_seva.png')
    #     else:
    #         # non-editing steps loss computing
    #         camera, data = self.datamanager.next_train(step)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # also update the secret view w/o editing (important)
    #         model_outputs_secret = self.model(self.camera_secret)
    #         metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #         loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #         for k, v in metrics_dict_secret.items():
    #             metrics_dict[f"secret_{k}"] = v
    #         for k, v in loss_dict_secret.items():
    #             loss_dict[f"secret_{k}"] = v

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    # comment this function for 1st stage updating, __IGS2GS__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)
    
    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0: # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         # update the non-secret views w/o editing
    #         camera, data = self.datamanager.next_train(step)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         if step % 500 == 0:
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             save_image((rendered_image).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

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
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

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
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

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
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False

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


    