import glob
import os
import os.path as osp

import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import json
import random

from seva.data_io import get_parser
from seva.eval import (
    IS_TORCH_NIGHTLY,
    compute_relative_inds,
    create_transforms_simple,
    infer_prior_inds,
    infer_prior_stats,
    run_one_scene,
)
from seva.geometry import (
    generate_interpolated_path,
    generate_spiral_path,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_lookat,
    get_preset_pose_fov,
)
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model, generate_ves_poses, generate_ves_poses_opengl

device = "cuda:0"

from datetime import datetime

from dreifus.matrix import Intrinsics, Pose, CameraCoordinateConvention, PoseType
import pyvista as pv
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum

# Constants.
WORK_DIR = "work_dirs/demo_stabilization/" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

if IS_TORCH_NIGHTLY:
    COMPILE = True
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
else:
    COMPILE = False

AE = AutoEncoder(chunk_size=1).to(device)
CONDITIONER = CLIPConditioner().to(device)
DENOISER = DiscreteDenoiser(num_idx=1000, device=device)

if COMPILE:
    CONDITIONER = torch.compile(CONDITIONER, dynamic=False)
    AE = torch.compile(AE, dynamic=False)


def parse_task(
    task,
    scene,
    num_inputs,
    T,
    version_dict,
):
    options = version_dict["options"]

    anchor_indices = None
    anchor_c2ws = None
    anchor_Ks = None

    if task == "img2trajvid_s-prob":
        if num_inputs is not None:
            assert (
                num_inputs == 1
            ), "Task `img2trajvid_s-prob` only support 1-view conditioning..."
        else:
            num_inputs = 1
        num_targets = options.get("num_targets", T - 1)
        num_anchors = infer_prior_stats(
            T,
            num_inputs,
            num_total_frames=num_targets,
            version_dict=version_dict,
        )

        input_indices = [0]
        anchor_indices = np.linspace(1, num_targets, num_anchors).tolist()

        all_imgs_path = [scene] + [None] * num_targets

        start_w2c = torch.eye(4)
        look_at = torch.Tensor([0, 0, 10])

        c2ws, fovs = get_preset_pose_fov(
            option=options.get("traj_prior", "orbit"),
            num_frames=num_targets + 1,
            start_w2c=start_w2c,
            look_at=look_at,
        )

        # ------------------------- generate ves poses -------------------------

        # secret_view_idx = 6
        secret_view_name = "DSC00043.png"
        angle_limits = [3.0, 15.0]

        print(f"using angle limit: {angle_limits[0]} degrees for VES")

        # Set paths
        scene_name = '49a82360aa'
        # scene_name = 'fb5a96b1a2'
        # scene_name = '0cf2e9402d'
        DATA_DIR = "./data"

        # ScanNetpp preprocessing
        ScanNetpp_path = DATA_DIR + '/ScanNetpp_512'
        scene_list = [scene_name]

        # go over all scenes
        for scene_id in tqdm(scene_list):
            # load scene transforms
            json_path = os.path.join(ScanNetpp_path, scene_id, "transforms.json")
            with open(json_path, "r") as f:
                data = json.load(f)

            # always use the train frames, for train and val splits. only interested in using different scenes as train/val!
            train_frames = data["frames"]

            # subsample for faster loading
            random.shuffle(train_frames)

            # search for the secret view index
            secret_view_idx = train_frames.index(
                next(frame for frame in train_frames if frame["file_path"] == "./images/" + secret_view_name)
            )
            print(f"Secret view index: {secret_view_idx}")

            # create w2c matrices in opengl / nerfstudio convention
            train_c2w = np.array([np.array(frame["transform_matrix"], dtype=np.float32) for frame in train_frames])

            # generate VES (Viewpoint Ensemble Stabilization) viewpoints
            c2w_secret = train_c2w[secret_view_idx]

            ves_poses = generate_ves_poses_opengl(c2w_secret, angle_limit_degrees=angle_limits[0])

            print(ves_poses)

            # ves_c2w = torch.from_numpy(np.stack(ves_poses, axis=0))
            ves_c2w = np.stack(ves_poses, axis=0)

            # dreifus visualization
            # # construct intrinsics --> gets normalized here s.t. the center cropping aftwards works
            # store_h, store_w = data["h"], data["w"]
            # fx, fy, cx, cy = (
            #     data["fl_x"],
            #     data["fl_y"],
            #     data["cx"],
            #     data["cy"],
            # )
            # normalized_fx = float(fx) / float(store_w)
            # normalized_fy = float(fy) / float(store_h)
            # normalized_cx = float(cx) / float(store_w)
            # normalized_cy = float(cy) / float(store_h)

            # intrinsics = Intrinsics(normalized_fx, normalized_fy, normalized_cx, normalized_cy)

            # p = pv.Plotter()

            # add_coordinate_axes(p)

            # poses = []
            # for extrinsic in ves_c2w:
            #     pose = Pose(extrinsic, pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)
            #     poses.append(pose)

            #     add_camera_frustum(p, pose, intrinsics)
        
            # p.show()

            c2ws = ves_c2w

            # convert from OpenGL to OpenCV camera format
            c2ws = c2ws @ np.diag([1, -1, -1, 1])

        # ------------------------- generate ves poses -------------------------
        
        with Image.open(scene) as img:
            W, H = img.size
            aspect_ratio = W / H
        Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)  # unormalized
        Ks[:, :2] *= (
            torch.tensor([W, H]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
        )  # normalized
        Ks = Ks.numpy()

        anchor_c2ws = c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks = Ks[[round(ind) for ind in anchor_indices]]

    return (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        torch.tensor(c2ws[:, :3]).float(),
        torch.tensor(Ks).float(),
        (torch.tensor(anchor_c2ws[:, :3]).float() if anchor_c2ws is not None else None),
        (torch.tensor(anchor_Ks).float() if anchor_Ks is not None else None),
    )


def main(
    data_path,
    data_items=None,
    version=1.1,
    task="img2img",
    save_subdir="",
    H=None,
    W=None,
    T=None,
    use_traj_prior=False,
    pretrained_model_name_or_path="stabilityai/stable-virtual-camera",
    weight_name="model.safetensors",
    seed=23,
    **overwrite_options,
):
    MODEL = SGMWrapper(
        load_model(
            model_version=version,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            weight_name=weight_name,
            device="cpu",
            verbose=True,
        ).eval()
    ).to(device)

    if COMPILE:
        MODEL = torch.compile(MODEL, dynamic=False)

    VERSION_DICT = {
        "H": H or 576,
        "W": W or 576,
        "T": ([int(t) for t in T.split(",")] if isinstance(T, str) else T) or 21,
        "C": 4,
        "f": 8,
        "options": {
            "chunk_strategy": "nearest-gt",
            "video_save_fps": 30.0,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": 1,
            "cfg": 2.0,
            "camera_scale": 2.0,
            "num_steps": 50,
            "cfg_min": 1.2,
            "encoding_t": 1,
            "decoding_t": 1,
        },
    }

    options = VERSION_DICT["options"]
    options.update(overwrite_options)

    if data_items is not None:
        if not isinstance(data_items, (list, tuple)):
            data_items = data_items.split(",")
        scenes = [os.path.join(data_path, item) for item in data_items]
    else:
        scenes = [
            item for item in glob.glob(osp.join(data_path, "*")) if os.path.isfile(item)
        ]

    for scene in tqdm(scenes):
        num_inputs = options.get("num_inputs", None)
        save_path_scene = os.path.join(
            WORK_DIR, task, save_subdir, os.path.splitext(os.path.basename(scene))[0]
        )
        if options.get("skip_saved", False) and os.path.exists(
            os.path.join(save_path_scene, "transforms.json")
        ):
            print(f"Skipping {scene} as it is already sampled.")
            continue

        # parse_task -> infer_prior_stats modifies VERSION_DICT["T"] in-place.
        (
            all_imgs_path,
            num_inputs,
            num_targets,
            input_indices,
            anchor_indices,
            c2ws,
            Ks,
            anchor_c2ws,
            anchor_Ks,
        ) = parse_task(
            task,
            scene,
            num_inputs,
            VERSION_DICT["T"],
            VERSION_DICT,
        )

        print(save_path_scene)

        assert num_inputs is not None
        # Create image conditioning.
        image_cond = {
            "img": all_imgs_path,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        # Create camera conditioning.
        camera_cond = {
            "c2w": c2ws.clone(),
            "K": Ks.clone(),
            "input_indices": list(range(num_inputs + num_targets)),
        }
        # run_one_scene -> transform_img_and_K modifies VERSION_DICT["H"] and VERSION_DICT["W"] in-place.
        video_path_generator = run_one_scene(
            task,
            VERSION_DICT,  # H, W maybe updated in run_one_scene
            model=MODEL,
            ae=AE,
            conditioner=CONDITIONER,
            denoiser=DENOISER,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=save_path_scene,
            use_traj_prior=use_traj_prior,
            traj_prior_Ks=anchor_Ks,
            traj_prior_c2ws=anchor_c2ws,
            seed=seed,
        )
        for _ in video_path_generator:
            pass

        # Convert from OpenCV to OpenGL camera format. 
        #TODO why should we do the conversion? The c2ws are assumed to be in OpenCV convention?
        c2ws = c2ws @ torch.tensor(np.diag([1, -1, -1, 1])).float()
        img_paths = sorted(glob.glob(osp.join(save_path_scene, "samples-rgb", "*.png")))
        if len(img_paths) != len(c2ws):
            input_img_paths = sorted(
                glob.glob(osp.join(save_path_scene, "input", "*.png"))
            )
            assert len(img_paths) == num_targets
            assert len(input_img_paths) == num_inputs
            assert c2ws.shape[0] == num_inputs + num_targets
            target_indices = [i for i in range(c2ws.shape[0]) if i not in input_indices]
            img_paths = [
                input_img_paths[input_indices.index(i)]
                if i in input_indices
                else img_paths[target_indices.index(i)]
                for i in range(c2ws.shape[0])
            ]
        create_transforms_simple(
            save_path=save_path_scene,
            img_paths=img_paths,
            img_whs=np.array([VERSION_DICT["W"], VERSION_DICT["H"]])[None].repeat(
                num_inputs + num_targets, 0
            ),
            c2ws=c2ws,
            Ks=Ks,
        )


if __name__ == "__main__":
    fire.Fire(main)
