if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import json
import pathlib

import click
import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        if image.dtype == np.uint8:
            return image
        image = image.astype(np.float32, copy=False)
        if image.max(initial=0.0) <= 1.0 and image.min(initial=0.0) >= 0.0:
            image = image * 255.0
        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape={image.shape}")

    channel_axes = [i for i, dim in enumerate(image.shape) if dim == 3]
    if len(channel_axes) != 1:
        raise ValueError(f"Cannot infer channel axis from shape={image.shape}")
    image = np.moveaxis(image, channel_axes[0], -1)

    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32, copy=False)
    if image.max(initial=0.0) <= 1.0 and image.min(initial=0.0) >= 0.0:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


def _save_image(image: np.ndarray, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_hwc_uint8(image)).save(str(path))


def _save_side_by_side(left: np.ndarray, right: np.ndarray, path: pathlib.Path) -> None:
    left_hwc = _to_hwc_uint8(left)
    right_hwc = _to_hwc_uint8(right)
    h = max(left_hwc.shape[0], right_hwc.shape[0])
    w = left_hwc.shape[1] + right_hwc.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: left_hwc.shape[0], : left_hwc.shape[1]] = left_hwc
    canvas[: right_hwc.shape[0], left_hwc.shape[1] : left_hwc.shape[1] + right_hwc.shape[1]] = right_hwc
    _save_image(canvas, path)


def _save_channel_images(raw_image: np.ndarray, formatted: np.ndarray, out_dir: pathlib.Path, obs_key: str) -> None:
    # raw_image assumed HWC-like, formatted assumed CHW-like after wrapper conversion.
    raw_hwc = _to_hwc_uint8(raw_image)
    if formatted.ndim != 3:
        return
    formatted_chw = np.asarray(formatted)
    if formatted_chw.shape[0] != 3:
        return
    formatted_hwc = _to_hwc_uint8(formatted_chw)
    for c, name in enumerate(["R", "G", "B"]):
        _save_image(raw_hwc[:, :, c], out_dir / f"{obs_key}_raw_channel_{name}.png")
        _save_image(formatted_hwc[:, :, c], out_dir / f"{obs_key}_formatted_channel_{name}.png")


def _get_shape_meta(cfg):
    if "shape_meta" in cfg:
        return OmegaConf.to_container(cfg.shape_meta, resolve=True)
    if "task" in cfg and "shape_meta" in cfg.task:
        return OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    raise ValueError("Could not find shape_meta in config.")


def _build_env_from_dataset(dataset_path: pathlib.Path, shape_meta: dict):
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    env_kwargs = env_meta["env_kwargs"]

    # Match runner behavior so env creation works with robosuite composite controllers.
    for val in shape_meta["obs"].values():
        if len(val["shape"]) == 3:
            env_kwargs["camera_heights"] = val["shape"][-1]
            env_kwargs["camera_widths"] = val["shape"][-1]
            break
    env_kwargs["use_object_obs"] = False

    modality_mapping = dict()
    for key, attr in shape_meta["obs"].items():
        modality = attr.get("type", "low_dim")
        modality_mapping.setdefault(modality, []).append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    robots = env_kwargs.get("robots", ["Panda"])
    robot_type = robots[0] if isinstance(robots, (list, tuple)) else robots
    arms = ["right"]
    env_cfg = env_kwargs["controller_configs"]
    env_cfg.setdefault("input_type", "delta")
    env_cfg.setdefault("input_ref_frame", "base")
    env_cfg.setdefault("damping_ratio", env_cfg.get("damping", 1))
    env_cfg.setdefault("damping_ratio_limits", env_cfg.get("damping_limits", [0, 10]))
    env_kwargs["controller_configs"] = refactor_composite_controller_config(
        controller_config=env_cfg,
        robot_type=robot_type,
        arms=arms,
    )

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    return env


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _compute_axis_checks(raw_image: np.ndarray, formatted: np.ndarray) -> dict:
    raw = np.asarray(raw_image)
    fmt = np.asarray(formatted)
    out = {}
    if raw.ndim == 3 and fmt.ndim == 3:
        # Expected conversion if input is HWC uint8-like.
        expected = np.moveaxis(raw, -1, 0).astype(np.float32) / 255.0
        out["max_abs_diff_vs_moveaxis_hwc_to_chw_div255"] = float(np.max(np.abs(fmt - expected)))

        # Check a wrong hypothesis: transposing H/W before channel move.
        hw_swapped = np.moveaxis(np.transpose(raw, (1, 0, 2)), -1, 0).astype(np.float32) / 255.0
        out["max_abs_diff_vs_hw_swapped_hwc_to_chw_div255"] = float(np.max(np.abs(fmt - hw_swapped)))

        # Reconstruct uint8 and compare to raw.
        recon = np.clip(np.rint(np.moveaxis(fmt, 0, -1) * 255.0), 0, 255).astype(np.uint8)
        if raw.dtype == np.uint8:
            out["roundtrip_equal_uint8"] = bool(np.array_equal(recon, raw))
            out["roundtrip_num_diff_pixels"] = int(np.count_nonzero(recon != raw))
        else:
            out["roundtrip_equal_uint8"] = None
            out["roundtrip_num_diff_pixels"] = None
    return out


@click.command()
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("-k", "--obs-key", default="agentview_image", show_default=True)
@click.option("-o", "--out-dir", default="data/debug_image_format", show_default=True)
@click.option("--mode", type=click.Choice(["env", "dataset"]), default="env", show_default=True)
@click.option("--dataset-path", default=None, help="Override dataset path from config.")
@click.option("--demo-idx", default=0, show_default=True, type=int)
@click.option("--timestep", default=0, show_default=True, type=int)
@click.option("--delta", default=0.0, show_default=True, type=float, help="Camera x-offset for modify_environment in env mode.")
def main(config_path, obs_key, out_dir, mode, dataset_path, demo_idx, timestep, delta):
    config_path = pathlib.Path(config_path).expanduser()
    out_dir = pathlib.Path(out_dir).expanduser()
    cfg = OmegaConf.load(str(config_path))
    shape_meta = _get_shape_meta(cfg)

    if dataset_path is None:
        dataset_path = cfg.task.dataset_path
    dataset_path = pathlib.Path(dataset_path).expanduser()

    expected_shape = tuple(shape_meta["obs"][obs_key]["shape"])

    if mode == "dataset":
        with h5py.File(str(dataset_path), "r") as f:
            raw_image = f[f"data/demo_{demo_idx}/obs/{obs_key}"][timestep]
        formatter = RobomimicImageWrapper.__dict__["_format_image_obs"]
        fake_wrapper = type("FakeWrapper", (), {"shape_meta": shape_meta})()
        formatted = formatter(fake_wrapper, obs_key, raw_image)
    else:
        env = _build_env_from_dataset(dataset_path, shape_meta)
        wrapper = RobomimicImageWrapper(
            env=env,
            shape_meta=shape_meta,
            init_state=None,
            render_obs_key=obs_key,
        )
        raw_obs = env.reset()
        if abs(delta) > 0:
            wrapper.modify_environment(delta=delta, render_obs_key=obs_key)
            raw_obs = env.get_observation()
        raw_image = raw_obs[obs_key]
        formatted = wrapper._format_image_obs(obs_key, raw_image)

    before_path = out_dir / f"{obs_key}_before_raw.png"
    after_path = out_dir / f"{obs_key}_after_format.png"
    panel_path = out_dir / f"{obs_key}_before_after.png"
    raw_npy_path = out_dir / f"{obs_key}_before_raw.npy"
    formatted_npy_path = out_dir / f"{obs_key}_after_format.npy"
    stats_path = out_dir / f"{obs_key}_format_stats.json"

    _save_image(raw_image, before_path)
    _save_image(formatted, after_path)
    _save_side_by_side(raw_image, formatted, panel_path)
    _save_channel_images(raw_image, formatted, out_dir, obs_key)
    np.save(raw_npy_path, np.asarray(raw_image))
    np.save(formatted_npy_path, np.asarray(formatted))

    axis_checks = _compute_axis_checks(raw_image, formatted)

    payload = {
        "obs_key": obs_key,
        "mode": mode,
        "expected_shape": list(expected_shape),
        "raw": _stats(raw_image),
        "formatted": _stats(formatted),
        "axis_checks": axis_checks,
        "model_interpretation": "formatted tensor is float32 CHW in [0, 1]",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(payload, indent=2))

    print(f"saved: {before_path}")
    print(f"saved: {after_path}")
    print(f"saved: {panel_path}")
    print(f"saved: {raw_npy_path}")
    print(f"saved: {formatted_npy_path}")
    print(f"saved: {stats_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
