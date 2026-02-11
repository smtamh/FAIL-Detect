import h5py, json
import robomimic.utils.file_utils as FileUtils

## 1. check hdf5 data

p = "data/robomimic/datasets/square/ph/image_abs.hdf5"
f = h5py.File(p, "r")

def walk(name, obj):
		if isinstance(obj, h5py.Dataset):
				if "image" in name or "rgb" in name:
						print(name, obj.shape, obj.dtype)
                                          
# f.visititems(walk)

"""
result:

data/demo_0/next_obs/agentview_image (127, 84, 84, 3) uint8
data/demo_0/next_obs/robot0_eye_in_hand_image (127, 84, 84, 3) uint8
data/demo_0/obs/agentview_image (127, 84, 84, 3) uint8
data/demo_0/obs/robot0_eye_in_hand_image (127, 84, 84, 3) uint8
...
"""

########################################################################

## 2. check env meta
"""
# diffusion_policy/env_runner/robomimic_image_runner.py
- line 79: read env meta from dataset
	- env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
- (modify some env kwargs)
- line 156: make robosuite env (robomimic/envs/env_robosuite.py)
	- env_fns = [env_fn] * n_envs
	- env_fn -> create_env
			 -> robomimic.utils.env_utils.create_env_from_metadata
			 -> robomimic.utils.env_utils.create_env
			 -> robomimic.utils.env_utils.get_env_class
			 -> robomimic.envs.env_robosuite.EnvRobosuite (or EnvGym, etc.)

# robomimic/envs/env_robosuite.py
- line 116: make robosuite env (check env kwargs here)
	- self.env = robosuite.make(self._env_name, **kwargs)

env meta saved as robosuite(<=1.4.1) style controller_configs (converted in robomiic_image_runner.py)
"""

print("top keys:", list(f.keys()))
env_meta = FileUtils.get_env_metadata_from_dataset(p)
print("env meta:", json.dumps(env_meta, indent=4))

"""
top keys: ['data', 'mask']
env meta: {
    "env_name": "NutAssemblySquare",
    "type": 1,
    "env_kwargs": {
        "has_renderer": false,
        "has_offscreen_renderer": true,
        "ignore_done": true,
        "use_object_obs": true,
        "use_camera_obs": true,
        "control_freq": 20,
        "controller_configs": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [
                0.05,
                0.05,
                0.05,
                0.5,
                0.5,
                0.5
            ],
            "output_min": [
                -0.05,
                -0.05,
                -0.05,
                -0.5,
                -0.5,
                -0.5
            ],
            "kp": 150,
            "damping": 1,
            "impedance_mode": "fixed",
            "kp_limits": [
                0,
                300
            ],
            "damping_limits": [
                0,
                10
            ],
            "position_limits": null,
            "orientation_limits": null,
            "uncouple_pos_ori": true,
            "control_delta": true,
            "interpolation": null,
            "ramp_ratio": 0.2
        },
        "robots": [
            "Panda"
        ],
        "camera_depths": false,
        "camera_heights": 84,
        "camera_widths": 84,
        "reward_shaping": false,
        "camera_names": [
            "agentview",
            "robot0_eye_in_hand"
        ],
        "render_gpu_device_id": 0
    }
}

"""

