## Prerequisite

Use `pyproject.toml` to manage the `uv` environment. Do not use `conda_environnment.yaml`.

### 1. Install uv
```
sudo apt install curl
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Clone Repository

Clone this repository.  

```
git clone https://github.com/smtamh/FAIL-Detect.git
```
`pytorch3d` is only used in `FAIL-Detect/diffusion_policy/model/common/rotation_transformer.py`.  
You can either vendor it or add it as a dependency.
```
# vendoring
cd FAIL-Detect/diffusion_policy/model/common/
git clone https://github.com/facebookresearch/pytorch3d.git

# (not recommended) add dependency
uv add "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
```

### 3. Create uv Project
```
cd FAIL-Detect
uv init             # initialize FAIL-Detect as a uv project
uv sync             # uv reads `pyproject.toml` and downloads dependencies in 'FAIL-Detect/.venv'

# 'uv run ...' use 'FAIL-Detect/.venv' automatically.
```

### 4. Download Dataset
**Download**: You can download diffusion-policy datasets from [this link](https://diffusion-policy.cs.columbia.edu/data/training/)  

**Save**: Save your datasets in `FAIL-Detect/data`. (e.g., FAIL-Detect/data/robomimic/datasets/...)  

**Info**: For robomimic datasets, if you want to check hdf5 file, use `check_hdf5_data.py`.

### (5. Using Gymnasium)
You can use `gymnasium` instead of `gym`.
```
uv remove gym       # remove gym in dependency    (automatically deleted in pyproject.toml)
uv add gymnasium    # add gymnasium in dependency (automatically added in pyproject.toml)
```
Then, change some codes:  
```
1. import gym -> import gymnasium
2. from gym.  -> from gymnasium.
3. check the commented sections in FAIL-Detect/diffusion_policy/gym_util/async_vector_env.py
```

## Usage

### 1. Policy training

**Tasks**: we consider `square`, `transport`, `tool_hang`, and `can` tasks in [robomimic](https://robomimic.github.io/).

**Policy backbone**: Either diffusion policy or flow-matching policy. Both policies have the same network architecture and are trained on the same datasets with same hyperparameters.

**Usage**: see `diffusion_policy/configs_robomimic` for the set of configs.

**Config Folder**: The primary config directory is defined in `train.py` as `diffusion_policy/config`.  
You can modify it or use `--config-dir` to specify a different directory.
```
# This trains a flow policy (e.g, on the square task)
uv run train.py --config-dir=diffusion_policy/configs_robomimic --config-name=image_square_ph_visual_flow_policy_cnn.yaml training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}'

# This trains a diffusion policy (e.g, on the square task)
uv run train.py --config-dir=diffusion_policy/configs_robomimic --config-name=image_square_ph_visual_diffusion_policy_cnn.yaml training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}'

# For other tasks, change 'square' to be among ['transport', 'tool_hang', 'can']
```

### 2. Obtain $\{(A_t, O_t)\}$ given a trained policy

Here, 
- $O_t$ = [Embedded visual features, non-visual information (e.g., robot states)]. 
- $A_t$ = corresponding action *in training data*.

```
# For flow policy (e.g, on the square task)
uv run save_data.py --config-dir=diffusion_policy/configs_robomimic \
--config-name=image_square_ph_visual_flow_policy_cnn.yaml \
training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}' 

# For diffusion policy (e.g, on the square task)
uv run save_data.py --config-dir=diffusion_policy/configs_robomimic \
--config-name=image_square_ph_visual_diffusion_policy_cnn.yaml \
training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}'

# For other tasks, change 'square' to be among ['transport', 'tool_hang', 'can']
```

### 3. Train scalar scores given $\{(A_t, O_t)\}$

We give the examples of using **logpZO** and **RND**, which are the best performings ones. The other baselines are similar by switching to the corresponding folders

```
cd UQ_baselines/logpZO/ # Or change to /RND/, /CFM/, /NatPN/, /DER/ ...
# flow policy
uv run train.py --policy_type='flow' --type 'square'
# diffusion policy
uv run train.py --policy_type='diffusion' --type 'square'
cd ../..

# For other tasks, change 'square' to be among ['transport', 'tool_hang', 'can']
```

### 4. Run evaluation

```
cd UQ_test
# modify = False is ID
uv run eval_together.py --policy_type='flow' --task_name='square' --device=0 --modify=false --num=2000
uv run eval_together.py --policy_type='diffusion' --task_name='square' --device=0 --modify=false --num=2000

# modify = True is OOD
uv run eval_together.py --policy_type='flow' --task_name='square' --device=0 --modify=true --num=2000
uv run eval_together.py --policy_type='diffusion' --task_name='square' --device=0 --modify=true --num=2000
cd ..

# For other tasks, change 'square' to be among ['transport', 'tool_hang', 'can']
```

### 5. CP band + visualization

```
cd UQ_test
# flow
uv run plot_with_CP_band.py # Generate CP band and make decision
uv run barplot.py # Generate barplots

# diffusion
uv run plot_with_CP_band.py --diffusion_policy # Generate CP band and make decision
uv run barplot.py --diffusion_policy # Generate barplots
```
