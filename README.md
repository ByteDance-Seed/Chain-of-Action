<p align="center">
  <img src="assets/logo.png" alt="Project Logo" width="640"/>
</p>

<p align="center">
  <a href="https://chain-of-action.github.io/"><img src="https://img.shields.io/badge/Website-Visit-0A66C2?logo=safari&logoColor=white" alt="Website" /></a> <a href="https://arxiv.org/pdf/2506.09990"><img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoColor=red" alt="Paper on arXiv" /></a> <a href="https://huggingface.co/Solomonz/Chain-of-Action"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&logoColor=yellow" alt="HuggingFace Model" /></a> <a href="https://huggingface.co/datasets/Solomonz/Chain-of-Action"><img src="https://img.shields.io/badge/HuggingFace-Data-blue?logo=huggingface&logoColor=blue" alt="HuggingFace Dataset" /></a>
</p>

## Project Overview
This repository offers the official implementation of Chain-of-Action on RLBench. The framework aims to integrate representative visuomotor policies, including the [ACT](https://github.com/tonyzhaozh/act) and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (To-do soon), with standardized training and evaluation protocols to facilitate fair comparison and reproducibility for the community.

## Quick start
### Set up environment
  
   ```bash
   conda create -n coa python=3.9 -y
   conda activate coa
   bash scripts/init.sh
   source ~/.bashrc
   ```

install dependencies and RLBench enviroment, see [init.sh](scripts/init.sh) for details

### One-click Evaluation

The script will automatically download the required pretrained snapshot and the necessary evaluation dataset for the specified task.  

```bash
bash scripts/eval.sh task=push_button
```

## Train & Eval
### Download RLBench datasets

Execute the command to download all data.

```bash
python scripts/download_dataset.py
```

#### Detailed usage

```bash
python scripts/download_dataset.py --task reach_target --train-episodes 100 --eval-episodes 25
```

- `--task`: Specify the task name to download (e.g., reach_target, stack_wine). Only one task will be downloaded.
- `--train-episodes`: Number of training episodes per task (default: 100, total 100).
- `--eval-episodes`: Number of evaluation episodes per task (default: 25, total 50).
- To download the recommended 10-task subset, add the `--subset` flag. To download all tasks, do not specify `--task` or `--subset`.

### Evaluation



  ```bash
  python scripts/eval.py task=task_name snapshot=path_to_snapshot
   ```

### Training

  ```bash
  python scripts/train.py task=task_name 
   ```
   
**For detailed parameter settings, please refer to [launch.yaml](src/cfgs/launch.yaml).**

Key parameters include:

- `num_train_steps`: total training steps (default: 20000)
- `batch_size`: training batch size (default: 128)
- `task`: task name (must be specified)
- `demos`: number of demonstrations per task (default: 100)
- `eval_every_steps`: evaluation interval in steps (default: 10000)
- `vis_every_steps`: visualization interval in steps (default: 2000)
- `save_every_steps`: model checkpoint interval in steps (default: 10000)
- `num_eval_episodes`: number of episodes per evaluation (default: 25)

You can customize these parameters by editing `src/cfgs/launch.yaml` directly, or override them via command line arguments (e.g., `python scripts/train.py task=push_button batch_size=64`).

## Directory Structure
- `scripts/`：Training, evaluation, data/snapshot downloading scripts

- `src/`: Main source code directory, including the following subfolders and files:
  - `cfgs/`: Configuration files
  - `dataset/`: Dataset loading and preprocessing 
  - `envs/`: Simulation environment 
  - `methods/`: Algorithm implementations
    - `coa/`: Chain-of-Action
    - `act/`: ACT (To-do)
    - `dp/`: Diffusion policy (To-do)
    - `base.py`, `utils.py`: Common base classes and utilities
  - `utils.py`, `logger.py`, `video.py`, : General utilities and main control scripts
  - `workspace.py`: training workflow

- `exp_local/`: Local experiment results.
  - `checkpoints/`: Model weights
  - `eval_videos/`: Evaluation videos
  - `train.log`: Training log
  - `.hydra/`: Configuration snapshots and Hydra management files
- `README.md`：Project documentation



## Citation
```bibtex
@inproceedings{zhang2025chainofaction,
  author={Zhang, Wenbo and Hu, Tianrun and Qiao, Yanyuan and Zhang, Hanbo and Qin, Yuchu and Li, Yang and Liu, Jiajun and Kong, Tao and Liu, Lingqiao and Ma, Xiao},
  title={Chain-of-Action: Trajectory Autoregressive Modeling for Robotic Manipulation},
  journal= {arxiv},
  year={2025},
}
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details. 
