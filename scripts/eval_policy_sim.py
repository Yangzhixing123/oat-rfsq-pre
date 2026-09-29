"""
Usage:
python experiments/eval_policy_sim.py --checkpoint path/to/ckpt -o path/to/output_dir
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import random
import click
import hydra
import torch
import wandb
import json
import numpy as np
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from typing import List, Optional

@click.command()
@click.option('-c', '--checkpoint', required=True, help="either a .ckpt file or a directory containing .ckpt files")
@click.option('-o', '--output_dir', required=True, help="output directory for eval info dump")
@click.option('-n', '--num_exp', default=1, help="num experiments to run")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('--exclude-tasks', default=None, help='Comma-separated MimicGen tasks to skip.')
@click.option('--temperature', default=None, type=float, help="temperature for policy inference")
@click.option('--topk', default=None, type=int, help="topk for policy inference")
@click.option('--use_k_tokens', default=None, type=int, help="number of tokens to use for policy inference")
@click.option(
    '--seed',
    default=None,
    type=int,
    help=(
        "optional RNG seed reused for every experiment. If omitted, do not "
        "explicitly seed policy sampling (the original evaluation behavior)"
    ),
)
@click.option(
    '--seeds',
    default=None,
    help=(
        "comma-separated RNG seed for each experiment; the count must equal "
        "--num_exp and this option cannot be combined with --seed"
    ),
)
def eval_policy_sim(
    checkpoint: str,
    output_dir: str,
    num_exp: int = 1,
    device: str = 'cuda:0',
    # policy inference args
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    use_k_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    seeds: Optional[str] = None,
    exclude_tasks: Optional[str] = None,
):
    if num_exp < 1:
        raise click.BadParameter("must be at least 1", param_hint="--num_exp")
    if seed is not None and seeds is not None:
        raise click.UsageError("Use either --seed or --seeds, not both.")

    if seeds is None:
        experiment_seeds = [seed] * num_exp
    else:
        try:
            experiment_seeds = [
                int(item.strip()) for item in seeds.split(",") if item.strip()
            ]
        except ValueError as exc:
            raise click.BadParameter(
                "must be a comma-separated list of integers", param_hint="--seeds"
            ) from exc
        if len(experiment_seeds) != num_exp:
            raise click.BadParameter(
                f"contains {len(experiment_seeds)} values, but --num_exp is {num_exp}",
                param_hint="--seeds",
            )

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        os.system(f"rm -rf {output_dir}")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # grab all checkpoints
    ckpts: List[str]    # file paths to checkpoints to evaluate
    if os.path.isdir(checkpoint):
        ckpts = [
            os.path.join(checkpoint, f) 
            for f in os.listdir(checkpoint) 
            if f.endswith('.ckpt') and f != 'latest.ckpt'
        ]
    else:
        ckpts = [checkpoint,]

    base_output_dir = output_dir
    for ckpt in ckpts:
        # format output dir
        if len(ckpts) > 1:
            ckpt_name = os.path.basename(ckpt).replace('.ckpt', '')
            output_dir = os.path.join(base_output_dir, ckpt_name)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir
        
        # load checkpoint
        policy, cfg = BasePolicy.from_checkpoint(ckpt, return_configuration=True)
        
        device = torch.device(device)
        policy.to(device)
        policy.eval()
        
        # Use either one shared RNG seed or an explicit seed for each repetition
        # so different K values can be compared under the same seed schedule.
        print(f"Running evaluation on {ckpt}")

        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if topk is not None:
            kwargs['topk'] = topk
        if use_k_tokens is not None:
            kwargs['use_k_tokens'] = use_k_tokens
        runner_cfg = cfg.task.policy.env_runner
        excluded = [t.strip() for t in (exclude_tasks or '').split(',') if t.strip()]
        runner_overrides = {}
        if excluded:
            if runner_cfg._target_ != 'oat.env_runner.mimicgen_runner.MimicGenRunner':
                raise click.UsageError('--exclude-tasks currently supports MimicGen only')
            runner_overrides['exclude_tasks'] = excluded
        base_env_seed = int(runner_cfg.get('test_start_seed', 1000))
        n_test = int(runner_cfg.n_test)

        # Store all runs for computing statistics and keep videos from all runs.
        all_runs = []
        runner_log = None
        rng_seeds = []
        env_seed_ranges = []

        for i in range(num_exp):
            this_seed = experiment_seeds[i]
            this_env_start_seed = base_env_seed
            this_env_end_seed = this_env_start_seed + n_test - 1
            rng_seeds.append(this_seed)
            env_seed_ranges.append(
                {'start': this_env_start_seed, 'end': this_env_end_seed}
            )

            if this_seed is not None:
                random.seed(this_seed)
                np.random.seed(this_seed)
                torch.manual_seed(this_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(this_seed)

            env_runner: BaseRunner = hydra.utils.instantiate(
                runner_cfg,
                output_dir=output_dir,
                test_start_seed=this_env_start_seed,
                **runner_overrides,
            )
            evaluation_episodes = None
            if hasattr(env_runner, 'episodes'):
                evaluation_episodes = [dict(task=task, seed=env_seed)
                                       for task, env_seed, _ in env_runner.episodes]
                print(f'Evaluating {env_runner.n_test} episodes across {env_runner.tasks}')
            try:
                this_log = env_runner.run(policy, **kwargs)
            finally:
                env_runner.close()

            print(
                f"Exp {i + 1}: rng_seed = "
                f"{this_seed if this_seed is not None else 'unfixed'}, "
                f"env_seeds = {this_env_start_seed}-{this_env_end_seed}, "
                f"success rate = {this_log['mean_success_rate']}"
            )
            all_runs.append({
                k: v
                for k, v in this_log.items()
                if not isinstance(v, wandb.sdk.data_types.video.Video)
            })

            if runner_log is None:
                runner_log = {}
                for key, value in this_log.items():
                    if isinstance(value, wandb.sdk.data_types.video.Video):
                        runner_log[key] = [value]
                    else:
                        runner_log[key] = value
            else:
                for key, value in this_log.items():
                    if isinstance(value, wandb.sdk.data_types.video.Video):
                        runner_log.setdefault(key, []).append(value)
                    else:
                        assert key in runner_log
                        runner_log[key] += value

        assert runner_log is not None
        
        # Compute mean and std for all numeric metrics
        numeric_keys = [k for k in all_runs[0].keys()]
        mean_log = {}
        std_log = {}
        
        for key in numeric_keys:
            values = [run[key] for run in all_runs]
            mean_log[key] = np.mean(values)
            if num_exp > 1:
                std_log[key] = np.std(values, ddof=1)  # sample std
        
        # dump log to json
        json_log = dict()
        json_log['checkpoint'] = ckpt
        json_log['num_exp'] = num_exp
        json_log['excluded_tasks'] = excluded
        json_log['evaluation_episodes'] = evaluation_episodes
        json_log['n_test_evaluated'] = len(evaluation_episodes) if evaluation_episodes is not None else n_test
        json_log['inference_kwargs'] = kwargs
        json_log['n_action_steps'] = int(policy.n_action_steps)
        json_log['rng_seeds'] = rng_seeds
        json_log['env_seed_ranges'] = env_seed_ranges
        
        # Add mean values
        for key, value in mean_log.items():
            json_log[f'{key}_mean'] = float(value)
        
        # Add standard deviation & error values if multiple experiments
        if num_exp > 1:
            for key, value in std_log.items():
                json_log[f'{key}_std'] = float(value)
                json_log[f'{key}_stderr'] = float(value / np.sqrt(num_exp))
        
        # Add video paths
        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    assert isinstance(video, wandb.sdk.data_types.video.Video)
                    json_log[f'{key}_{i}'] = video._path
        
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    eval_policy_sim()
