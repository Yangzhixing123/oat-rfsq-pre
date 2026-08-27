import math
import pathlib
from typing import List, Optional

import dill
import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wandb_video

from oat.common.pytorch_util import dict_apply
from oat.env.robomimic.env import RobomimicEnv
from oat.env.robomimic.factory import get_subtasks
from oat.env_runner.base_runner import BaseRunner
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from oat.policy.base_policy import BasePolicy


def maybe_to_torch(x, device, dtype):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return x


class RobomimicRunner(BaseRunner):
    def __init__(
        self,
        output_dir,
        task_name: str,
        n_test: int = 22,
        n_test_vis: int = 6,
        test_start_seed: int = 1000,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 20,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_parallel_envs: Optional[int] = None,
        image_size: int = 128,
        camera_names: List[str] = [
            "agentview",
            "robot0_eye_in_hand",
        ],
        state_ports: List[str] = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        max_episode_steps: int = 800,
    ):
        super().__init__(output_dir)

        if n_parallel_envs is None:
            n_parallel_envs = n_test
        n_parallel_envs = min(n_parallel_envs, n_test)
        if n_parallel_envs <= 0:
            raise ValueError("n_parallel_envs must be positive")
        if n_test_vis > n_test:
            raise ValueError("n_test_vis must be less than or equal to n_test")

        subtask_names = get_subtasks(task_name)
        env_task_names = [
            subtask_names[i % len(subtask_names)] for i in range(n_test)
        ]

        env_seeds = []
        env_fns = []
        env_init_fn_dills = []
        for i in range(n_test):
            this_task_name = env_task_names[i]
            this_seed = test_start_seed + i
            env_seeds.append(this_seed)
            enable_video = i < n_test_vis

            if i < n_parallel_envs:
                def env_fn(task_name=this_task_name, seed=this_seed):
                    return self._make_wrapped_env(
                        task_name=task_name,
                        seed=seed,
                        image_size=image_size,
                        camera_names=camera_names,
                        state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                        fps=fps,
                        crf=crf,
                        n_obs_steps=n_obs_steps,
                        n_action_steps=n_action_steps,
                        enable_render=True,
                    )

                env_fns.append(env_fn)

            def init_fn(
                env,
                task_name=this_task_name,
                seed=this_seed,
                enable_video=enable_video,
            ):
                if env.env.env.task_name != task_name:
                    env.env.env.close()
                    env.env.env = RobomimicEnv(
                        task_name=task_name,
                        image_size=image_size,
                        seed=seed,
                        camera_names=camera_names,
                        state_ports=state_ports,
                        max_episode_steps=max_episode_steps,
                    )
                else:
                    env.env.env.seed(seed)

                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_video:
                    filename = pathlib.Path(output_dir).joinpath(
                        f"media/{task_name}",
                        wandb_video.util.generate_id() + ".mp4",
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)
                env.reset()

            env_init_fn_dills.append(dill.dumps(init_fn))

        if len(env_fns) != n_parallel_envs:
            raise RuntimeError("Failed to create the requested parallel environments")

        def dummy_env_fn():
            return self._make_wrapped_env(
                task_name=env_task_names[0],
                seed=test_start_seed,
                image_size=image_size,
                camera_names=camera_names,
                state_ports=state_ports,
                max_episode_steps=max_episode_steps,
                fps=fps,
                crf=crf,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                enable_render=False,
            )

        self.env = AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            dummy_env_fn=dummy_env_fn,
        )
        self.task_name = task_name
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_task_names = env_task_names
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    @staticmethod
    def _make_wrapped_env(
        task_name,
        seed,
        image_size,
        camera_names,
        state_ports,
        max_episode_steps,
        fps,
        crf,
        n_obs_steps,
        n_action_steps,
        enable_render,
    ):
        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicEnv(
                    task_name=task_name,
                    image_size=image_size,
                    seed=seed,
                    camera_names=camera_names,
                    state_ports=state_ports,
                    max_episode_steps=max_episode_steps,
                    enable_render=enable_render,
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec="h264",
                    input_pix_fmt="rgb24",
                    crf=crf,
                    thread_type="FRAME",
                    thread_count=1,
                ),
                file_path=None,
                steps_per_render=1,
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            reward_agg_method="max",
        )

    @torch.inference_mode()
    def run(self, policy: BasePolicy, **kwargs):
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        all_video_paths = [None] * n_inits
        all_success = np.zeros(n_inits, dtype=bool)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            global_slice = slice(start, end)
            n_active_envs = end - start
            local_slice = slice(0, n_active_envs)

            init_fns = list(self.env_init_fn_dills[global_slice])
            init_fns.extend(
                [self.env_init_fn_dills[0]] * (n_envs - len(init_fns))
            )
            self.env.call_each(
                "run_dill_function",
                args_list=[(init_fn,) for init_fn in init_fns],
            )

            obs, _ = self.env.reset()
            policy.reset()
            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=(
                    f"Eval {policy_name} in RoboMimic::{self.task_name} "
                    f"{chunk_idx + 1}/{n_chunks}"
                ),
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            chunk_done = False
            while not chunk_done and pbar.n < pbar.total:
                obs_dict = dict_apply(
                    obs,
                    lambda x: maybe_to_torch(x, device=device, dtype=dtype),
                )
                action = policy.predict_action(
                    {
                        port: obs_dict[port]
                        for port in policy.get_observation_ports()
                    },
                    **kwargs,
                )["action"].detach().cpu().numpy()

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN or Inf action")

                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_done = np.logical_or(terminated, truncated)
                active_done = np.logical_or(
                    episode_done[local_slice],
                    all_success[global_slice],
                )
                all_success[global_slice] = np.logical_or(
                    all_success[global_slice],
                    reward[local_slice] >= 1,
                )
                chunk_done = bool(np.all(active_done))
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[global_slice] = self.env.render()[local_slice]

        self.env.reset()

        log_data = {}
        for task_name in set(self.env_task_names):
            task_success = [
                all_success[i]
                for i in range(n_inits)
                if self.env_task_names[i] == task_name
            ]
            log_data[f"{task_name}/mean_success_rate"] = np.mean(task_success)

        for i, video_path in enumerate(all_video_paths):
            if video_path is not None:
                task_name = self.env_task_names[i]
                seed = self.env_seeds[i]
                log_data[f"{task_name}/video_{seed}"] = wandb.Video(
                    video_path, format="mp4"
                )

        log_data["mean_success_rate"] = np.mean(all_success)
        return log_data

    def close(self):
        self.env.close()
