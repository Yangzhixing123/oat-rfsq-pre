from typing import List, Optional

import numpy as np

from oat.env.metaworld.env import DEFAULT_CAMERA_NAMES, MetaworldEnv


MT_TASKS = {
    "mt4": [
        "box-close",
        "coffee-pull",
        "disassemble",
        "stick-pull",
    ],
    "mt10": [
        "hammer",
        "peg-insert-side",
        "push-wall",
        "box-close",
        "coffee-pull",
        "coffee-push",
        "shelf-place",
        "disassemble",
        "stick-pull",
        "stick-push",
    ],
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    return [task_name]


def get_metaworld_env(
    task_name: str,
    device: str = "cuda:0",
    image_size: int = 128,
    seed: Optional[int] = None,
    camera_names: List[str] = DEFAULT_CAMERA_NAMES,
    oracle: bool = False,
    video_camera: str = "corner2",
    max_episode_steps: int = 200,
) -> List[MetaworldEnv]:
    return [
        MetaworldEnv(
            task_name=subtask,
            device=device,
            image_size=image_size,
            seed=seed,
            camera_names=camera_names,
            oracle=oracle,
            video_camera=video_camera,
            max_episode_steps=max_episode_steps,
        )
        for subtask in get_subtasks(task_name)
    ]


def take_a_glance(
    task_names: List[str], camera_name: Optional[str]
) -> List[np.ndarray]:
    observations = []
    for task_name in task_names:
        cameras = [camera_name] if camera_name is not None else ["corner2"]
        env = get_metaworld_env(
            task_name=task_name,
            image_size=512,
            camera_names=cameras,
        )[0]
        if camera_name is None:
            observations.append(env.render())
        else:
            obs, _ = env.reset()
            observations.append(obs[f"{camera_name}_rgb"])
        env.close()
    if camera_name in {"topview", "behindGripper"}:
        observations = [np.flipud(obs) for obs in observations]
    return observations
