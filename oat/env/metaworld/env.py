from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium
import numpy as np


DEFAULT_CAMERA_NAMES = [
    "topview",
    "corner",
    "corner2",
    "corner3",
    "behindGripper",
    "gripperPOV",
]


def _get_metaworld_registry():
    try:
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    except ImportError as exc:
        raise ImportError(
            "MetaWorld is required for simulation rollouts. Install the "
            "MetaWorld dependencies described by sim-env before setting "
            "task.policy.lazy_eval=false."
        ) from exc
    return ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaworldEnv(gymnasium.Env):
    """Gymnasium adapter for the goal-observable MetaWorld v2 tasks."""

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }
    render_mode = "rgb_array"

    def __init__(
        self,
        task_name: str,
        device: str = "cuda:0",
        image_size: int = 128,
        seed: Optional[int] = None,
        camera_names: List[str] = DEFAULT_CAMERA_NAMES,
        oracle: bool = False,
        video_camera: str = "corner2",
        video_resolution: int = 512,
        max_episode_steps: int = 200,
        enable_render: bool = True,
    ):
        super().__init__()

        registry = _get_metaworld_registry()
        env_key = f"{task_name}-v2-goal-observable"
        if env_key not in registry:
            raise ValueError(f"Unknown MetaWorld task: {task_name}")

        env_cls = registry[env_key]
        try:
            env = env_cls(seed=seed)
        except TypeError:
            env = env_cls()
        if seed is not None:
            self._seed_env(env, seed)

        env._freeze_rand_vec = not oracle
        self._reset_env(env)
        env_init_state = env.get_env_state()

        # Match the camera setup used to generate the MetaWorld datasets.
        env.sim.model.vis.map.znear = 0.1
        env.sim.model.vis.map.zfar = 1.5
        corner2_id = env.sim.model.camera_name2id("corner2")
        env.sim.model.cam_pos0[corner2_id] = [0.6, 0.295, 0.8]
        env.sim.model.cam_pos[corner2_id] = [0.6, 0.295, 0.8]

        self.env = env
        self.env_init_state = env_init_state
        self.camera_names = list(camera_names)
        self.oracle = oracle
        self.image_size = image_size
        self.video_resolution = video_resolution
        self.task_name = task_name
        self.video_camera = video_camera
        self.max_episode_steps = max_episode_steps
        self.enable_render = enable_render
        self.gpu_id = self._device_id(device)
        self.cur_step = 0
        self.success_steps = 0
        self.done = False

        observation_space = gymnasium.spaces.Dict({})
        for camera_name in self.camera_names:
            observation_space.spaces[f"{camera_name}_rgb"] = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(image_size, image_size, 3),
                dtype=np.uint8,
            )
        observation_space.spaces["agent_pos"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._get_agent_pos().shape,
            dtype=np.float32,
        )
        if oracle:
            observation_space.spaces["full_state"] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=env.observation_space.shape,
                dtype=np.float32,
            )
        self.observation_space = observation_space

        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(
            low=action_low,
            high=action_high,
            shape=action_low.shape,
            dtype=np.float32,
        )

    @staticmethod
    def _device_id(device: str) -> int:
        if ":" not in device:
            return 0
        return int(device.rsplit(":", maxsplit=1)[-1])

    @staticmethod
    def _seed_env(env, seed: int) -> None:
        if hasattr(env, "seed"):
            env.seed(seed)
        else:
            env.reset(seed=seed)

    @staticmethod
    def _reset_env(env):
        result = env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result

    def _get_agent_pos(self) -> np.ndarray:
        return np.concatenate(
            [
                self.env.get_endeff_pos(),
                self.env._get_site_pos("leftEndEffector"),
                self.env._get_site_pos("rightEndEffector"),
            ]
        ).astype(np.float32)

    def _get_rgb(
        self,
        camera_names: Optional[List[str]] = None,
        image_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        camera_names = self.camera_names if camera_names is None else camera_names
        image_size = self.image_size if image_size is None else image_size
        if not self.enable_render:
            return {
                camera_name: np.zeros(
                    (image_size, image_size, 3), dtype=np.uint8
                )
                for camera_name in camera_names
            }
        return {
            camera_name: self.env.sim.render(
                width=image_size,
                height=image_size,
                camera_name=camera_name,
                depth=False,
                device_id=self.gpu_id,
            ).astype(np.uint8)
            for camera_name in camera_names
        }

    def _extract_obs(self, full_state=None) -> Dict[str, np.ndarray]:
        obs = {"agent_pos": self._get_agent_pos()}
        for camera_name, rgb in self._get_rgb().items():
            obs[f"{camera_name}_rgb"] = rgb
        if self.oracle:
            if full_state is None:
                full_state = self.env._get_obs()
            obs["full_state"] = np.asarray(full_state, dtype=np.float32)
        return obs

    def step(self, action: np.ndarray):
        result = self.env.step(action)
        if len(result) == 5:
            full_state, _, terminated, truncated, info = result
            underlying_done = bool(terminated or truncated)
        else:
            full_state, _, underlying_done, info = result

        self.cur_step += 1
        success = bool(info.get("success", False))
        if success:
            self.success_steps += 1
        self.done = bool(
            self.done
            or underlying_done
            or self.cur_step >= self.max_episode_steps
            or self.success_steps >= 10
        )
        info = dict(info)
        info["success"] = success
        return self._extract_obs(full_state), float(success), self.done, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options=None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self._reset_env(self.env)
        if hasattr(self.env, "reset_model"):
            self.env.reset_model()
        full_state = self._reset_env(self.env)
        self.env.set_env_state(self.env_init_state)
        if hasattr(self.env.sim, "forward"):
            self.env.sim.forward()

        self.cur_step = 0
        self.success_steps = 0
        self.done = False
        return self._extract_obs(full_state), {}

    def seed(self, seed=None):
        if seed is None:
            return
        self._seed_env(self.env, seed)
        self.env_init_state = self._env_init_state_for_seed(seed)

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        return self._get_rgb(
            [self.video_camera], image_size=self.video_resolution
        )[self.video_camera]

    def close(self):
        self.env.close()

    def _env_init_state_for_seed(self, seed: int):
        registry = _get_metaworld_registry()
        env_cls = registry[f"{self.task_name}-v2-goal-observable"]
        try:
            env = env_cls(seed=seed)
        except TypeError:
            env = env_cls()
            self._seed_env(env, seed)
        self._reset_env(env)
        init_state = env.get_env_state()
        env.close()
        return init_state
