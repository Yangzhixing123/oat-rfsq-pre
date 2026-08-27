from typing import List, Optional

import gymnasium
import numpy as np
import robosuite
from robosuite.controllers import load_controller_config


def create_env(
    task_name: str,
    camera_names: List[str] = [
        "agentview",
        "robot0_eye_in_hand",
    ],
    camera_width: int = 128,
    camera_height: int = 128,
    enable_render: bool = True,
    seed: int = 42,
    robot: str = "Panda",
):
    # RoboMimic's released datasets use OSC_POSE with a 7-D action:
    # xyz delta, axis-angle delta, and gripper command.
    # robosuite 1.4 does not accept ``seed`` in environment constructors and
    # uses NumPy's global RNG while building and resetting the environment.
    np.random.seed(seed)
    controller_config = load_controller_config(default_controller="OSC_POSE")
    return robosuite.make(
        env_name=task_name,
        robots=robot,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_width,
        camera_heights=camera_height,
        has_renderer=False,
        has_offscreen_renderer=enable_render,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=enable_render,
        camera_depths=False,
    )


class RobomimicEnv(gymnasium.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        task_name: str,
        image_size: int = 128,
        seed: int = 42,
        camera_names: List[str] = [
            "agentview",
            "robot0_eye_in_hand",
        ],
        state_ports: List[str] = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        video_camera: str = "agentview",
        video_resolution: int = 512,
        max_episode_steps: int = 800,
        enable_render: bool = True,
    ):
        super().__init__()

        all_camera_names = list(dict.fromkeys([*camera_names, video_camera]))
        env = create_env(
            task_name=task_name,
            camera_names=all_camera_names,
            camera_width=image_size,
            camera_height=image_size,
            seed=seed,
            enable_render=enable_render,
            robot="Panda",
        )
        env.hard_reset = False

        self.env = env
        self.task_name = task_name
        self.state_ports = state_ports
        self.camera_names = camera_names
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.enable_render = enable_render
        self.done = False
        self.cur_step = 0
        self._seed = seed

        raw_obs = env._get_observations(force_update=True)
        observation_space = gymnasium.spaces.Dict({})
        for port in state_ports:
            observation_space.spaces[port] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=raw_obs[port].shape,
                dtype=np.float32,
            )
        for camera_name in camera_names:
            observation_space.spaces[f"{camera_name}_rgb"] = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(image_size, image_size, 3),
                dtype=np.uint8,
            )
        self.observation_space = observation_space

        action_low, action_high = env.action_spec
        self.action_space = gymnasium.spaces.Box(
            low=np.asarray(action_low, dtype=np.float32),
            high=np.asarray(action_high, dtype=np.float32),
            shape=action_low.shape,
            dtype=np.float32,
        )

    def _extract_obs(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env._get_observations(force_update=True)

        obs_dict = {
            port: raw_obs[port].astype(np.float32)
            for port in self.state_ports
        }
        for camera_name in self.camera_names:
            obs_dict[f"{camera_name}_rgb"] = np.flip(
                raw_obs[f"{camera_name}_image"], axis=0
            ).astype(np.uint8)
        return obs_dict

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        raw_obs = self.env.reset()
        self.done = False
        self.cur_step = 0
        return self._extract_obs(raw_obs), {}

    def step(self, action):
        raw_obs, _, robosuite_done, info = self.env.step(action)
        self.cur_step += 1
        success = bool(self.env._check_success())
        reward = float(success)
        self.done = bool(
            self.done
            or robosuite_done
            or success
            or self.cur_step >= self.max_episode_steps
        )
        return self._extract_obs(raw_obs), reward, self.done, False, info

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        return np.flip(
            self.env.sim.render(
                height=self.video_resolution,
                width=self.video_resolution,
                camera_name=self.video_camera,
            ),
            axis=0,
        ).astype(np.uint8)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
