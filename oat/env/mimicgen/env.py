"""Gymnasium adapter matching the stored 84 -> 128 RGB dataset pipeline."""
import warnings
import cv2
import gymnasium as gym
import numpy as np
from oat.env.mimicgen.factory import get_task

CAMERAS = ('agentview', 'robot0_eye_in_hand')
STATES = {'robot0_eef_pos': 3, 'robot0_eef_quat': 4, 'robot0_gripper_qpos': 2}


def encode_observation(raw, task_uid, image_size=128):
    obs = {}
    for key, width in STATES.items():
        value = np.asarray(raw[key], dtype=np.float32)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid observation {key}: {value.shape}')
        obs[key] = value
    if not np.isclose(np.linalg.norm(obs['robot0_eef_quat']), 1, atol=1e-3):
        raise ValueError('Non-unit xyzw quaternion')
    for camera in CAMERAS:
        # Raw MuJoCo images are bottom-up; robomimic stored images are top-down.
        frame = np.asarray(raw[camera + '_image'])
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f'Invalid RGB camera {camera}')
        obs[camera + '_rgb'] = cv2.resize(frame[::-1], (image_size, image_size), interpolation=cv2.INTER_AREA)
    obs['task_uid'] = np.array([task_uid], dtype=np.int64)
    return obs


class MimicGenEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, task_name, seed=1000, image_size=128, source_image_size=84,
                 max_episode_steps=800, strict_version=True):
        import robosuite
        import mimicgen  # noqa: F401 -- registers task classes
        task = get_task(task_name)
        meta = task['env_args']
        if robosuite.__version__ != meta['env_version']:
            message = f"Dataset robosuite={meta['env_version']}; installed={robosuite.__version__}. Use a matching isolated environment."
            if strict_version:
                raise RuntimeError(message)
            warnings.warn(message)
        if max_episode_steps < 1 or image_size < 1 or source_image_size < 1:
            raise ValueError('Image sizes and episode limit must be positive')
        kwargs = meta['env_kwargs']
        kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                      camera_names=list(CAMERAS), camera_heights=source_image_size,
                      camera_widths=source_image_size, camera_depths=False, ignore_done=True,
                      reward_shaping=False)
        self.task_name = task_name
        self.task_uid = task['task_uid']
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self._seed = seed
        np.random.seed(seed)
        self.env = robosuite.make(env_name=meta['env_name'], **kwargs)
        low, high = self.env.action_spec
        if np.shape(low) != (7,) or np.shape(high) != (7,):
            self.env.close()
            raise ValueError('Expected seven-dimensional OSC_POSE action')
        self.action_space = gym.spaces.Box(np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))
        spaces = {key: gym.spaces.Box(-np.inf, np.inf, (width,), np.float32) for key,width in STATES.items()}
        spaces.update({camera+'_rgb':gym.spaces.Box(0,255,(image_size,image_size,3),np.uint8) for camera in CAMERAS})
        spaces['task_uid'] = gym.spaces.Box(0,3,(1,),np.int64)
        self.observation_space = gym.spaces.Dict(spaces)
        self.steps = 0
        self.success = False
        self.last_obs = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        np.random.seed(self._seed)
        raw = self.env.reset()
        self.steps, self.success = 0, False
        self.last_obs = encode_observation(raw,self.task_uid,self.image_size)
        return self.last_obs, {'success':False,'task_uid':self.task_uid}

    def step(self, action):
        action = np.asarray(action,dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Action must be seven finite source-controller values')
        raw, _, done, info = self.env.step(action)
        self.steps += 1
        self.success = self.success or bool(self.env._check_success())
        self.last_obs = encode_observation(raw,self.task_uid,self.image_size)
        terminated = self.success or bool(done)
        truncated = self.steps >= self.max_episode_steps and not terminated
        return self.last_obs, float(self.success), terminated, truncated, {**info,'success':self.success,'task_uid':self.task_uid}

    def render(self):
        if self.last_obs is None:
            raise RuntimeError('Reset before rendering')
        return self.last_obs['agentview_rgb'].copy()

    def close(self):
        self.env.close()
