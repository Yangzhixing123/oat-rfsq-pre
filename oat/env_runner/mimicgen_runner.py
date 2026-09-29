"""Balanced MimicGen evaluation with optional parallel simulation workers."""
from collections import deque
from pathlib import Path
import time
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from tqdm import tqdm
import numpy as np
import torch
from oat.env_runner.base_runner import BaseRunner
from oat.env.mimicgen.factory import get_subtasks
from oat.env.mimicgen.env import MimicGenEnv


class _RolloutWorker:
    """Own one simulator and its episode history inside a spawned process."""

    def __init__(self, env_factory, env_kwargs, n_obs_steps):
        self.env_factory = env_factory
        self.env_kwargs = env_kwargs
        self.n_obs_steps = n_obs_steps
        self.env = None
        self.task = None
        self.video = None

    def observation(self):
        return {key: np.stack([obs[key] for obs in self.history])
                for key in self.history[0]}

    def reset(self, task, seed, video_path):
        self.close_video()
        if self.task != task:
            self.close()
            self.env = self.env_factory(task_name=task, **self.env_kwargs)
            self.task = task
        obs, _ = self.env.reset(seed=seed)
        self.history = deque([obs] * self.n_obs_steps, maxlen=self.n_obs_steps)
        self.steps, self.success = 0, False
        if video_path is not None:
            import imageio.v2 as imageio
            self.video = imageio.get_writer(video_path, fps=20, codec='libx264')
            self.video.append_data(self.env.render())
        return self.observation()

    def step(self, actions):
        done = False
        for action in actions:
            obs, _, terminated, truncated, info = self.env.step(action)
            self.history.append(obs)
            self.steps += 1
            self.success |= bool(info['success'])
            done = terminated or truncated
            if self.video is not None:
                self.video.append_data(self.env.render())
            if done:
                self.close_video()
                break
        return self.observation(), done, self.success, self.steps

    def close_video(self):
        if self.video is not None:
            self.video.close()
            self.video = None

    def close(self):
        self.close_video()
        if self.env is not None:
            self.env.close()
            self.env = None
            self.task = None


def _init_worker(env_factory, env_kwargs, n_obs_steps):
    global _worker
    # Each worker simulates one environment; avoid CPU thread oversubscription.
    torch.set_num_threads(1)
    _worker = _RolloutWorker(env_factory, env_kwargs, n_obs_steps)


def _worker_call(method, *args):
    return getattr(_worker, method)(*args)


class MimicGenRunner(BaseRunner):
    def __init__(self, output_dir, task_name='mt4', n_test=40, n_test_vis=4,
                 test_start_seed=1000, n_obs_steps=2, n_action_steps=16,
                 n_parallel_envs=1, exclude_tasks=None, **env_kwargs):
        super().__init__(output_dir)
        self.tasks = get_subtasks(task_name)
        if n_test < len(self.tasks) or n_test % len(self.tasks):
            raise ValueError('n_test must be a positive multiple of the task count')
        if not 0 <= n_test_vis <= n_test or min(n_obs_steps,n_action_steps) < 1:
            raise ValueError('Invalid video count or history/action horizon')
        if not isinstance(n_parallel_envs, int) or n_parallel_envs < 1:
            raise ValueError('n_parallel_envs must be a positive integer')
        # Filter the original schedule, preserving task IDs, seeds and videos.
        excluded = set(exclude_tasks or [])
        if excluded - set(self.tasks):
            raise ValueError(f'Unknown excluded tasks: {sorted(excluded - set(self.tasks))}')
        self.episodes = [(task, test_start_seed + trial*len(self.tasks) + task_index,
                          trial*len(self.tasks) + task_index)
                         for task_index, task in enumerate(self.tasks)
                         for trial in range(n_test//len(self.tasks))
                         if task not in excluded]
        self.tasks = [task for task in self.tasks if task not in excluded]
        if not self.tasks:
            raise ValueError('Cannot exclude all evaluation tasks')
        self.n_parallel_envs = min(n_parallel_envs, len(self.episodes))
        self.n_test, self.n_test_vis = len(self.episodes), n_test_vis
        self.seed = test_start_seed
        self.n_obs_steps, self.n_action_steps = n_obs_steps,n_action_steps
        self.env_kwargs = env_kwargs
        self.env = None

    @torch.inference_mode()
    def run(self, policy, **kwargs):
        if self.n_parallel_envs > 1:
            return self._run_parallel(policy, **kwargs)
        directory = Path(self.output_dir)/'media'/('mimicgen_'+str(time.time_ns()))
        directory.mkdir(parents=True,exist_ok=False)
        rows, logs = [], {}
        was_training = policy.training
        policy.eval()
        try:
            # Group tasks to avoid rebuilding the simulator for every episode.
            for task in self.tasks:
                self.env = MimicGenEnv(task_name=task, **self.env_kwargs)
                try:
                    task_episodes = [episode for episode in self.episodes if episode[0] == task]
                    for _, seed, index in tqdm(task_episodes,
                                               desc=f'MimicGen {task}', unit='episode'):
                        obs,_ = self.env.reset(seed=seed)
                        history = deque([obs]*self.n_obs_steps,maxlen=self.n_obs_steps)
                        policy.reset()
                        video = None
                        video_path = directory/f'{task}_{seed}.mp4'
                        if index < self.n_test_vis:
                            import imageio.v2 as imageio
                            video = imageio.get_writer(str(video_path),fps=20,codec='libx264')
                        success, done, steps = False,False,0
                        try:
                            if video is not None: video.append_data(self.env.render())
                            while not done:
                                batch = {key:torch.from_numpy(np.stack([o[key] for o in history])[None]).to(policy.device)
                                         for key in policy.get_observation_ports()}
                                prediction = policy.predict_action(batch,**kwargs)['action']
                                actions = prediction.detach().cpu().numpy()
                                if actions.ndim!=3 or actions.shape[0]!=1 or actions.shape[2]!=7 or actions.shape[1]<1 or not np.isfinite(actions).all():
                                    raise ValueError('Policy must return finite action [1,N,7]')
                                for action in actions[0,:self.n_action_steps]:
                                    obs,_,terminated,truncated,info = self.env.step(action)
                                    history.append(obs)
                                    steps += 1
                                    success |= bool(info['success'])
                                    done = terminated or truncated
                                    if video is not None: video.append_data(self.env.render())
                                    if done: break
                        finally:
                            if video is not None: video.close()
                        rows.append(dict(task=task,seed=seed,success=success,steps=steps))
                        if video is not None:
                            import wandb
                            logs[f'{task}/video_{seed}'] = wandb.Video(str(video_path),format='mp4')
                finally:
                    self.close()
        finally:
            policy.train(was_training)
        for task in self.tasks:
            logs[f'{task}/mean_success_rate'] = float(np.mean([r['success'] for r in rows if r['task']==task]))
        logs['mean_success_rate'] = float(np.mean([r['success'] for r in rows]))
        (directory/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
        return logs

    @torch.inference_mode()
    def _run_parallel(self, policy, **kwargs):
        directory = Path(self.output_dir)/'media'/('mimicgen_'+str(time.time_ns()))
        directory.mkdir(parents=True, exist_ok=False)
        # Keep the original task-major order and task/seed/video assignment.
        episodes = self.episodes
        rows, logs = [], {}
        was_training = policy.training
        policy.eval()
        try:
            with ExitStack() as stack:
                workers = [stack.enter_context(ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                    initializer=_init_worker,
                    initargs=(MimicGenEnv, self.env_kwargs, self.n_obs_steps),
                )) for _ in range(self.n_parallel_envs)]
                try:
                    with tqdm(total=self.n_test, desc='MimicGen evaluation', unit='episode') as progress:
                        for start in range(0, self.n_test, self.n_parallel_envs):
                            group = episodes[start:start+self.n_parallel_envs]
                            paths = [directory/f'{task}_{seed}.mp4' if index < self.n_test_vis else None
                                     for task, seed, index in group]
                            futures = [workers[i].submit(_worker_call, 'reset', task, seed,
                                                         str(paths[i]) if paths[i] else None)
                                       for i, (task, seed, _) in enumerate(group)]
                            observations = [future.result() for future in futures]
                            active = list(range(len(group)))
                            results = {}
                            policy.reset()
                            while active:
                                batch = {key: torch.from_numpy(np.stack(
                                    [observations[i][key] for i in active])).to(policy.device)
                                    for key in policy.get_observation_ports()}
                                actions = policy.predict_action(batch, **kwargs)['action'].detach().cpu().numpy()
                                if (actions.ndim != 3 or actions.shape[0] != len(active)
                                        or actions.shape[2] != 7 or actions.shape[1] < 1
                                        or not np.isfinite(actions).all()):
                                    raise ValueError('Policy must return finite action [B,N,7]')
                                futures = [workers[i].submit(_worker_call, 'step',
                                                             actions[j, :self.n_action_steps])
                                           for j, i in enumerate(active)]
                                pending = []
                                for i, future in zip(active, futures):
                                    observations[i], done, success, steps = future.result()
                                    if done:
                                        task, seed, _ = group[i]
                                        results[i] = dict(task=task, seed=seed, success=success, steps=steps)
                                        progress.update(1)
                                    else:
                                        pending.append(i)
                                active = pending
                            for i, (task, seed, _) in enumerate(group):
                                rows.append(results[i])
                                if paths[i] is not None:
                                    import wandb
                                    logs[f'{task}/video_{seed}'] = wandb.Video(str(paths[i]), format='mp4')
                finally:
                    # Close video writers and EGL contexts before joining workers.
                    futures = []
                    for worker in workers:
                        try:
                            futures.append(worker.submit(_worker_call, 'close'))
                        except RuntimeError:
                            pass  # A broken worker pool has already exited.
                    for future in futures:
                        try:
                            future.result()
                        except Exception:
                            pass  # Preserve the original rollout error.
        finally:
            policy.train(was_training)
        for task in self.tasks:
            logs[f'{task}/mean_success_rate'] = float(np.mean([r['success'] for r in rows if r['task'] == task]))
        logs['mean_success_rate'] = float(np.mean([r['success'] for r in rows]))
        (directory/'episodes.json').write_text(json.dumps(rows, indent=2)+'\n')
        return logs

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
