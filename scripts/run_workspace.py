"""
Usage:
python scripts/run_workspace.py --config-name=train_tokenizer
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    # Prefer this checkout over another editable ``oat`` installation.
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

    # mujoco-py checks LD_LIBRARY_PATH during import, and the dynamic loader
    # needs those paths at process startup. Bootstrap them once for MetaWorld
    # jobs so callers do not need shell-specific exports.
    if any(
        arg.startswith("task/policy=metaworld/")
        for arg in sys.argv[1:]
    ):
        env = os.environ.copy()
        ld_paths = [
            str(pathlib.Path.home().joinpath(".mujoco", "mujoco210", "bin")),
            str(pathlib.Path(sys.prefix).joinpath("lib")),
            "/usr/lib/nvidia",
            "/usr/local/cuda/lib64",
        ]
        current_ld_paths = [
            path for path in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if path
        ]
        required_ld_paths = [
            path for path in ld_paths
            if pathlib.Path(path).is_dir()
        ]
        missing_ld_paths = [
            path for path in required_ld_paths
            if path not in current_ld_paths
        ]
        python_bin = str(pathlib.Path(sys.executable).parent)
        current_path = [
            path for path in env.get("PATH", "").split(os.pathsep)
            if path
        ]
        needs_python_bin = python_bin not in current_path

        if missing_ld_paths or needs_python_bin:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [*missing_ld_paths, *current_ld_paths]
            )
            if needs_python_bin:
                env["PATH"] = os.pathsep.join([python_bin, *current_path])
            os.execve(
                sys.executable,
                [sys.executable, *sys.argv],
                env,
            )

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from oat.workspace.base_workspace import BaseWorkspace
from oat.common.hydra_util import register_new_resolvers

register_new_resolvers()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'oat','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
