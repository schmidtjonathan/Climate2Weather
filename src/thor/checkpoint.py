import os
import re

from lightning.fabric import Fabric


# Adapted from:
# https://github.com/NVlabs/edm2/blob/4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/torch_utils/distributed.py#L85
class CheckpointIO:
    def __init__(self, **kwargs):
        self.state_objs = kwargs

    def save(self, fabric: Fabric, pt_path, verbose=True):
        if verbose:
            fabric.print(f"Saving {pt_path} ... ", end="", flush=True)

        data = dict()
        for name, obj in self.state_objs.items():
            if obj is None:
                data[name] = None
            elif isinstance(obj, dict):
                data[name] = obj
            elif hasattr(obj, "state_dict"):
                data[name] = obj
            elif hasattr(obj, "__getstate__"):
                data[name] = obj
            elif hasattr(obj, "__dict__"):
                data[name] = obj.__dict__
            else:
                raise ValueError(f"Invalid state object of type {type(obj).__name__}")

        fabric.save(pt_path, data)

        if verbose:
            fabric.print("done.")

    def load(self, fabric: Fabric, pt_path, verbose=True):
        if verbose:
            fabric.print(f"Loading {pt_path} ... ", end="", flush=True)

        data = fabric.load(pt_path)
        for name, obj in self.state_objs.items():
            if obj is None:
                pass
            elif isinstance(obj, dict):
                obj.clear()
                obj.update(data[name])
            elif hasattr(obj, "load_state_dict"):
                obj.load_state_dict(data[name])
            elif hasattr(obj, "__setstate__"):
                obj.__setstate__(data[name])
            elif hasattr(obj, "__dict__"):
                obj.__dict__.clear()
                obj.__dict__.update(data[name])
            else:
                raise ValueError(f"Invalid state object of type {type(obj).__name__}")

        if verbose:
            fabric.print("done.")

    def load_latest(
        self,
        fabric: Fabric,
        run_dir,
        pattern=r"training-state-(\d+).ckpt",
        verbose=True,
    ):
        fnames = [
            entry.name
            for entry in os.scandir(run_dir)
            if entry.is_file() and re.fullmatch(pattern, entry.name)
        ]
        if len(fnames) == 0:
            return None
        pt_path = os.path.join(
            run_dir, max(fnames, key=lambda x: float(re.fullmatch(pattern, x).group(1)))
        )
        self.load(fabric, pt_path, verbose=verbose)
        return pt_path
