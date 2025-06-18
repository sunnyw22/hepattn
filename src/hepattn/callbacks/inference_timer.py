from pathlib import Path

import numpy as np
import torch
from lightning import Callback

from hepattn.utils.cuda_timer import cuda_timer


class InferenceTimer(Callback):
    def __init__(self):
        super().__init__()
        self.times = []
        self.dims = []
        self.n_warm_start = 10
        self._tmp_dims = None

    def on_test_start(self, trainer, pl_module):
        assert trainer.global_rank == 0, "InferenceTimer should only be used with a single process."
        model = pl_module
        if hasattr(model, "model"):
            model = model.model
        self.old_forward = model.forward

        def new_forward(*args, **kwargs):
            self._tmp_dims = sum(v.shape[1] for v in args[0].values())
            with cuda_timer(self.times):
                return self.old_forward(*args, **kwargs)

        model.forward = new_forward

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._tmp_dims is not None:
            self.dims.append(self._tmp_dims)
            self._tmp_dims = None

    def on_test_end(self, trainer, pl_module):
        pl_module.forward = self.old_forward
        self.times = self.times[self.n_warm_start :]  # ensure warm start
        self.dims = self.dims[self.n_warm_start :]

        if not len(self.times):
            raise ValueError("No times recorded.")

        self.times = torch.tensor(self.times)
        self.mean_time = self.times.mean().item()
        self.std_time = self.times.std().item()

        self.times_path = Path(trainer.log_dir) / "times"
        self.times_path.mkdir(parents=True, exist_ok=True)

        np.save(self.times_path / f"{pl_module.name}_times.npy", self.times)
        np.save(self.times_path / f"{pl_module.name}_dims.npy", self.dims)

    def teardown(self, trainer, pl_module, stage):
        if len(self.times):
            print("-" * 80)
            print(f"Mean inference time: {self.mean_time:.2f} ± {self.std_time:.2f} ms")
            print(f"Saved timing info to {self.times_path}")
            print("-" * 80)
