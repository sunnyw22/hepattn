from typing import Literal

import torch
from lightning import LightningModule
from lion_pytorch import Lion
from torch import Tensor, nn
from torch._functorch import config as functorch_config  # noqa: PLC2701
from torch.optim import AdamW
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad


def gls_per_layer(layer_losses: dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """Computes the geometric mean across the layer.

    Parameters
    ----------
    layer_losses: Losses from the tasks in a layer (if available).
    """
    losses = torch.stack(list(layer_losses.values()))
    return torch.exp(torch.log(losses + eps).mean())


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: Literal["AdamW", "Lion"] = "AdamW",
        loss_mode: Literal["wsum", "GLS"] = "wsum",
        mtl: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        self.mtl = mtl

        loss_modes = ["wsum", "GLS"]
        assert loss_mode in loss_modes, f"Allowed loss modes are {loss_modes}, but got {loss_mode}"
        self.loss_mode = loss_mode
        if loss_mode == "GLS":
            for task in self.model.tasks:
                if hasattr(task, "losses"):
                    assert all(w == 1.0 for w in task.losses.values()), (
                        f"GLS mode requires all loss weights = 1, but got {list(task.losses.values())} in {task.name}"
                    )
                elif hasattr(task, "loss_weight"):  # For IncidenceBasedRegressionTasks in CLIC
                    assert task.loss_weight == 1.0, f"GLS mode requires loss weights=1, but got {task.loss_weight} in {task.name}"

        # If we are doing multi-task-learning, optimisation step must be done manually
        if mtl:
            # Donated buffers can cause issues with graph retention needed for MTL
            functorch_config.donated_buffer = False
            # If we are doing multi-task-learning, optimisation step must be done manually
            self.automatic_optimization = False
            # MTL does not currently support intermediate losses
            assert all(task.has_intermediate_loss is False for task in self.model.tasks)

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model.predict(outputs)

    def aggregate_losses(self, losses: dict[str, Tensor], stage: str | None = None) -> Tensor:
        device = next(self.model.parameters()).device
        # total_loss = torch.tensor(0.0, device=device)

        # log loss per task
        per_task_loss = {}

        for layer_name, layer_losses in losses.items():
            layer_loss = 0
            for task_name, task_losses in layer_losses.items():
                task_layer_sum = torch.tensor(0.0, device=device)

                for loss_value in task_losses.values():
                    task_layer_sum = task_layer_sum + loss_value
                    layer_loss += loss_value

                if task_name not in per_task_loss:
                    per_task_loss[task_name] = torch.tensor(0.0, device=device)

                per_task_loss[task_name] = per_task_loss[task_name] + task_layer_sum

            # Log the total loss from the layer
            self.log(f"{stage}/{layer_name}_loss", layer_loss, sync_dist=True)

        # Log the total loss
        total_loss = torch.stack(list(per_task_loss.values())).sum() if self.loss_mode == "wsum" else gls_per_layer(per_task_loss)

        self.log(f"{stage}/loss", total_loss, sync_dist=True)
        return total_loss

    def log_task_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        # Log any task specific metrics
        for task in self.model.tasks:
            # Check that the task actually has some metrics to log
            if not hasattr(task, "metrics"):
                continue

            # Just log the predictions from the final layer for now
            task_metrics = task.metrics(preds["final"][task.name], targets)

            # If the task returned a non-empty metrics dict, log it
            if task_metrics:
                self.log_dict({f"{stage}/final_{task.name}_{k}": v for k, v in task_metrics.items()}, sync_dist=True)

    def log_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        # First log any task metrics
        self.log_task_metrics(preds, targets, stage)

        # Log any custom metrics implemented by subclass
        if hasattr(self, "log_custom_metrics"):
            self.log_custom_metrics(preds, targets, stage)

    def training_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]], batch_idx: int) -> dict[str, Tensor] | None:
        inputs, targets = batch

        # Get the model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        losses, targets = self.model.loss(outputs, targets)

        # Get the predictions from the model, avoid calling predict if possible
        if batch_idx % self.trainer.log_every_n_steps == 0:
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")

        if self.mtl:
            self.mlt_opt(losses, outputs)
            return None
        total_loss = self.aggregate_losses(losses, stage="train")
        return {"loss": total_loss, **outputs}

    def validation_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> dict[str, Tensor]:
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Compute losses then aggregate and log them
        losses, targets = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")

        return {"loss": total_loss, **outputs}

    def test_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss to also run matching
        losses, targets = self.model.loss(outputs, targets)

        # Get the predictions from the model
        preds = self.model.predict(outputs)

        return outputs, preds, losses

    def on_train_start(self) -> None:
        # Manually overwride the learning rate in case we are starting
        # from a checkpoint that had a LRS and now we want a flat LR
        if self.lrs_config.get("skip_scheduler"):
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lrs_config["initial"]

    def configure_optimizers(self):
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.opt_config['opt']}")

        opt = optimizer(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            # Configure the learning rate scheduler
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            sch = {"scheduler": sch, "interval": "step"}
            return [opt], [sch]

        print("Skipping learning rate scheduler.")
        return opt

    def mlt_opt(self, losses: dict[str, Tensor], outputs: dict[str, Tensor]) -> None:
        opt = self.optimizers()
        opt.zero_grad()

        # TODO: Make this not hard coded?
        feature_names = ["query_embed", "key_embed"]

        # Remove any duplicate features that are used by multiple tasks
        features = [outputs["final"][feature_name] for feature_name in feature_names]

        # TODO: Figure out if we can set retain_graph to false somehow, since it uses a lot of memory
        task_losses = [sum(losses["final"][task.name].values()) for task in self.model.tasks]
        mtl_backward(losses=task_losses, features=features, aggregator=UPGrad(), retain_graph=True)

        # Manually perform the optimizer step
        opt.step()
