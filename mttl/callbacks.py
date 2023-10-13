import datetime
import time
import sys, os
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT

import pytorch_lightning as pl
from pytorch_lightning import callbacks as cb
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


from mttl.utils import Averager, logger


class MMLUCallback(cb.Callback):
    def __init__(
        self, eval_every, every_val_epochs=1, max_input_length=None, eval_split="test"
    ):
        super().__init__()

        self.val_epoch = 0
        self.eval_every = eval_every
        self.every_val_epochs = every_val_epochs
        self.max_input_length = max_input_length
        self.eval_split = eval_split
        self.evaluator = None

    def on_train_batch_start(
        self, trainer, pl_module, batch: Any, batch_idx: int
    ) -> None:
        if batch_idx == 0:
            metrics = self.eval_mmlu(pl_module)
            self.log_metrics(metrics, pl_module)

        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(
        self, trainer, pl_module, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if (
            (batch_idx != 0 and batch_idx % self.eval_every == 0)
            or self.val_epoch % self.every_val_epochs == 0
            or batch_idx == len(trainer.train_dataloader) - 1
        ):
            self.val_epoch += 1
            metrics = self.eval_mmlu(pl_module)
            self.log_metrics(metrics, pl_module)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def log_metrics(self, metrics, pl_module: pl.LightningModule):
        pl_module.log("val/mmlu", metrics["all"]["mean"], on_step=True, prog_bar=True)
        for t, v in metrics.items():
            pl_module.log(
                f"val/mmlu_{t}",
                v["mean"],
                on_step=True,
            )

    def eval_mmlu(self, pl_module):
        from mttl.evaluators import MMLUEvaluator
        from mttl.datamodule.mmlu_data_module import MMLUDataConfig

        if self.evaluator is None:
            mmlu_data_config = MMLUDataConfig(
                model=pl_module.hparams.model,
                predict_batch_size=pl_module.hparams.predict_batch_size,
                max_input_length=pl_module.hparams.max_input_length
                if self.max_input_length is None
                else self.max_input_length,
                max_output_length=pl_module.hparams.max_input_length,  # not necessary
                model_family=pl_module.hparams.model_family,
                finetune_task_name=pl_module.hparams.finetune_task_name,
            )
            self.evaluator = MMLUEvaluator(
                mmlu_data_config,
                split=self.eval_split,
            )

        metrics = self.evaluator.evaluate(pl_module)
        return metrics


class NICallback(cb.Callback):
    def __init__(self, every_val_epochs=1, **kwargs):
        super().__init__()

        self.val_epoch = 0
        self.every_val_epochs = every_val_epochs
        self.eval_kwargs = kwargs

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.val_epoch += 1

        if trainer.global_step == 0:
            return

        if self.val_epoch % self.every_val_epochs != 0:
            return

        from mttl.evaluators import NIEvaluator

        evaluator = NIEvaluator(
            pl_module.hparams,
            data_dir=os.environ["NI_DATA_DIR"],
            num_pos_examples=2,
            **self.eval_kwargs,
        )
        metrics = evaluator.evaluate(
            pl_module,
            eval_batches=50,
        )
        pl_module.log(
            "val/sni",
            metrics["all"]["mean"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class MiniProgress(cb.ProgressBar):
    def __init__(self):
        super().__init__()
        self.averager = Averager(0.9)

    def on_train_batch_start(
        self, trainer, pl_module, batch: Any, batch_idx: int
    ) -> None:
        self.time_start = time.time()

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        self.time_end = time.time()
        metrics = self.get_metrics(trainer, pl_module)
        metrics = {k: v for k, v in metrics.items()}
        it_per_sec = 1 / (self.time_end - self.time_start)

        # num total steps will be min of num_training_batches and max_steps
        if trainer.max_steps > -1:
            num_total_steps = min(
                trainer.num_training_batches * trainer.max_epochs, trainer.max_steps
            )
        else:
            num_total_steps = (
                trainer.num_training_batches // trainer.accumulate_grad_batches
            ) * trainer.max_epochs

        eta = (num_total_steps - batch_idx) / (
            1.0 / ((self.time_end - self.time_start))
        )

        time_metrics = self.averager.update({"it/s": it_per_sec, "eta": eta})
        for k, v in {**metrics, **time_metrics}.items():
            if k == "eta":
                metrics[k] = "{}".format(datetime.timedelta(seconds=v))
            else:
                metrics[k] = "{:.2f}".format(v) if isinstance(v, float) else v

        msg_start = (
            f"Trn - Epc {trainer.current_epoch} / {trainer.global_step} / {num_total_steps}"
            + " | "
        )
        dict_msg = " | ".join([f"{k} -> {v}" for k, v in metrics.items()]) + " | "
        msg = msg_start + dict_msg
        logger.info(msg)

    def on_validation_batch_start(
        self, trainer, pl_module, batch: Any, batch_idx: int
    ) -> None:
        self.time_start = time.time()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        self.time_end = time.time()
        metrics = self.get_metrics(trainer, pl_module)
        metrics = {k: v for k, v in metrics.items()}
        metrics["it/s"] = 1.0 / (self.time_end - self.time_start)
        for k, v in metrics.items():
            metrics[k] = "{:.2f}".format(v) if isinstance(v, float) else v

        msg_start = (
            f"Val - Epc {trainer.current_epoch} / {batch_idx} / {trainer.num_val_batches[0]}"
            + " | "
        )
        dict_msg = " | ".join([f"{k} -> {v}" for k, v in metrics.items()]) + " | "
        msg = msg_start + dict_msg
        logger.info(msg)


class ProgressCallback(cb.TQDMProgressBar):
    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        bar = Tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc="Training",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            smoothing=0,
        )
        return bar

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = Tqdm(
            desc="Predicting",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = Tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar
