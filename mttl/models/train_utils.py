import os

import torch
from tqdm.auto import tqdm

from mttl.datamodule.base import DataModule
from mttl.logging import logger
from mttl.models.base_model import WEIGHTS_NAME, BaseExpertModel
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.utils import transfer_batch_to_device
import wandb
from transformers import TrainingArguments
from trl import SFTTrainer  




@torch.no_grad()
def evaluate_model(dataloader, model):
    """Evaluation loop."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        with torch.autocast(
            device_type=model.device.type,
            dtype=model.dtype,
        ):
            batch = transfer_batch_to_device(batch, model.device)
            output = model.forward(**batch)
            total_loss += output.loss.item()
            total_samples += 1
    return total_loss / total_samples


def train_model(
    args: "TrainingArguments",
    model: BaseExpertModel,
    datamodule: DataModule,
    do_test=False,
) -> BaseExpertModel:
    """Mini-training loop."""
    import copy

    args = copy.deepcopy(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # wandb = args.get("wandb")
    wandb_run = args.wandb
    

    (optimizer, scheduler), _ = get_optimizer_and_scheduler(
        model, args, num_train_examples=len(datamodule.train_dataset)
    )
    dataloader = datamodule.train_dataloader()
    num_train_steps = len(dataloader)
    iter_train = iter(dataloader)

    if args.eval_every_n_epoch != -1:
        args.eval_every = num_train_steps * args.eval_every_n_epoch

    steps_per_epoch = len(dataloader)
    args.total_steps = steps_per_epoch * args.num_train_epochs
    bar = tqdm(range(args.total_steps))
    best_val_loss = float("inf")
    running_loss = 0.0

    if wandb_run:
        wandb_run.watch(model, log="all", log_freq=10)  # Watch model parameters & gradients


    for step in bar:
        current_epoch = step // steps_per_epoch

        loss_accum = 0.0
        model.train()
        optimizer.zero_grad()
        current_epoch = step // steps_per_epoch

        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(iter_train)
            except StopIteration:
                iter_train = iter(dataloader)
                batch = next(iter_train)

            with torch.autocast(
                device_type=model.device.type,
                dtype=model.dtype,
            ):
                batch = transfer_batch_to_device(batch, model.device)
                loss = model.forward(**batch).loss
                loss = loss / args.gradient_accumulation_steps
                loss_accum += loss.detach()
                loss.backward()

        if loss_accum:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            running_loss += loss_accum.item()
            optimizer.step()
            scheduler.step()
            if model.device.type == "cuda":
                torch.cuda.synchronize()

            bar.set_description_str(
                f"Epoch {current_epoch+1}/{args.num_train_epochs}, "
                f"Step {step + 1}/{args.total_steps},"
                f" Loss: {running_loss / (step + 1):.4f},"
                f" Lr: {scheduler.get_last_lr()[0]:.4f},"
                f" Val: {best_val_loss:.4f}"
            )

            # Log to wandb every N steps to reduce overhead
            if wandb_run and step % 10 == 0:  # Log every 10 steps
                wandb_run.log({
                    "train/loss": running_loss / (step + 1),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": norm.item(),
                    "train/epoch": current_epoch + 1,
                    "step": step + 1,
                    "val/loss": best_val_loss,
                }, commit=False)  # Commit=False for better efficiency

        # eval and save best model
        if (
            args.eval_every > 0
            and step % args.eval_every == 0
            and datamodule.dev_dataset
        ):
            val_loss = evaluate_model(datamodule.val_dataloader(), model)
            
            if wandb_run:
                wandb_run.log({"val/loss": val_loss, "train/epoch": current_epoch + 1}, commit=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.output_dir:
                    model.save_pretrained(args.output_dir + "/best_model")
            running_loss = 0.0

    # reload best model
    if args.output_dir and os.path.exists(
        args.output_dir + f"/best_model/{WEIGHTS_NAME}"
    ):
        logger.info("Reloading best model!")

        model.load_state_dict(
            torch.load(
                args.output_dir + f"/best_model/{WEIGHTS_NAME}", weights_only=True
            ),
            strict=False,
        )

    # do test evaluation
    if do_test and datamodule.test_dataset:
        test_loss = evaluate_model(datamodule.test_dataloader(), model)
        logger.info(f"Test loss: {test_loss:.4f}")
        if wandb_run:
            wandb_run.log({"test/loss": test_loss})

        if wandb_run:
            wandb_run.log({"test/loss": test_loss})
            

    return model

from transformers import TrainerCallback

class WandbEpochCallback(TrainerCallback):
    """Custom Callback to log loss per epoch to Weights & Biases (W&B)."""
    
    def __init__(self, wandb_run):
        self.wandb_run = wandb_run
        self.epoch_loss = 0.0
        self.epoch_steps = 0
        self.current_epoch = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.epoch_loss = 0.0
        self.epoch_steps = 0
        self.current_epoch = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Accumulate loss after each step."""
        if state.log_history:
            loss = state.log_history[-1].get("loss", None)
            if loss is not None:
                self.epoch_loss += loss
                self.epoch_steps += 1

    def on_epoch_end(self, args, state, control, **kwargs):
        """Log average loss at the end of each epoch."""
        if self.epoch_steps > 0:
            avg_loss = self.epoch_loss / self.epoch_steps
            if self.wandb_run:
                self.wandb_run.log({"epoch": self.current_epoch + 1, "epoch_loss": avg_loss})
            print(f"Epoch {self.current_epoch + 1}: Loss = {avg_loss:.4f}")
        
        self.epoch_loss = 0.0
        self.epoch_steps = 0
        self.current_epoch += 1


def train_sft_model(args: TrainingArguments, cfg,  wandb_run, model, tokenizer, datamodule, do_test=False):
    """Fine-tune the model using SFTTrainer 
    with integrated Weights & Biases logging."""
    
    # Initialize W&B
    if wandb_run:
        wandb_run.watch(model, log="all", log_freq=10)
    # args.model_init_kwargs ={}
        
    # Ensure that the dataset has the correct column names
    if 'source' in datamodule.train_dataset.column_names:
        datamodule.train_dataset = datamodule.train_dataset.rename_column("source", "text")
    if 'target' in datamodule.train_dataset.column_names:
        datamodule.train_dataset = datamodule.train_dataset.rename_column("target", "labels")
    

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=datamodule.train_dataset,
        eval_dataset=datamodule.test_dataset if datamodule.test_dataset else None,
        tokenizer=datamodule.tokenize_dataset,
        # data_collator=datamodule.data_collator,
        # compute_metrics=None,  # Define if needed
        callbacks=[WandbEpochCallback(wandb_run)],  # Custom callback for epoch logging
    )

    # Start training
    trainer.train()

    # Log final model
    if args.output_dir:
        trainer.save_model(args.output_dir)

    # Test evaluation
    if do_test and datamodule.test_dataset:
        test_metrics = trainer.evaluate(datamodule.test_dataset)
        if wandb_run:
            wandb_run.log(test_metrics)

    return trainer.model
