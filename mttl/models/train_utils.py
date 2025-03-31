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
import copy
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
import re
import signal
import sys
import json
import glob
import shutil





@torch.no_grad()
def evaluate_model(dataloader, model,):
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
    checkpoint_path=None, 
    resume_from_checkpoint=False,
) -> BaseExpertModel:
    """Mini-training loop."""


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

    # if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
    #     print(f"Resuming training from checkpoint: {checkpoint_path}")
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     global_step = checkpoint['step']
    #     print(f"Resumed from epoch {start_epoch}, step {global_step}")

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



def save_checkpoint(model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': global_step,
        'wandb_run_id': wandb_run.id if wandb_run else None
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save(checkpoint, f'{checkpoint_path}/checkpoint_{epoch}.pth')

  
def my_train_model(
    args: "TrainingArguments",
    model: BaseExpertModel,
    datamodule: DataModule,
    do_test=False,
    checkpoint_path=None, 
    resume_from_checkpoint=False,
    wandb_run=None
    ) -> BaseExpertModel:
    
    
    # optimizer, scheduler = get_optimizer_and_scheduler(model, args)
    model.train()

    start_epoch, global_step = 0, 0

    args = copy.deepcopy(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    wandb_run = args.wandb
    

    (optimizer, scheduler), _ = get_optimizer_and_scheduler(
        model, args, num_train_examples=len(datamodule.train_dataset)
    )
    dataloader = datamodule.train_dataloader()
    num_train_steps = len(dataloader)
    iter_train = iter(dataloader)

    # if args.eval_every_n_epoch != -1:
    #     args.eval_every = num_train_steps * args.eval_every_n_epoch

    # steps_per_epoch = len(dataloader)
    # args.total_steps = steps_per_epoch * args.num_train_epochs
    # bar = tqdm(range(args.total_steps))
    best_val_loss = float("inf")
    running_loss = 0.0


    # Load checkpoint if exists
    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        saved_path =  f'{checkpoint_path}/checkpoint_{epoch}.pth'
        checkpoint = torch.load(saved_path) ## add model.pt 
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['step']
        wandb_run_id = checkpoint.get('wandb_run_id')
        
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            resume="allow",
            id=wandb_run_id  # This resumes your previous run
        )
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # else:
        # wandb_run = wandb.init(
        #     project=args.wandb.project,
        #     entity=args.wandb.entity,
        #     name=args.run_name
        # )


    for epoch in range(start_epoch, args.num_train_epochs):
        dataloader = datamodule.train_dataloader()
        iter_train = iter(dataloader)
        step = 0
        with tqdm(total=len(dataloader) // args.gradient_accumulation_steps, desc=f"Epoch {epoch+1}/{args.num_train_epochs}") as bar:
            for _ in range(len(dataloader) // args.gradient_accumulation_steps):

                optimizer.zero_grad()
                loss_accumulated = 0.0
                loss_accum = 0.0

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
                        loss.backward()
                        loss_accumulated += loss.item()
                        loss_accum += loss.detach()

                
                
                if loss_accum:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    running_loss += loss_accum.item() ###
                    optimizer.step()
                    scheduler.step()
                    if model.device.type == "cuda":
                        torch.cuda.synchronize()
                
                
                bar.set_description_str(
                    f"Epoch {epoch+1}/{args.num_train_epochs}, "
                    f"Step {global_step+1}/{args.total_steps},"
                    f" Loss_accu: {loss_accumulated:.4f},"
                    f" Loss: {running_loss / (global_step + 1):.4f},"
                    f" Lr: {scheduler.get_last_lr()[0]:.4f},"
                    f" Val: {best_val_loss:.4f}"
                )
                

                if global_step % args.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path)
                    print(f"Checkpoint saved at step {global_step}")

                
                # Log to wandb every N steps to reduce overhead
                if wandb_run and step % 10 == 0:  # Log every 10 steps
                    wandb_run.log({
                        "train/loss_accu": loss_accumulated,
                        "train/loss": running_loss / (global_step + 1),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": norm.item(),
                        "train/epoch": epoch + 1,
                        "step": global_step + 1,
                        "val/loss": best_val_loss,
                    }, commit=False)  # Commit=False for better efficiency
                
                
                
                # if wandb_run and global_step % args.log_every == 0:
                #     wandb_run.log({
                #         "train/loss": loss_accumulated,
                #         "train/epoch": epoch,
                #         "train/step": global_step
                    # })

                global_step += 1
                # step += 1
                bar.update(1)
                
                

                # eval and save best model
                if (
                    args.eval_every > 0
                    and global_step % args.eval_every == 0
                    and datamodule.dev_dataset):

                    val_loss = evaluate_model(datamodule.val_dataloader(), model)
                    
                    if wandb_run:
                        wandb_run.log({"val/loss": val_loss, "train/epoch": epoch + 1, "step": global_step + 1}, commit=True)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if args.output_dir:
                            model.save_pretrained(args.output_dir + "/best_model")
                    # running_loss = 0.0
    
        val_loss = evaluate_model(datamodule.val_dataloader(), model)
        if wandb_run:
            wandb_run.log({"val/loss": val_loss, "train/epoch": epoch + 1}, commit=True)

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






# def save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path):    
#     checkpoint = {
#         'model_state_dict': accelerator.unwrap_model(model).state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'epoch': epoch,
#         'step': global_step,
#         'wandb_run_id': wandb_run.id if wandb_run else None
#     }
#     # os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
#     # accelerator.save(checkpoint, f'{checkpoint_path}/checkpoint_{epoch}.pth')

#     checkpoint_dir = checkpoint_path  # now it's clearly a directory
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')


# def handler(signum, frame):
#     print(f"Received signal {signum}, saving checkpoint and exiting...")
#     # Save your model/checkpoint here
#     # torch.save(...), etc.
#     sys.exit(0)

# # Handle SIGTERM (15) and SIGINT (2)
# signal.signal(signal.SIGTERM, handler)
# signal.signal(signal.SIGINT, handler)


def save_checkpoint_acce_before(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")
    checkpoint = {
        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': global_step,
        'wandb_run_id': wandb_run.id if wandb_run else None
    }
    accelerator.save(checkpoint, checkpoint_file)
    accelerator.print(f" Saved checkpoint to: {checkpoint_file}")

def save_checkpoint_acce_2(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, output_dir):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save model, optimizer, scheduler, RNG, and deepspeed internals
    accelerator.save_state(checkpoint_dir)

    if accelerator.is_main_process:
        client_state = {
            "epoch": epoch,
            "step": global_step,
            "wandb_run_id": wandb_run.id if wandb_run else None
        }
        with open(os.path.join(checkpoint_dir, "client_state.json"), "w") as f:
            json.dump(client_state, f)

        accelerator.print(f" Checkpoint saved to {checkpoint_dir}")
        # cleaning up
        max_checkpoints = 3
        all_ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=os.path.getmtime)
        if len(all_ckpts) > max_checkpoints:
            for ckpt_to_remove in all_ckpts[:-max_checkpoints]:
                shutil.rmtree(ckpt_to_remove)


def save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, output_dir):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # All processes must call this
        accelerator.save_state(checkpoint_dir)
    except Exception as e:
        accelerator.print(f"Error in save_state on rank {accelerator.process_index}: {repr(e)}")
        import traceback
        traceback.print_exc()
        accelerator.end_training()
        raise

    # Only main process does metadata stuff
    if accelerator.is_main_process:
        try:
            client_state = {
                "epoch": epoch,
                "step": global_step,
                "wandb_run_id": wandb_run.id if wandb_run else None
            }
            with open(os.path.join(checkpoint_dir, "client_state.json"), "w") as f:
                json.dump(client_state, f)

            accelerator.print(f"Checkpoint saved to {checkpoint_dir}")
        except Exception as e:
            accelerator.print(f"Failed writing client_state.json: {repr(e)}")


def get_latest_checkpoint_file(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    # ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
    if not ckpts:
        return None
    # Extract step number from filename: checkpoint_123.pth
    # ckpts.sort(key=lambda f: int(re.findall(r"checkpoint_(\\d+).pth", f)[0]))
    # ckpts.sort(key=lambda f: int(re.search(r'\d+', f).group()) for f in ckpts)
    # ckpts.sort(int([re.findall(r'\d+', ckpt)[0] for ckpt in ckpts]))
    max_ckpt = max(int(re.findall(r'\d+', x)[0]) for x in ckpts)-1
    last_ckp = "checkpoint-"+str(max_ckpt) #last_ckp = "checkpoint_"+str(max_ckpt)+".pth"
    
    return os.path.join(checkpoint_dir, last_ckp)




def my_train_model_acce_1(args, model, datamodule, do_test=False, checkpoint_path=None, resume_from_checkpoint=False):
 
    start_epoch, global_step = 0, 0
    args = copy.deepcopy(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    wandb_run = args.wandb

    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps)
    # tracker = accelerator.get_tracker("wandb")
    # wandb_run_id = tracker._wandb_run.id
    # print("Current WandB run ID:", wandb_run_id)
    print('----------------------------------------------------------')
    accelerator.print(f"Using DeepSpeed: {accelerator.distributed_type == 'DEEPSPEED'}")
    accelerator.print(f"Accelerator state:\n{accelerator.state}")
    accelerator.print(f"Local rank: {accelerator.local_process_index} | Device: {accelerator.device}")



    (optimizer, scheduler), _ = get_optimizer_and_scheduler(
        model, args, num_train_examples=len(datamodule.train_dataset))
    
            
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, datamodule.train_dataloader())
    model_engine = accelerator.unwrap_model(model)
    print(f"Model engine class: {model_engine.__class__}")

    # if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
    # # if checkpoint_path and os.path.exists(checkpoint_path):
    #     accelerator.print(f"Resuming training from checkpoint: {checkpoint_path}")
    #     checkpoint_file = get_latest_checkpoint_file(checkpoint_path)
    #     checkpoint = torch.load(checkpoint_file, weights_only=False) #checkpoint = torch.load(checkpoint_file)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     global_step = checkpoint['step']
    #     wandb_run_id = checkpoint.get('wandb_run_id')
    
    #     wandb_run = wandb.init(
    #         project=args.wandb_project,
    #         entity=args.wandb_entity,
    #         name=args.run_name,
    #         resume="allow",
    #         id=wandb_run_id  # This resumes your previous run
    #     )
    #     print(f"Resumed from epoch {start_epoch}, step {global_step}")


    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        accelerator.print(f"Resuming from checkpoint at: {checkpoint_path}")
        checkpoint_path_last = get_latest_checkpoint_file(checkpoint_path)
        accelerator.load_state(checkpoint_path_last)
        # Load any custom training state
        client_state_path = os.path.join(checkpoint_path_last, "client_state.json")
        if os.path.exists(client_state_path):
            with open(client_state_path, "r") as f:
                client_state = json.load(f)
            start_epoch = client_state.get("epoch", 0)
            global_step = client_state.get("step", 0)
            wandb_run_id = client_state.get("wandb_run_id")

            wandb_run = wandb.init(
                project=args.wandb.project,
                entity=args.wandb.entity,
                name=args.wandb.name,
                resume="allow",
                id=wandb_run_id
            )
        print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # accelerator.init_trackers(
    #         project_name=args.wandb.project,
    #         config=vars(args),
    #         init_kwargs={"wandb": {"id": args.wandb.id, "resume": "allow"}}
    #     )

    
    ########################################################

    # def handle_sigusr1(signum, frame):
    #     accelerator.print("\nCaught SIGUSR1 — saving checkpoint before timeout...")
    #     save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path)
    #     accelerator.print("Checkpoint saved. Exiting gracefully.")
    #     accelerator.end_training()
    #     exit(0)

    # signal.signal(signal.SIGUSR1, handle_sigusr1)

    

    def handler(signum, frame):
        # print(" ------ Received SIGTERM, saving checkpoint... ----")
        accelerator.print("\n Caught SIGTERM — saving checkpoint before timeout... \n")
        save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path)
        accelerator.print("Checkpoint saved. Exiting gracefully. \n")
        accelerator.end_training()
        
        sys.exit(0) 


    # Handle SIGTERM (15) and SIGINT (2)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    
    ###############################################

    model.train()

    dataloader = datamodule.train_dataloader()
    num_train_steps = len(dataloader)
    iter_train = iter(dataloader)
    best_val_loss = float("inf")
    running_loss = 0.0

    for epoch in range(start_epoch, args.num_train_epochs):
        
        
        total_steps = len(dataloader) // args.gradient_accumulation_steps
        # progress_bar = tqdm(range(total_steps), desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        step = 0
        with tqdm(total=len(dataloader) // args.gradient_accumulation_steps, desc=f"Epoch {epoch+1}/{args.num_train_epochs}") as bar:
            for _ in range(len(dataloader) // args.gradient_accumulation_steps):

                optimizer.zero_grad()
                loss_accumulated = 0.0
                loss_accum = 0.0

                for micro_step in range(args.gradient_accumulation_steps):
                    try:
                        batch = next(iter_train)
                    except StopIteration:
                        iter_train = iter(dataloader)
                        batch = next(iter_train)

                    with torch.autocast(
                        device_type=accelerator.device.type,
                        dtype=model.dtype,
                    ):
                        batch = transfer_batch_to_device(batch, accelerator.device)
                        loss = model.forward(**batch).loss ##??
                        loss = loss / args.gradient_accumulation_steps
                        loss_accum += loss.detach()
                        loss_accumulated += loss.item()
                        accelerator.backward(loss)
                    
                    # batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    # outputs = model(**batch)
                    # loss = outputs.loss / args.gradient_accumulation_steps
                    # accelerator.backward(loss)
                    # loss_accumulated += loss.item()
                    # loss_accum += loss.detach()

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                running_loss += loss_accum.item() ###
                optimizer.step()
                scheduler.step()
                
                
                if accelerator.is_main_process:
                    bar.set_description_str(
                            f"Epoch {epoch+1}/{args.num_train_epochs}, "
                            f"Step {step+1}/{total_steps},"
                            f"global_step: {global_step + 1}",
                            f" Loss_accu: {loss_accumulated:.4f},"
                            f" Loss: {running_loss / (global_step + 1):.4f},"
                            f" Lr: {scheduler.get_last_lr()[0]:.4f},"
                            f" Val: {best_val_loss:.4f}"
                        )

                    if global_step % args.save_every == 0:
                        # save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, accelerator.get_tracker("wandb"), checkpoint_path)
                        save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path)
                        accelerator.print(f"Checkpoint saved at step {global_step}")

                
                    # Log to wandb every N steps to reduce overhead
                    # if wandb_run and step % 10 == 0:  # Log every 10 steps
                    #     wandb_run.log({
                    #         "train/loss_accu": loss_accumulated,
                    #         "train/loss": running_loss / (global_step + 1),
                    #         "train/lr": scheduler.get_last_lr()[0],
                    #         "train/grad_norm": norm.item(),
                    #         "train/epoch": epoch + 1,
                    #         "step": global_step + 1,
                    #         "val/loss": best_val_loss,
                    #     }, commit=False)  # Commit=False for better efficiency

                    accelerator.log({
                            "train/loss_accu": loss_accumulated,
                            "train/loss": running_loss / (global_step + 1),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/grad_norm": norm.item(),
                            "train/epoch": epoch + 1,
                            "step": global_step + 1,
                            "val/loss": best_val_loss,
                        }, step=global_step)
                    

                    # if global_step % args.log_every == 0:
                    #     accelerator.log({
                    #         "train/loss": loss_accumulated,
                    #         "train/epoch": epoch,
                    #         "train/step": global_step,
                    #         "train/lr": scheduler.get_last_lr()[0]
                    #     }, step=global_step)

                global_step += 1
                step += 1
                bar.update(1)



                # eval and save best model
                if (
                    args.eval_every > 0
                    and global_step % args.eval_every == 0
                    and datamodule.dev_dataset):

                    val_loss = evaluate_model(datamodule.val_dataloader(), model)
                    
                    if wandb_run:
                        # wandb_run.log({"val/loss": val_loss, "train/epoch": epoch + 1, "step": global_step}, commit=True)
                        accelerator.log({"val/loss": val_loss, "train/epoch": epoch + 1, "step": global_step}, step=global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if args.output_dir:
                            model.save_pretrained(args.output_dir + "/best_model")
                    # running_loss = 0.0
    
        val_loss = evaluate_model(datamodule.val_dataloader(), model)
        if wandb_run:
            # wandb_run.log({"val/loss": val_loss, "train/epoch": epoch + 1}, commit=True)
            accelerator.log({"val/loss": val_loss, "train/epoch": epoch + 1}, step=global_step)
            

    
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
            # wandb_run.log({"test/loss": test_loss})
            accelerator.log({"test/loss": test_loss})


    
    accelerator.end_training()

    return accelerator.unwrap_model(model)

def my_train_model_acce(args, model, datamodule, do_test=False, checkpoint_path=None, resume_from_checkpoint=False, wandb_run=None):
 
    start_epoch, global_step = 0, 0
    args = copy.deepcopy(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.print(f"Rank: {accelerator.process_index}, device: {accelerator.device}, is_main_process: {accelerator.is_main_process}")

    # tracker = accelerator.get_tracker("wandb")
    # wandb_run_id = tracker._wandb_run.id
    # print("Current WandB run ID:", wandb_run_id)
    print('----------------------------------------------------------')
    accelerator.print(f"Using DeepSpeed: {accelerator.distributed_type == 'DEEPSPEED'}")
    accelerator.print(f"Accelerator state:\n{accelerator.state}")
    accelerator.print(f"Local rank: {accelerator.local_process_index} | Device: {accelerator.device}")



    wandb_run = args.wandb

    (optimizer, scheduler), _ = get_optimizer_and_scheduler(
        model, args, num_train_examples=len(datamodule.train_dataset))
    
    # optimizer = DummyOptim(model.parameters())
    # scheduler = DummyScheduler(optimizer)
   



    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, datamodule.train_dataloader())
    model_engine = accelerator.unwrap_model(model)
    print(f"Model engine class: {model_engine.__class__}")

    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        accelerator.print(f"Resuming from checkpoint at: {checkpoint_path}")
        checkpoint_path_last = get_latest_checkpoint_file(checkpoint_path)
        accelerator.load_state(checkpoint_path_last)
        # Load any custom training state
        client_state_path = os.path.join(checkpoint_path_last, "client_state.json")
        if os.path.exists(client_state_path):
            with open(client_state_path, "r") as f:
                client_state = json.load(f)
            start_epoch = client_state.get("epoch", 0)
            global_step = client_state.get("step", 0)
            wandb_run_id = client_state.get("wandb_run_id")

            wandb_run = wandb.init(
                project=args.wandb.project,
                entity=args.wandb.entity,
                name=args.wandb.name,
                resume="allow",
                id=wandb_run_id
            )
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    
    # else:
    #     wandb_run_id = None


    # accelerator.init_trackers(project_name=args.wandb_project, config=vars(args), init_kwargs={'wandb': {'id': wandb_run_id, 'resume': 'allow'}})

    

    # Signal handler to save checkpoint before timeout
    
    ########################################################

    # def handle_sigusr1(signum, frame):
    #     accelerator.print("\nCaught SIGUSR1 — saving checkpoint before timeout...")
    #     save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path)
    #     accelerator.print("Checkpoint saved. Exiting gracefully.")
    #     accelerator.end_training()
    #     exit(0)

    # signal.signal(signal.SIGUSR1, handle_sigusr1)

    

    def handler(signum, frame):
        # print(" ------ Received SIGTERM, saving checkpoint... ----")
        accelerator.print("\n Caught SIGTERM — saving checkpoint before timeout... \n")
        save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path)
        accelerator.print("Checkpoint saved. Exiting gracefully. \n")
        accelerator.end_training()
        
        sys.exit(0) 


    # Handle SIGTERM (15) and SIGINT (2)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    
    ###############################################

    model.train()

    dataloader = datamodule.train_dataloader()
    num_train_steps = len(dataloader)
    iter_train = iter(dataloader)
    best_val_loss = float("inf")
    running_loss = 0.0

    for epoch in range(start_epoch, args.num_train_epochs):
        
        total_steps = len(dataloader) // args.gradient_accumulation_steps
        # progress_bar = tqdm(range(total_steps), desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        step = 0
        with tqdm(total=len(dataloader) // args.gradient_accumulation_steps, desc=f"Epoch {epoch+1}/{args.num_train_epochs}") as bar:
            for _ in range(len(dataloader) // args.gradient_accumulation_steps):

                optimizer.zero_grad()
                loss_accumulated = 0.0
                loss_accum = 0.0

                for micro_step in range(args.gradient_accumulation_steps):
                    try:
                        batch = next(iter_train)
                    except StopIteration:
                        iter_train = iter(dataloader)
                        batch = next(iter_train)

                    with torch.autocast(
                        device_type=accelerator.device.type,
                        dtype=model.dtype,
                    ):
                        batch = transfer_batch_to_device(batch, accelerator.device)
                        loss = model.forward(**batch).loss ##??
                        loss = loss / args.gradient_accumulation_steps
                        loss_accum += loss.detach()
                        loss_accumulated += loss.item()
                        accelerator.backward(loss)
                    
                    # batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    # outputs = model(**batch)
                    # loss = outputs.loss / args.gradient_accumulation_steps
                    # accelerator.backward(loss)
                    # loss_accumulated += loss.item()
                    # loss_accum += loss.detach()

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                running_loss += loss_accum.item() ###
                optimizer.step()
                scheduler.step()
                
                
                if accelerator.is_main_process:
                    bar.set_description_str(
                            f"Epoch {epoch+1}/{args.num_train_epochs}, "
                            f"Step {step+1}/{total_steps},"
                            f"global_step: {global_step + 1}",
                            f" Loss_accu: {loss_accumulated:.4f},"
                            f" Loss: {running_loss / (global_step + 1):.4f},"
                            f" Lr: {scheduler.get_last_lr()[0]:.4f},"
                            f" Val: {best_val_loss:.4f}"
                        )

                    

                
                    # Log to wandb every N steps to reduce overhead
                    if wandb_run and step % 10 == 0:  # Log every 10 steps
                        wandb_run.log({
                            "train/loss_accu": loss_accumulated,
                            "train/loss": running_loss / (global_step + 1),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/grad_norm": norm.item(),
                            "train/epoch": epoch + 1,
                            "step": global_step + 1,
                            "val/loss": best_val_loss,
                        }, commit=False)  # Commit=False for better efficiency
                        

                    # if global_step % args.log_every == 0:
                    #     accelerator.log({
                    #         "train/loss": loss_accumulated,
                    #         "train/epoch": epoch,
                    #         "train/step": global_step,
                    #         "train/lr": scheduler.get_last_lr()[0]
                    #     }, step=global_step)          

                
                
                if global_step % args.save_every == 0:
                        # save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, accelerator.get_tracker("wandb"), checkpoint_path)
                        save_checkpoint_acce(accelerator, model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path)
                        accelerator.print(f"Checkpoint saved at step {global_step}")
                
                global_step += 1
                step += 1
                bar.update(1)



                # eval and save best model
                if (
                    args.eval_every > 0
                    and global_step % args.eval_every == 0
                    and datamodule.dev_dataset):

                    val_loss = evaluate_model(datamodule.val_dataloader(), model)
                    
                    if wandb_run:
                        wandb_run.log({"val/loss": val_loss, "train/epoch": epoch + 1, "step": global_step + 1}, commit=True)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if args.output_dir:
                            model.save_pretrained(args.output_dir + "/best_model")
                    # running_loss = 0.0
    
        val_loss = evaluate_model(datamodule.val_dataloader(), model)
        if wandb_run:
            wandb_run.log({"val/loss": val_loss, "train/epoch": epoch + 1}, commit=True)

    
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

    
    accelerator.end_training()

    return accelerator.unwrap_model(model)





# def my_train_model_backup(args, model, datamodule, checkpoint_path=None, resume_from_checkpoint=False, wandb_run=None):
#     optimizer, scheduler = get_optimizer_and_scheduler(model, args)
#     model.train()

#     start_epoch, global_step = 0, 0
#     # wandb_run_id = None
#     wandb_run = args.wandb


#     # Load checkpoint if exists
#     if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         start_epoch = checkpoint['epoch']
#         global_step = checkpoint['step']
#         wandb_run_id = checkpoint.get('wandb_run_id')
        
#         wandb_run = wandb.init(
#             project=args.wandb_project,
#             entity=args.wandb_entity,
#             name=args.run_name,
#             resume="allow",
#             id=wandb_run_id  # This resumes your previous run
#         )
#         print(f"Resumed from epoch {start_epoch}, step {global_step}")

#     else:
#         wandb_run = wandb.init(
#             project=args.wandb_project,
#             entity=args.wandb_entity,
#             name=args.run_name
#         )


#     # # Initialize WandB with resume
#     # wandb_run = wandb.init(
#     #     project=args.wandb_project,
#     #     entity=args.wandb_entity,
#     #     name=args.run_name,
#     #     resume="allow",
#     #     id=wandb_run_id
#     # )

#     try:
#         for epoch in range(start_epoch, args.num_train_epochs):
#             for batch in datamodule.train_dataloader():
#                 loss_accum = 0.0
#                 optimizer.zero_grad()
                
                
#                 for micro_step in range(args.gradient_accumulation_steps):
#                     try:
#                         batch = next(iter_train)
#                     except StopIteration:
#                         iter_train = iter(dataloader)
#                         batch = next(iter_train)

#                     with torch.autocast(
#                         device_type=model.device.type,
#                         dtype=model.dtype,
#                     ):
#                         batch = transfer_batch_to_device(batch, model.device)
#                         loss = model.forward(**batch).loss
#                         loss = loss / args.gradient_accumulation_steps
#                         loss_accum += loss.detach()
#                         loss.backward()
                    
                    

#                 optimizer.step()
#                 scheduler.step()
#                 global_step += 1

#                 if global_step % args.save_every == 0:
#                     save_checkpoint(model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path)
#                     print(f"Checkpoint saved at step {global_step}")

#                 if global_step % args.save_every == 0:
#                     checkpoint = {
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'scheduler_state_dict': scheduler.state_dict(),
#                         'epoch': epoch,
#                         'step': global_step,
#                         'wandb_run_id': wandb_run.id
#                     }
#                     torch.save(checkpoint, checkpoint_path)
#                     print(f"Checkpoint saved at step {global_step}")

#                 if global_step % args.log_every == 0:
#                     wandb_run.log({
#                         "train/loss": loss.item(),
#                         "train/epoch": epoch,
#                         "train/step": global_step
#                     })

#     except KeyboardInterrupt:
#         print("Training interrupted, saving state...")
#         checkpoint = {
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'epoch': epoch,
#             'step': global_step,
#             'wandb_run_id': wandb_run.id
#         }
#         torch.save(checkpoint, checkpoint_path)
#         print("Checkpoint saved due to interruption.")

    
#         # reload best model
#     if args.output_dir and os.path.exists(
#         args.output_dir + f"/best_model/{WEIGHTS_NAME}"
#             ):
#         logger.info("Reloading best model!")

#         model.load_state_dict(
#             torch.load(
#                 args.output_dir + f"/best_model/{WEIGHTS_NAME}", weights_only=True
#             ),
#             strict=False,
#         )

#     # do test evaluation
#     if do_test and datamodule.test_dataset:
#         test_loss = evaluate_model(datamodule.test_dataloader(), model)
#         logger.info(f"Test loss: {test_loss:.4f}")
#         if wandb_run:
#             wandb_run.log({"test/loss": test_loss})

#         if wandb_run:
#             wandb_run.log({"test/loss": test_loss})
         
    
    
    
#     finally:
#         wandb_run.finish()

#     return model




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
