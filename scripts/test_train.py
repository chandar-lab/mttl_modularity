import torch
import os
import signal
from tqdm.auto import tqdm
from mttl.models.get_optimizer import get_optimizer_and_scheduler
import wandb

def train_model(args, model, datamodule, checkpoint_path=None, resume_from_checkpoint=False, wandb_run=None):
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)
    model.train()

    start_epoch, global_step = 0, 0
    interrupted = False

    def save_checkpoint(epoch, step):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'step': global_step
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}, step {global_step}")

    def handle_preemption(signum, frame, epoch):
        nonlocal interrupted
        print("Preemption signal received! Saving checkpoint...")
        save_checkpoint(epoch, global_step)
        if wandb_run:
            wandb_run.finish()
        interrupted = True

    # Register signal handler for SIGTERM
    import signal
    interrupted = False
    signal.signal(signal.SIGTERM, handle_preemption)

    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['step']
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Initialize wandb if not already initialized
    if wandb_run is None:
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, resume="allow")

    import signal
    interrupted = False
    signal.signal(signal.SIGTERM, handle_preemption)

    for epoch in range(start_epoch, args.num_train_epochs):
        if interrupted:
            break
        for batch in datamodule.train_dataloader():
            if interrupted:
                break

            optimizer.zero_grad()
            batch = transfer_batch_to_device(batch, model.device)

            loss = model.forward(**batch).loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % args.save_every == 0:
                save_checkpoint(epoch, global_step)

            if wandb_run and global_step % args.log_every == 0:
                wandb_run.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch,
                    "train/step": global_step
                })

    if wandb_run:
        wandb_run.finish()

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
    torch.save(checkpoint, checkpoint_path)



def train_model(args, model, datamodule, checkpoint_path=None, resume_from_checkpoint=False, wandb_run=None):
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)
    model.train()

    start_epoch, global_step = 0, 0
    wandb_run_id = None

    # Load checkpoint if exists
    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
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

    else:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name
        )


    # # Initialize WandB with resume
    # wandb_run = wandb.init(
    #     project=args.wandb_project,
    #     entity=args.wandb_entity,
    #     name=args.run_name,
    #     resume="allow",
    #     id=wandb_run_id
    # )

    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            for batch in datamodule.train_dataloader():
                optimizer.zero_grad()
                batch = transfer_batch_to_device(batch, model.device)

                loss = model.forward(**batch).loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                global_step += 1

                if global_step % args.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, wandb_run, checkpoint_path)
                    print(f"Checkpoint saved at step {global_step}")

                if global_step % args.save_every == 0:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'step': global_step,
                        'wandb_run_id': wandb_run.id
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Checkpoint saved at step {global_step}")

                if global_step % args.log_every == 0:
                    wandb_run.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch,
                        "train/step": global_step
                    })

    except KeyboardInterrupt:
        print("Training interrupted, saving state...")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'step': global_step,
            'wandb_run_id': wandb_run.id
        }
        torch.save(checkpoint, checkpoint_path)
        print("Checkpoint saved due to interruption.")

    finally:
        wandb_run.finish()

    return model
