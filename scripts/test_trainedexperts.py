












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






