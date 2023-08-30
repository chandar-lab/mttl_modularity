from collections import defaultdict
from copy import deepcopy
import tqdm
import torch
import numpy as np
import pytorch_lightning as pl

from mttl.dataloader.ni_metrics import compute_metrics
from mttl.models.utils import transfer_batch_to_device


class NIEvaluator(object):
    def __init__(self, config, data_dir=None, num_pos_examples=0, device="cuda"):
        from mttl.datamodule.ni_original_data_module import NIOriginalDataModule

        self.config = deepcopy(config)
        self.device = device
        self.config.num_pos_examples = num_pos_examples

        if data_dir is None:
            data_dir = config.data_dir

        self.data_dir = data_dir
        self.datamodule = NIOriginalDataModule(
            self.config,
            data_dir=data_dir,
            for_generation=True,
        )
        self.datamodule.setup("test")

    def evaluate(self, model, metric_per_task=True, eval_batches=-1):
        was_train = model.training
        if was_train:
            model.eval()

        tokenizer = self.datamodule.tokenizer
        samples_seen = 0

        # DDP
        if hasattr(model, "module"):
            model = model.module

        def decode(preds):
            preds[preds == -100] = tokenizer.pad_token_id
            preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [pred.strip() for pred in preds]
            return preds

        all_predictions = []
        all_references = []
        task_names = []
        all_rougeL = []

        dataloader = self.datamodule.test_dataloader(shuffle=eval_batches > 0)

        if eval_batches == -1:
            eval_batches = len(dataloader)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=min(len(dataloader), eval_batches),
        )
        for step, batch in pbar:
            task_name = batch.pop("task_names", None)
            batch.pop("input_texts", None)
            
            # we use labels texts here for evaluation, because some tokenizers do not skip
            # pad token when decoding, even if skip_special_tokens=True
            labels_texts = batch.pop("labels_texts", None)

            extra_kwargs = {}
            max_length = 128  # default output length for NI

            if self.config.model_family == 'gpt':
                max_length += batch['input_ids'].shape[-1]

                extra_kwargs['pad_token_id'] = tokenizer.pad_token_id

            batch = transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                if isinstance(model, pl.LightningModule):
                    predictions = model.generate(
                        batch,
                        max_length=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                    )
                else:
                    predictions = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                    )

            predictions = predictions.sequences
            predictions = predictions[:, batch["input_ids"].shape[-1] :]
            predictions = decode(predictions)

            references = labels_texts

            # If we are in a multiprocess environment, the last batch has duplicates
            if step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                references = references[: len(dataloader.dataset) - samples_seen]
                task_name = task_name[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += len(references)

            all_predictions += predictions
            all_references += references
            task_names += task_name
        
            eval_metrics = compute_metrics(
               predictions, [[r] for r in references], reduction="mean"
            )
            all_rougeL.append(eval_metrics["rougeL"])
            pbar.set_description(f"Task: {task_name[0] if task_name else None}, rougeL: {np.mean(all_rougeL):.4f}")
            
            if step == eval_batches:
                break

        eval_metrics = compute_metrics(
            all_predictions, [[r] for r in all_references], reduction="none"
        )
        mean_metrics = {}
        for metric_name, metric_value in eval_metrics.items():
            metric_value = sum(eval_metrics[metric_name]) / len(eval_metrics[metric_name])
            mean_metrics[metric_name] = metric_value

        if metric_per_task:
            metric_values_all = {}

            for metric_name in eval_metrics.keys():
                metric_values = defaultdict(list)
                for task_name, v in zip(task_names, eval_metrics[metric_name]):
                    metric_values[task_name] += [v]
                metric_values["all"] = [mean_metrics[metric_name]]
                metric_values = {
                    task_name: sum(vs) / len(vs)
                    for task_name, vs in metric_values.items()
                }
                metric_values_all[metric_name] = metric_values
            metric_values = metric_values_all
        else:
            metric_values = mean_metrics

        if was_train:
            model.train()
        return metric_values
