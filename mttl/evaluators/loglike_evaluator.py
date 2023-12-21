import tqdm
import torch
import numpy as np

from mttl.evaluators.base import Evaluator, switch_to_eval_mode
from mttl.models.utils import EfficientCheckpointModule, transfer_batch_to_device
from mttl.utils import logger


def compute_loglike_loss(logits, labels, reduction="none"):
    # calculate loss, could also be done inside of the model
    bs = logits.size(0)
    vocab_size = logits.size(-1)
    labels = labels.squeeze(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    # reshape back
    if reduction == "none":
        loss = loss.view((bs, -1))
        # mean only non-zero
        non_zero_loss = (loss != 0).sum(dim=-1)
        non_zero_loss[non_zero_loss == 0] = 1
        loss = loss.sum(dim=-1) / non_zero_loss
    return loss


class LogLikeEvaluator(Evaluator):
    def __init__(
        self, datamodule, device="cuda", use_vllm=False, generation_kwargs=None
    ):
        super().__init__(
            datamodule=datamodule,
            device=device,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="val",
        subsample=-1,
        num_batches=None,
        verbose=True,
        shuffle=False,
    ):
        dataloader = self.get_dataloader(split, subsample, shuffle=shuffle)

        if self.use_vllm:
            return self.evaluate_with_vllm(model, dataloader, num_batches, verbose)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        all_losses = []
        all_accuracies = []

        for num_batch, batch in pbar:
            batch_size = len(batch["labels_index"])
            labels_texts = batch["labels_texts"]
            sources_texts = batch["sources_texts"]

            batch = transfer_batch_to_device(batch, self.device)

            with torch.no_grad():
                if isinstance(model, EfficientCheckpointModule):
                    loss_per_example = model.forward(batch, reduction="none")
                else:
                    logits = model.forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                    loss_per_example = compute_loglike_loss(
                        logits, batch["labels"], reduction="none"
                    )

                loss_per_example = loss_per_example.cpu().numpy()
                all_losses.append(loss_per_example)

                if "labels_index" in batch:
                    counter = 0
                    for i in range(batch_size):
                        num_options = batch["num_options"][i]
                        pred = np.argmin(
                            loss_per_example[counter : counter + num_options]
                        )
                        all_accuracies.append(pred == batch["labels_index"][i])
                        counter += num_options

            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Label:\n%s", labels_texts[0])

            if num_batches is not None and num_batch >= num_batches:
                break

            if all_accuracies:
                pbar.set_description(
                    "Accuracy: {:.4f}".format(np.array(all_accuracies).mean())
                )

        all_losses = np.concatenate(all_losses)

        if all_accuracies:
            accuracy = np.array(all_accuracies).mean()
        else:
            accuracy = None

        return {
            "loss": all_losses.mean(),
            "loglike": -all_losses.mean(),
            "accuracy": accuracy,
        }
