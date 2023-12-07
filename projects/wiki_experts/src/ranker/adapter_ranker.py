import torch

from abc import ABC, abstractmethod

from projects.wiki_experts.src.ranker.baseline_rankers import KATERanker
from projects.wiki_experts.src.ranker.classifier_ranker import (
    SentenceTransformerClassifier,
)
from projects.wiki_experts.src.ranker.clip_ranker import CLIPRanker


class AdapterRanker(ABC):
    @abstractmethod
    def predict_batch(self, batch, n=1):
        """Predicts the top n tasks for each input in the batch."""
        pass

    @abstractmethod
    def predict_task(self, query, n=1):
        """Predicts the top n tasks for the input query."""
        pass


class AdapterRankerHelper:
    @staticmethod
    def get_ranker_instance(ranker_model, ranker_path, device="cuda"):
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"

        if ranker_model == "clip":
            model = CLIPRanker.from_pretrained(ranker_path).to(device)
            return model
        elif ranker_model == "classifier":
            model = SentenceTransformerClassifier.from_pretrained(ranker_path).to(
                device
            )
            return model
        elif ranker_model == "kate":
            model = KATERanker.from_pretrained(ranker_path)
            return model
        else:
            raise ValueError(f"Unknown retrieval model: {ranker_model}")
