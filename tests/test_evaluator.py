import pytest
import numpy as np
from mttl.datamodule.base import AutoDataModule
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.ni_data_module import NiDataConfig
from mttl.evaluators import MMLUEvaluator
from mttl.evaluators import NIEvaluator
from transformers import AutoModelForCausalLM


def test_mmlu_eval():
    mmlu = MMLUEvaluator(
        MMLUDataConfig(
            "mmlu",
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=1024,
            train_batch_size=1,
            predict_batch_size=1,
            finetune_task_name="high_school_government_and_politics",
        ),
        device="cpu",
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    results = mmlu.evaluate(model, subsample=80)
    assert results["all"]["mean"] == 50
    assert results["all"]["stderr"] == 50


def test_ni_eval():
    import os

    if os.environ.get("NI_DATA_DIR") is None:
        return

    ni = NIEvaluator(
        NiDataConfig(
            "ni",
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=1024,
            train_batch_size=1,
            predict_batch_size=1,
            finetune_task_name="task893_gap_fill_the_blank_coreference_resolution",
        ),
        device="cpu",
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    results = ni.evaluate(model, subsample=80)
    assert results["all"]["mean"] == pytest.approx(1.98, 0.1)
