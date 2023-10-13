from dataclasses import dataclass
import numpy as np
import os
from datasets import load_dataset
import tqdm

from mttl.dataloader.platypus_dataset_reader import (
    InversePlatypusTemplate,
    PlatypusTemplate,
)
from mttl.utils import logger

from src.data_transforms.base import (
    TransformConfig,
    DataTransformTemplate,
    TransformModel,
)
from src.data_transforms.engines import (
    OpenAI,
    AutoEngine,
    free_memory,
)
from src import mmlu_subject_configs
from src.data_transforms.utils import (
    INVALID_RESPONSE,
    read_jsonl_dataset,
    dump_jsonl_dataset,
    upload_to_hf_,
    reject_output,
    count_repeating_sentences,
)


class QAPlatyInstructionGenerationTemplate(DataTransformTemplate):
    @classmethod
    def apply(cls, output, context=None, icl_examples=None):
        """
        We provide the context as output (response) if and only if a response is not
        present. Otherwise, we provide the context as context in addition to the previously generated response.
        """
        return InversePlatypusTemplate.apply(
            output if output is not None else context,
            input if output is not None else None,
            icl_examples,
        )

    @classmethod
    def post_process_generation(cls, output):
        return {"instruction": output}


class QAPlatyResponseGenerationTemplate(PlatypusTemplate):
    @classmethod
    def apply(cls, instruction, context=None):
        return PlatypusTemplate.apply(instruction, input=context)

    @classmethod
    def post_process_generation(cls, output):
        return {"response": output}


class OAITemplate:
    @classmethod
    def post_process_generation(cls, output):
        try:
            if "Response:" in output:
                response = output.split("Response:")[1].strip()
            else:
                raise

            instruction = output.split("Response:")[0]
            if "Instruction:" in instruction:
                instruction = instruction.split("Instruction:")[1].strip()
                data = {
                    "instruction": instruction.replace("#", ""),
                    "response": response.replace("#", ""),
                }
            else:
                raise
        except:
            data = {"instruction": INVALID_RESPONSE, "response": INVALID_RESPONSE}
        return data

    @classmethod
    def apply(cls, output, input=None, icl_examples=None):
        task_description = (
            "\nYour task is to generate clear, comprehensive, and precise instructions."
        )
        if icl_examples is not None:
            task_description += f"\n\n Here are examples of good instructions that you should imitate:\n"
            for icl_example in icl_examples:
                task_description += f"\n### Instruction:\n{icl_example}"
            task_description += "\n\n"

        task_description += "\n\nDomain context:"
        task_description += f"\n\n{output}"

        task_description += (
            "\nWrite an instruction suitable for the given context. "
            "Ensure it's complete, precise, and stands alone, without relying on provided context."
        )

        if icl_examples is not None:
            task_description += (
                " Strive to match the style, tone, and length of the previous examples."
            )

        task_description += "\nRemember to also provide a concise response to the generated instruction.\
            Format your ourput as follows: ### Instruction: <your instruction> ### Response: <your response>."
        return task_description


@dataclass
class QAModelSetting:
    inverse_model_path: str
    model_path: str
    instruction_template: str
    response_template: str

    @property
    def model_paths(self):
        return self.inverse_model_path, self.model_path


QA_MODEL_SETTINGS = {
    "platy": QAModelSetting(
        inverse_model_path="sordonia/llama2-13b-platypus-inverse",
        model_path="sordonia/llama2-13b-platypus",
        instruction_template=QAPlatyInstructionGenerationTemplate(),
        response_template=QAPlatyResponseGenerationTemplate(),
    )
}


@dataclass
class QATransformConfig(TransformConfig):
    model_setting: str
    max_context_length: int = 512
    max_tokens_instruction: int = 128
    max_tokens_response: int = 1024
    top_p: float = 0.9
    num_iterations: int = 1
    temperature: float = 0.7
    max_documents_per_subject: float = -1
    max_contexts_per_subject: float = -1
    icl_examples: int = 0
    icl_dataset: str = "lukaemon/mmlu"
    icl_split: str = "validation"
    icl_use_options: str = True


class MMLUICLSampler:
    def __init__(self, dataset="lukaemon/mmlu", split="validation", use_options=True):
        from datasets import get_dataset_config_names

        subject_names = get_dataset_config_names(dataset)
        logger.info("Creating MMLU ICL Sampler...")

        self.dataset = {}
        for subject in subject_names:
            self.dataset[subject] = load_dataset(dataset, subject, split=split)
        self.use_options = use_options

    def sample(self, num_examples, subject):
        dataset = self.dataset[subject].shuffle()
        examples = []
        for i in range(num_examples):
            example = dataset[i]["input"]
            if self.use_options:
                for ans_option in ["A", "B", "C", "D"]:
                    option = f"\n{ans_option}: " + dataset[i][ans_option]
                    example += option
            examples.append(example)
        return examples


class QATransformModel(TransformModel):
    """Transform a dataset of documents into a question answering dataset."""

    def __init__(
        self,
        config: QATransformConfig,
    ):
        self.config = config
        self.model_path = QA_MODEL_SETTINGS[self.config.model_setting].model_path
        self.inverse_model_path = QA_MODEL_SETTINGS[
            self.config.model_setting
        ].inverse_model_path
        self.instruction_template = QA_MODEL_SETTINGS[
            self.config.model_setting
        ].instruction_template
        self.response_template = QA_MODEL_SETTINGS[
            self.config.model_setting
        ].response_template
        self._llm = None

    def get_dataset_name(self, iter_signature=""):
        args = [
            f"qa-{self.config.model_setting}",
            f"icl{self.config.icl_examples}",
            f"clen{self.config.max_context_length}",
            f"maxD{self.config.max_documents_per_subject}",
            f"maxC{self.config.max_contexts_per_subject}",
            f"{iter_signature}.jsonl",
        ]
        return "_".join(args)

    def _load_llm(self, from_path, **options):
        del self._llm
        free_memory()
        self._llm = AutoEngine.from_path(from_path, **options)
        return self._llm

    def transform(
        self,
        dataset_name,
        filter_subjects,
        upload_to_hub=False,
        output_path="./generated.jsonl",
    ):
        output_path = os.environ.get("AMLT_OUTPUT_DIR", output_path)
        if upload_to_hub:
            assert (
                os.environ.get("HF_TOKEN") is not None
            ), "Please set HF_TOKEN env variable."

        # start dataset
        prev_dataset = self.get_seed_dataset_(dataset_name, filter_subjects)

        for i in range(self.config.num_iterations):
            inst_filename = os.path.join(
                output_path, self.get_dataset_name("inst_%d" % i)
            )
            answ_filename = os.path.join(output_path, self.get_dataset_name("%d" % i))

            self._load_llm(self.model_path)
            is_openai = type(self._llm) == OpenAI

            if not os.path.exists(inst_filename):
                instruction_dataset = self.generate_instructions_(
                    self._llm,
                    prev_dataset,
                )
                dump_jsonl_dataset(
                    instruction_dataset, answ_filename if is_openai else inst_filename
                )
            else:
                instruction_dataset = read_jsonl_dataset(inst_filename)

            # we generate only once w openai
            if is_openai:
                break

            self._load_llm(self.inverse_model_path)
            if not os.path.exists(answ_filename):
                answer_dataset = self.generate_answers_(
                    self._llm,
                    instruction_dataset,
                )
                dump_jsonl_dataset(answer_dataset, answ_filename)
            else:
                answer_dataset = read_jsonl_dataset(answ_filename)
            prev_dataset = answer_dataset

        if upload_to_hub:
            print("Uploading the final dataset to HuggingFace Hub...")
            upload_to_hf_(answ_filename, configuration=self.config)

    def generate_instructions_(
        self,
        llm: AutoEngine,
        dataset,
    ):
        """
        To generate instructions, we take num_contexts_per_document chunks of length max_context_length from each document,
        then sample 1 instruction from each chunk.

        All instructions are appended as jsonl into output_filename.
        """
        import copy

        def get_templated_context(entry):
            return self.instruction_template.apply(
                context=entry["context"],
                output=entry["response"] if "response" in entry else None,
                icl_examples=entry["icl_examples"],
            )

        templated_contexts = [get_templated_context(entry) for entry in dataset]

        print("Example generation requests...")
        for context in np.random.choice(templated_contexts, 5):
            print(context)
            print()

        result = llm.generate(
            templated_contexts,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_instruction,
        )
        assert (
            len(result.outputs) == len(templated_contexts) == len(result.finish_reason)
        )

        new_dataset = []
        for entry, instruction, finish_reason in zip(
            dataset, result.outputs, result.finish_reason
        ):
            data = self.instruction_template.post_process_generation(instruction)
            if reject_output(data["instruction"], finish_reason):
                continue

            copied_entry = copy.deepcopy(entry)
            copied_entry.update(
                {
                    "author_instr": str(llm.model_name),
                }
            )
            copied_entry.update(data)
            new_dataset.append(copied_entry)

        print("Created a new instruction dataset of size:", len(new_dataset))
        return new_dataset

    def generate_answers_(
        self,
        llm: AutoEngine,
        dataset,
    ):
        requests = []
        for instance in tqdm.tqdm(dataset):
            requests.append(
                self.response_template.apply(
                    instance["instruction"],
                    instance["context"],
                )
            )

        result = llm.generate(
            requests,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_response,
        )
        assert len(result.outputs) == len(requests)

        new_dataset = []
        for entry, response, log_p, reason in zip(
            dataset, result.outputs, result.cumulative_logprobs, result.finish_reason
        ):
            import copy

            data = self.response_template.post_process_generation(response)
            if reject_output(data["response"], reason):
                continue

            # lets also avoid outputs that contains repetitions of the same sentence more than once
            n_rep = count_repeating_sentences(response)
            if n_rep > 0:
                continue

            entry = copy.deepcopy(entry)
            entry.update(
                {
                    "response": data["response"],
                    "author_response": str(llm.model_name),
                    "normalized_cumul_logprob_response": log_p,
                }
            )
            new_dataset.append(entry)

        print("Created a new answer dataset of size:", len(new_dataset))
        return new_dataset

    def get_seed_dataset_(self, dataset_name, filter_subjects, **options):
        """
        Convert a seed dataset of retrieved content into a tuple of (context, subject, icl_examples).
        """
        dataset = load_dataset(dataset_name)["train"].to_pandas()
        converted_dataset = []

        if type(filter_subjects) == str:
            filter_subject = getattr(mmlu_subject_configs, filter_subjects)

        for subject in filter_subject:
            subject_data = dataset[dataset["subject"] == subject]
            subject_data.sort_values(by="dfq", ascending=False, inplace=True)

            subject_contexts = []
            num_contexts_per_doc = [0]

            for i in tqdm.tqdm(
                range(len(subject_data)), desc=f"Processing {subject}..."
            ):
                document = subject_data.iloc[i]
                text = document["text"]

                sentences = text.split(".")
                sentences = [
                    sentence.strip().replace("\n", " ").replace("  ", " ")
                    for sentence in sentences
                    if len(sentence.strip()) > 0
                ]

                # new document
                document_contexts = []
                for sentence in sentences:
                    sentence = sentence + "."
                    if not document_contexts:
                        document_contexts.append(sentence)
                    else:
                        if (
                            len(document_contexts[-1].split()) + len(sentence.split())
                            < self.config.max_context_length
                        ):
                            document_contexts[-1] += " " + sentence
                        else:
                            document_contexts.append(sentence)

                num_contexts_per_doc.append(len(document_contexts))
                subject_contexts.extend(
                    {
                        "text": context,
                        "docno": str(document["docno"]),
                    }
                    for context in document_contexts
                )

                if (
                    self.config.max_contexts_per_subject > 0
                    and len(subject_contexts) > self.config.max_contexts_per_subject
                ):
                    print(
                        "Breaking early due to max_contexts_per_subject settings. ",
                        len(subject_contexts),
                    )
                    break

                if (
                    self.config.max_documents_per_subject > 0
                    and i > self.config.max_documents_per_subject
                ):
                    print(
                        "Breaking early due to max_documents_per_subject settings. ",
                        len(subject_contexts),
                    )
                    break

            print(
                "Contexts per document (Avg/Min/Max):",
                np.mean(num_contexts_per_doc),
                np.min(num_contexts_per_doc),
                np.max(num_contexts_per_doc),
            )

            for context in subject_contexts:
                converted_dataset.append(
                    {
                        "id": str(len(converted_dataset)),
                        "context": context["text"],
                        "docno": str(context["docno"]),
                        "subject": subject,
                        "icl_examples": self.icl_sampler.sample(
                            self.config.icl_examples, subject
                        )
                        if self.config.icl_examples > 0
                        else None,
                    }
                )
        return converted_dataset
