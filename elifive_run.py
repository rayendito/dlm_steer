import csv
from pathlib import Path

import torch
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer
from timpa_datasets import timpa_load_data_and_steer_artefacts
from timpa_experimental import visualize_timpa_probabilistic
from timpateks import timpa_probabilistic


def write_before_after_csv(html_file, text_before, text_after):
    if len(text_before) != len(text_after):
        raise ValueError(
            "text_before and text_after must have the same length."
        )

    csv_file = Path(html_file).with_suffix(".csv")
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text_before", "text_after"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "text_before": before,
                "text_after": after,
            }
            for before, after in zip(text_before, text_after)
        )
    return csv_file


#### MODELING
DEVICE = "cuda"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
config = LLaDAConfig.from_pretrained(
    MODEL_ID,
    local_files_only=True
)
model = LLaDAModelLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer.padding_side = "left"

### DATASET
elifive_data, elifive_artf = timpa_load_data_and_steer_artefacts(
    "elifive", "train", "timpa_probabilistic"
)

### Elifive call
ELIFIVE_REFILL_STEPS = 32
RANDOM_MASK_PROBABILITY = 0.5
RANDOM_SEED = 42

ELIFIVE_BASE_PROMPT = [elifive_artf["base"]] * len(elifive_data["text"])
TO_5YO_STEER = [elifive_artf["5yo"]] * len(elifive_data["text"])
TO_HIGHSCHOOL_STEER = [elifive_artf["highschool"]] * len(elifive_data["text"])
TO_PHD_STEER = [elifive_artf["phd"]] * len(elifive_data["text"])

tokenized_text, masking_probs, masked_positions, regenerated_texts = (
    timpa_probabilistic(
        model=model,
        tokenizer=tokenizer,
        identifier_model=None,
        identifier_tokenizer=None,
        steer=TO_5YO_STEER,
        text=elifive_data["text"],
        generator=torch.Generator(device=DEVICE).manual_seed(RANDOM_SEED),
        refill_steps=ELIFIVE_REFILL_STEPS,
        detection_strategy="random",
        random_mask_probability=RANDOM_MASK_PROBABILITY,
        base_assistant_prompt=ELIFIVE_BASE_PROMPT,
    )
)
html_file = visualize_timpa_probabilistic(
    tokenized_text,
    masking_probs,
    masked_positions,
    regenerated_texts,
    output_file="timpaprobs_elifive_to_5yo_random.html",
)
write_before_after_csv(
    html_file,
    elifive_data["text"],
    regenerated_texts,
)

tokenized_text, masking_probs, masked_positions, regenerated_texts = (
    timpa_probabilistic(
        model=model,
        tokenizer=tokenizer,
        identifier_model=None,
        identifier_tokenizer=None,
        steer=TO_HIGHSCHOOL_STEER,
        text=elifive_data["text"],
        generator=torch.Generator(device=DEVICE).manual_seed(RANDOM_SEED),
        refill_steps=ELIFIVE_REFILL_STEPS,
        detection_strategy="random",
        random_mask_probability=RANDOM_MASK_PROBABILITY,
        base_assistant_prompt=ELIFIVE_BASE_PROMPT,
    )
)
html_file = visualize_timpa_probabilistic(
    tokenized_text,
    masking_probs,
    masked_positions,
    regenerated_texts,
    output_file="timpaprobs_elifive_to_highschool_random.html",
)
write_before_after_csv(
    html_file,
    elifive_data["text"],
    regenerated_texts,
)

tokenized_text, masking_probs, masked_positions, regenerated_texts = (
    timpa_probabilistic(
        model=model,
        tokenizer=tokenizer,
        identifier_model=None,
        identifier_tokenizer=None,
        steer=TO_PHD_STEER,
        text=elifive_data["text"],
        generator=torch.Generator(device=DEVICE).manual_seed(RANDOM_SEED),
        refill_steps=ELIFIVE_REFILL_STEPS,
        detection_strategy="random",
        random_mask_probability=RANDOM_MASK_PROBABILITY,
        base_assistant_prompt=ELIFIVE_BASE_PROMPT,
    )
)
html_file = visualize_timpa_probabilistic(
    tokenized_text,
    masking_probs,
    masked_positions,
    regenerated_texts,
    output_file="timpaprobs_elifive_to_phd_random.html",
)
write_before_after_csv(
    html_file,
    elifive_data["text"],
    regenerated_texts,
)
