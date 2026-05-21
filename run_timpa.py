import os
import random

import torch
from tqdm import tqdm
from utils.args_utils import parse_args
from utils.data_utils import load_timpa_dataset
from utils.steer_utils import l2_normalize
from utils.eval_utils import rearrange_results

from transformers import AutoModelForCausalLM, AutoTokenizer
from llada.modeling_llada import LLaDAModelLM
from llada.configuration_llada import LLaDAConfig
from llada.generate import resteer_v2

DEVICE = "cuda"
MAIN_MODEL = "GSAI-ML/LLaDA-8B-Base"
EVAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# MAIN MODEL
main_config = LLaDAConfig.from_pretrained(MAIN_MODEL)
main_model = LLaDAModelLM.from_pretrained(MAIN_MODEL, config=main_config, torch_dtype=torch.bfloat16).to(DEVICE).eval()
main_tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL, trust_remote_code=True)
main_tokenizer.padding_side = "left"

# PARSING ARGS
args = parse_args()

def set_random_state(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def experiment_name(args):
    prefix = f"{args.random_state}_"
    return args.run_name if args.run_name.startswith(prefix) else f"{prefix}{args.run_name}"

def main() -> None:
    set_random_state(args.random_state)

    # DATA
    data = load_timpa_dataset(args.dataset_path)
    concepts = list(data.keys())
    # trimming data to siz only (debugging purposes)
    siz = 10
    data[concepts[0]] = data[concepts[0]][:siz]
    data[concepts[1]] = data[concepts[1]][:siz]

    # STEER VECTORS
    concept_vectors = torch.load(args.steer_vector_path, map_location=DEVICE)
    c1_vectors = concept_vectors[concepts[0]]
    c2_vectors = concept_vectors[concepts[1]]
    # defaults sentiment defaults first concept (concepts[0]) 
    steer_vectors_all = tuple(
        (l2_normalize(c1_vectors[i]) - l2_normalize(c2_vectors[i])) * args.steer_alpha
        for i in range(len(c2_vectors))
    )
    steer_vectors = {si: steer_vectors_all[si] for si in args.steer_layers}
    counter_steer_vectors = {si: -steer_vectors_all[si] for si in args.steer_layers}
    
    results_dir = f"results/{experiment_name(args)}"
    os.makedirs(results_dir, exist_ok=True)
    for conc, ster in tqdm(zip(concepts, [counter_steer_vectors, steer_vectors])):
        for rs in tqdm(args.refill_steps):
            for st in tqdm(args.sampling_temp):
                for it in tqdm(args.identify_temp):
                    filesave_name = f"from_{conc}-rs{rs}-st{st}-it{it}"
                    results = run_steer_on_dataset(
                        data[conc], ster,
                        refill_steps=rs, samp_temp=st, iden_temp=it
                    )
                    torch.save(results, f"{results_dir}/{filesave_name}.pt")

def run_steer_on_dataset(data, steer_vectors, refill_steps, samp_temp, iden_temp):
    all_results = []
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i:i + args.batch_size]
        tokenized_inputs = main_tokenizer(batch,add_special_tokens=False,padding=True,return_tensors="pt").to(DEVICE)
        steered_x = resteer_v2(
            main_model,
            tokenized_inputs,
            steer_vectors,
            resteer_steps = args.resteer_steps,
            refill_steps = refill_steps,
            sampling_temp = samp_temp,
            identify_temp = iden_temp,
            alpha_decay=False
        )
        steered_x_T = rearrange_results(steered_x)
        all_results += steered_x_T
    return all_results

if __name__ == "__main__":
    main()
