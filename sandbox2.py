import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from timpa_experimental import visualize_timpa, visualize_token_identification
from timpateks.llada.modeling_llada import LLaDAModelLM
from timpateks.llada.configuration_llada import LLaDAConfig

device = "cuda"
TEXT = [
    "The human brain works through billions of neurons that communicate using electrical impulses and chemical neurotransmitters across synapses. Different regions handle different functions: the cerebral cortex supports thinking, memory, language, and decision-making; the cerebellum coordinates movement and balance; and the brainstem controls automatic processes like breathing and heart rate. Signals travel through neural networks, where synaptic plasticity allows the brain to learn, adapt, and store information over time.",
    "The human brain works through billions of neurons that communicate using electrical impulses and chemical neurotransmitters across synapses. Different regions handle different functions: the cerebral cortex supports thinking, memory, language, and decision-making; the cerebellum coordinates movement and balance; and the brainstem controls automatic processes like breathing and heart rate. Signals travel through neural networks, where synaptic plasticity allows the brain to learn, adapt, and store information over time.",
    "The human brain works through billions of neurons that communicate using electrical impulses and chemical neurotransmitters across synapses. Different regions handle different functions: the cerebral cortex supports thinking, memory, language, and decision-making; the cerebellum coordinates movement and balance; and the brainstem controls automatic processes like breathing and heart rate. Signals travel through neural networks, where synaptic plasticity allows the brain to learn, adapt, and store information over time.",
]
STEER_PROMPTS = [
    "You're explaining to a 5 year old who knows nothing about biology:\n",
    "You're explaining to a highschool student who might know a little bit of biology:\n",
    "You're explaining to a medical professional:\n",
]

############################## MODELING
MODEL = "GSAI-ML/LLaDA-8B-Instruct"
config = LLaDAConfig.from_pretrained(MODEL)
model = LLaDAModelLM.from_pretrained(
    MODEL,
    config=config,
    torch_dtype=torch.bfloat16,
).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
)
tokenizer.padding_side = "left"

IDENTIFIER_MODEL = "Qwen/Qwen2.5-14B-Instruct"
identifier_model = AutoModelForCausalLM.from_pretrained(IDENTIFIER_MODEL)
identifier_tokenizer = AutoTokenizer.from_pretrained(IDENTIFIER_MODEL)

############################## TEST RUN

############################## VIZZ
visualize_timpa(
    model,
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    STEER_PROMPTS,
    TEXT,
    base_assistant_prompt="You are explaining to a neurosurgeon",
    temperature=0.25,
    margin=0
)