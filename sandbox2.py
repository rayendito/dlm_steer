import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from timpa_experimental import visualize_timpa, visualize_token_identification
from timpateks.llada.modeling_llada import LLaDAModelLM
from timpateks.llada.configuration_llada import LLaDAConfig

device = "cuda"
TEXT = [
    "My fellow citizens, the MBG is not merely about providing food to our children; it is a great investment in the future of our nation. We want every Indonesian child to grow healthy, strong, intelligent, and ready to compete toward Golden Indonesia 2045. The state must never allow its children to study while hungry. That is why this program is a concrete expression of the state’s presence: ensuring better nutrition, supporting families, strengthening farmers and village economies, and giving our young generation a strong foundation to build a more advanced and dignified Indonesia.",
]
STEER_PROMPTS = [
    "Pretend you're Kim Kardashian:\n",
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

IDENTIFIER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
identifier_model = AutoModelForCausalLM.from_pretrained(IDENTIFIER_MODEL)
identifier_tokenizer = AutoTokenizer.from_pretrained(IDENTIFIER_MODEL)

############################## TEST RUN
visualize_timpa(
    model,
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    STEER_PROMPTS,
    TEXT,
    base_assistant_prompt="Pretend you're Prabowo Subianto",
    temperature=0.05,
)

############################## VIZZ
# visualize_token_identification(
#     model, tokenizer, "AR", STEER_PROMPTS, TEXT,
# )
