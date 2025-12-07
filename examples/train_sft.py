from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import sparselora

sparselora.patch(target_density=0.08)  # 8% density = ~3.2x speedup

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# standard LoRA config but with very high r
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=1024,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
