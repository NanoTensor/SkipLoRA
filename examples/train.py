import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from skiptora import SkipLoRALayer, register_skip_hooks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="glue")
    parser.add_argument("--task", type=str, default="mrpc")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--r", type=int, default=8)
    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(args.dataset, args.task)
    def tokenize(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True)
    tokenized_dataset = dataset.map(tokenize, batched=True)

    # LoRA Config with SkipLoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.r,
        lora_alpha=2 * args.r,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],  # GPT-2 example
    )

    # Apply PEFT, but monkey-patch with SkipLoRA
    model = get_peft_model(model, peft_config)
    for name, module in model.named_modules():
        if "lora" in name.lower():
            module.__class__ = SkipLoRALayer  # Replace with SkipLoRA

    # Register hooks
    register_skip_hooks(model)

    # Training args
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        fp16=True,  # Complementary efficiency
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    trainer.train()
    model.save_pretrained("./skiplora_model")

if __name__ == "__main__":
    main()
