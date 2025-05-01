import os
import json
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------ Load Teacher Outputs Dataset ------------------ #
def load_teacher_outputs_dataset(json_path, tokenizer, max_length=512):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    completions = []
    for item in data:
        if "prompt" in item and "response" in item:
            prompts.append(item["prompt"])
            completions.append(item["response"])

    inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
    labels = tokenizer(completions, truncation=True, padding="max_length", max_length=max_length)["input_ids"]

    # Mask out pad tokens from loss
    masked_labels = []
    for ids in labels:
        masked_labels.append([-100 if tok_id == tokenizer.pad_token_id else tok_id for tok_id in ids])

    inputs["labels"] = masked_labels
    return Dataset.from_dict(inputs)

# ------------------ Main Function ------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--teacher_outputs_path", default="teacher_outputs.json")
    parser.add_argument("--output_dir", default="./distilled_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(args.student_model)
    dataset = load_teacher_outputs_dataset(args.teacher_outputs_path, tokenizer)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        report_to="none",
        fp16=args.fp16,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
