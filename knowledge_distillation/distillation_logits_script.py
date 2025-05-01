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
import torch
from torch.nn import functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_teacher_outputs_dataset(json_path, tokenizer, max_length=512):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts, completions = [], []
    for item in data:
        if "prompt" in item and "response" in item:
            prompts.append(item["prompt"])
            completions.append(item["response"])

    tokenized_inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
    tokenized_outputs = tokenizer(completions, truncation=True, padding="max_length", max_length=max_length)["input_ids"]

    masked_labels = [[-100 if tok == tokenizer.pad_token_id else tok for tok in seq] for seq in tokenized_outputs]
    tokenized_inputs["labels"] = masked_labels
    tokenized_inputs["raw_prompts"] = prompts
    return Dataset.from_dict(tokenized_inputs)

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, teacher_tokenizer=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        student_input_ids = inputs["input_ids"]
        student_labels = inputs["labels"]
        raw_prompts = inputs["raw_prompts"]

        student_outputs = model(input_ids=student_input_ids, attention_mask=inputs["attention_mask"])
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_inputs = self.teacher_tokenizer(raw_prompts, return_tensors="pt", padding=True, truncation=True, max_length=student_input_ids.shape[1]).to(model.device)
            teacher_logits = self.teacher_model(**teacher_inputs).logits

            teacher_preds = torch.argmax(teacher_logits, dim=-1)
            teacher_decoded = self.teacher_tokenizer.batch_decode(teacher_preds, skip_special_tokens=True)
            reencoded = self.tokenizer(teacher_decoded, return_tensors="pt", padding=True, truncation=True, max_length=student_input_ids.shape[1]).to(model.device)
            mapped_teacher_logits = F.log_softmax(model(**reencoded).logits / self.temperature, dim=-1)

        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, mapped_teacher_logits, reduction="batchmean", log_target=True)
        ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), student_labels.view(-1), ignore_index=-100)

        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        return (loss, student_outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--teacher_model", default="luohy/SAIL-7b")
    parser.add_argument("--teacher_outputs_path", default="teacher_outputs.json")
    parser.add_argument("--output_dir", default="./distilled_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    student_model = AutoModelForCausalLM.from_pretrained(args.student_model)

    dataset = load_teacher_outputs_dataset(args.teacher_outputs_path, student_tokenizer)
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

    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        temperature=2.0,
        alpha=0.5,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
