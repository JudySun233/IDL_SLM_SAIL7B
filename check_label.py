import json
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from finetune import IGNORE_TOKEN_ID, preprocess  # import your preprocess()

def main():
    # 1) Load a few examples
    data_path  = "datasets/SAIL_train_deberta_entailment.json"
    raw        = json.load(open(data_path, "r"))
    sample_raw = raw[:5]

    # 2) Init tokenizer
    tokenizer         = AutoTokenizer.from_pretrained("luohy/SAIL-7b", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # 3) Run preprocess
    data = preprocess([ex["conversations"] for ex in sample_raw], tokenizer)
    input_ids      = data["input_ids"]           # (N, L)
    attention_mask = data.get("attention_mask")  # if you return it
    labels         = data.get("labels", None)    # may be None

    # 4) If labels is None, use HF collator to build them
    if labels is None:
        collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=None, return_tensors="pt"
        )
        # build a mini‐batch dicts
        dicts = [
            {"input_ids":      input_ids[i],
             "attention_mask": attention_mask[i]}
            for i in range(len(sample_raw))
        ]
        batch = collator(dicts)
        labels = batch["labels"]  # now we have (N, L)

    # 5) Print valid‐label stats
    for i in range(len(sample_raw)):
        lab = labels[i].tolist()
        valid = sum(1 for x in lab if x != IGNORE_TOKEN_ID)
        total = len(lab)
        print(f"Example {i}: {valid}/{total} valid labels "
              f"({100*valid/total:.1f}% non-ignored)")
        print("  input_ids:",      input_ids[i][:20].tolist(), "…")
        print("  labels:   ",      lab[:20], "…\n")

    # 6) (Optional) quick forward to inspect a nonzero loss
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("luohy/SAIL-7b", torch_dtype=torch.float16)
    model.config.use_cache = False
    batch = {
      "input_ids":      input_ids,
      "attention_mask": attention_mask,
      "labels":         labels
    }
    out = model(**{k: v.to(model.device) for k,v in batch.items()})
    print("Batch loss:", out.loss.item())

if __name__ == "__main__":
    main()
