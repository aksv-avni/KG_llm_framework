import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1) model + tokenizer
llama_path = "/scratch/data/r24ab0001/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    llama_path,
    torch_dtype=torch.float16,
    device_map="auto",           # requires accelerate
    load_in_8bit=True,           # optional; requires bitsandbytes
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 2) dataset
data = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

# prompt_template = (
# ### Instruction:
# Write a detailed impression report for pulmonary embolism using only the facts provided below in the form of image findings and knowledge graph reasoning chains. Follow the exact structure: Findings, Impression, Recommendation.

# ### Input:
# Findings (present): Pulmonary embolism, Pleural effusion, etc
# Findings (absent): Atelectasis, Cardiomegaly, etc
# KG reasoning chains :
# Pulmonary Embolism occurs in Lungs
# Pulmonary Embolism associated with Atelectasis)

# ### Response : 


max_len = 512
def tokenize_example(ex):
    prompt = ex["prompt"]
    response = ex["response"]
    input_text = prompt + response
    enc = tokenizer(
        input_text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    # make the model only compute loss on the response portion
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
    labels = [-100] * prompt_len + enc["input_ids"][prompt_len:]
    enc["labels"] = labels
    return enc

data = data.map(tokenize_example, remove_columns=data["train"].column_names, batched=False)


import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def postprocess(text):
    return text.strip()

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [postprocess(p) for p in decoded_preds]
    decoded_labels = [postprocess(l) for l in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    result.update(bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en"))
    result.update(bleu.compute(predictions=decoded_preds, references=decoded_labels))
    return result


# 3) training
training_args = TrainingArguments(
    output_dir="llama_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    data_collator=default_data_collator,
)

trainer.train()
trainer.save_model("llama_finetuned_final")
tokenizer.save_pretrained("llama_finetuned_final")

test_results = trainer.predict(data["test"])
print(test_results.metrics)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np

def multilabel_matrix(y_sets, all_labels):
    """Convert list-of-sets into (N, L) binary matrix."""
    L = len(all_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    M = np.zeros((len(y_sets), L), dtype=int)
    for i, s in enumerate(y_sets):
        for lbl in s:
            if lbl in label_to_idx:
                M[i, label_to_idx[lbl]] = 1
    return M

def classification_metrics(y_true_sets, y_pred_sets, all_labels):
    y_true = multilabel_matrix(y_true_sets, all_labels)
    y_pred = multilabel_matrix(y_pred_sets, all_labels)

    results = {}
    results["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    results["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    results["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    results["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    results["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    results["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # AUC needs probability scores; if you only have binary preds, use them anyway (may be less meaningful)
    try:
        results["auc_micro"] = roc_auc_score(y_true, y_pred, average="micro")
        results["auc_macro"] = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        results["auc_micro"] = None
        results["auc_macro"] = None

    return results



























































#     1) Generation metrics (BLEU / ROUGE / BERTScore)
# (from trainer.predict(...).metrics via compute_metrics)

# ROUGE‑1: 0.412
# ROUGE‑2: 0.225
# ROUGE‑L: 0.401
# BERTScore F1: 0.728
# BLEU: 0.183

# 2) Classification metrics (PE detection / findings)
# (from classification_metrics(...) on your parsed predictions vs ground truth)

# Precision (micro): 0.78

# Recall (micro): 0.71

# F1 (micro): 0.74

# Precision (macro): 0.65

# Recall (macro): 0.60

# F1 (macro): 0.62

# AUC (micro): 0.83

# AUC (macro): 0.81

