import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorWithPadding
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_data_formats,train_on_responses_only
from trl import SFTTrainer
print("GPUs available:", torch.cuda.device_count())

# ======================
# Environment setup
# ======================

# ======================
# Load model and tokenizer
# ======================
model_name = "unsloth/gemma-3-4b-it"
max_seq_length = 512
load_in_4bit = False
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    full_finetuning=True,
    token="your_huggingfacetoken",
)
#raw_tokenizer = tokenizer
model.config.use_cache = False
#model.config.text_config.attn_logit_softcapping = 0
#model.config.text_config.final_logit_softcapping = 0
#model.config.text_config.use_cache = False
#model.base_model.config.attn_logit_softcapping = 0
#model.base_model.config.final_logit_softcapping = 0
#model.base_model.config.use_cache = False
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"
tokenizer.model_max_length = max_seq_length

# ======================
# Load and prepare dataset
# ======================
languages = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'ta', 'te', 'ur']
language_map = {
    'as': 'Assamese', 'bn': 'Bengali', 'gu': 'Gujarati', 'hi': 'Hindi', 'kn': 'Kannada',
    'ml': 'Malayalam', 'mr': 'Marathi', 'ne': 'Nepali', 'or': 'Odia', 'pa': 'Punjabi',
    'sa': 'Sanskrit', 'sd': 'Sindhi', 'ta': 'Tamil', 'te': 'Telugu', 'ur': 'Urdu'
}

def create_dataset():
    formatted_data = []
    for lang in languages:
        tsv_file = f"/data/akshantar/{lang}/{lang}.translit.sampled.train.tsv"
        try:
            df = pd.read_csv(tsv_file, sep="\t", names=["indic", "roman"], usecols=[0, 1], encoding="utf-8", encoding_errors="replace")
            df=df.head(100)
            for _, row in df.iterrows():
                formatted_data.append({
                    "conversations": [
                        {"role": "system", "content": str(f"Transliterate {language_map[lang]} text to Roman script. Output only the transliterated word.")},
                        {"role": "user", "content": str(f"Transliterate: {row['indic']}")},
                        {"role": "assistant", "content": str(row["roman"])}
                    ]
                })
        except FileNotFoundError:
            continue
    return Dataset.from_list(formatted_data)

dataset = create_dataset()

# ======================
# Apply chat template
# ======================
dataset = standardize_data_formats(dataset)

def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return {"text": texts}

dataset = dataset.map(apply_chat_template, batched=True)

# ======================
# Tokenize dataset
# ======================
'''def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_seq_length,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])'''

# ======================
# Training configuration
# ======================
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=50,
    num_train_epochs=2,
    learning_rate=2e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_torch",
    logging_steps=20,
    save_strategy="steps",
    save_steps=50000,
    evaluation_strategy="no",
    output_dir="./gemma-transliterator-finetune_1B",
    seed=3407,
    report_to="none",
)

# ======================
# Data collator
# ======================
#data_collator = DataCollatorWithPadding(tokenizer=raw_tokenizer, padding=True)

# ======================
# Trainer
# ======================
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    args=training_args,
    packing=False,
)
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

print(f"🚀 Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"🔋 Allocated Memory Before Training: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")

# ======================
# Train
# ======================
trainer.train()

# ======================
# Save model & tokenizer
# ======================
model.save_pretrained("./gemma-transliterator-finetune_1B")
tokenizer.save_pretrained("./gemma-transliterator-finetune_1B")
print("✅ Fine-tuning complete! Model and tokenizer saved.")
