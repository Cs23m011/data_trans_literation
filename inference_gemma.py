import os
import pandas as pd
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/data/gemma-transliterator-finetuneFULL",
    max_seq_length = 512,
    dtype = torch.float16,          # or torch.bfloat16 if you used bf16
    full_finetuning=True
)

# Attach chat template again
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# Put model in eval mode
model.eval()

# Create a sample input for inference
def generate_transliteration(indic_word, language="Hindi"):
    prompt = [
        {"role": "system", "content": f"Transliterate {language} text to Roman script. Output only the transliterated word."},
        {"role": "user", "content": f"Transliterate: {indic_word}"},
        #{"role": "assistant", "content": ""}
    ]
    
    # Apply chat template (returns string), then tokenize
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    #print("🧪 Prompt Text:", repr(prompt_text))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.1,
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    #print("🧪 Raw Output Tokens:", outputs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print("🧪 Decoded Output:", repr(decoded))
    #return decoded
    response = decoded.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""
    #return response  # Clean up if needed

# 🔍 Test it
language_map = {
    'as': 'Assamese', 'bn': 'Bengali', 'gu': 'Gujarati', 'hi': 'Hindi', 'kn': 'Kannada',
    'ml': 'Malayalam', 'mr': 'Marathi', 'ne': 'Nepali', 'or': 'Odia', 'pa': 'Punjabi',
    'sa': 'Sanskrit', 'sd': 'Sindhi', 'ta': 'Tamil', 'te': 'Telugu', 'ur': 'Urdu'}
languages = ['as','bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa','ta', 'te', 'ur']
#languages=['bn']
for lang in languages:
    tsv_file=f"/data/akshantar/{lang}/{lang}.translit.sampled.test.tsv"
    df = pd.read_csv(tsv_file, sep="\t", header=None) #change
    unique_indic_words = df[0].unique()
    unique_indic_words = set(unique_indic_words)
    with open(f"/data/finetune_res1_gemmafull/{lang}.txt", "w", encoding="utf-8") as file:  # Use 'w' to overwrite or 'a' to append
        for w in unique_indic_words:
            out = (generate_transliteration(w,language_map[lang]))
            x=out.split()[-1]
            file.write(f"{w}\t{x}\n")
