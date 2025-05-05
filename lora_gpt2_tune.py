import os
import math
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt

# Seed for reproducibility
random.seed(42)

def load_and_split_corpus(path, split_ratio=0.1, seed=42):
    """Load and split text lines into train/validation."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(seed)
    random.shuffle(lines)
    n_val = max(1, int(len(lines) * split_ratio))
    return lines[n_val:], lines[:n_val]


def tokenize_dataset(lines, tokenizer, max_length=512):
    """Convert list of texts to tokenized HF Dataset."""
    ds = Dataset.from_dict({'text': lines})
    def fn(batch):
        return tokenizer(batch['text'], truncation=True, max_length=max_length)
    return ds.map(fn, batched=True, remove_columns=['text'])


def compute_perplexity(model, dataloader, device):
    """Compute average loss & perplexity on a dataloader."""
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            b = {k: v.to(device) for k, v in batch.items()}
            out = model(**b)
            loss = out.loss.item()
            tokens = b['input_ids'].numel()
            total_loss += loss * tokens
            total_tokens += tokens
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def select_top_k_modules(model, dataloader, device, top_k=4):
    """Rank modules by gradient norms and return top_k target module names."""
    model.to(device).train()
    batch = next(iter(dataloader))
    b = {k: v.to(device) for k, v in batch.items()}
    out = model(**b)
    out.loss.backward()
    scores = {}
    for name, param in model.named_parameters():
        if param.grad is not None and ('attn.c_attn.weight' in name or 'attn.c_proj.weight' in name):
            scores[name.replace('.weight','')] = param.grad.norm().item()
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [m for m, _ in top]


def main():
    # Configuration
    model_name = 'gpt2'
    corpus_path = 'data/micro_corpus.txt'
    out_dir = 'lora_gpt2_results'
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-4
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    max_length = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure corpus exists
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    if not os.path.exists(corpus_path) or os.path.getsize(corpus_path) == 0:
        sample = [
            'I wake before dawn, the hush of my apartment settling around me.',
            'Steam rises from the kettle while I watch the city lights fade.',
            'My thoughts drift to the pages of the notebook always at my side.',
            'Morning light filters through curtains, soft as a whispered promise.'
        ]
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(sample))

    # Load and split corpus
    train_lines, val_lines = load_and_split_corpus(corpus_path)

    # Tokenizer and datasets
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_ds = tokenize_dataset(train_lines, tokenizer, max_length)
    val_ds = tokenize_dataset(val_lines, tokenizer, max_length)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collator)

    # Baseline evaluation
    print('=== Baseline GPT-2 Evaluation ===')
    base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    bl_loss, bl_ppl = compute_perplexity(base_model, val_loader, device)
    print(f'Val Loss: {bl_loss:.4f}, Perplexity: {bl_ppl:.2f}')

    # Module ranking
    top_modules = select_top_k_modules(base_model, train_loader, device, top_k=4)
    print('Selected LoRA modules for adaptation:', top_modules)

    # Freeze base model for trainable count
    for param in base_model.parameters():
        param.requires_grad = False
    total_base = sum(param.numel() for param in base_model.parameters())
    trainable_base = sum(param.numel() for param in base_model.parameters() if param.requires_grad)
    print(f'Baseline Total Params: {total_base}, Trainable: {trainable_base}\n')

    # LoRA-adapted model setup
    print('=== LoRA-adapted GPT-2 Setup ===')
    lora_model = GPT2LMHeadModel.from_pretrained(model_name)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=top_modules
    )
    lora_model = get_peft_model(lora_model, peft_config).to(device)
    total_lora = sum(p.numel() for p in lora_model.parameters())
    trainable_lora = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f'LoRA Total Params: {total_lora}, Trainable: {trainable_lora}\n')

    # Fine-tuning loop
    optimizer = AdamW(lora_model.parameters(), lr=learning_rate)
    ppl_values = []
    epoch_list = list(range(1, num_epochs+1))
    for epoch in epoch_list:
        lora_model.train()
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = lora_model(**inputs)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss, ppl = compute_perplexity(lora_model, val_loader, device)
        ppl_values.append(ppl)
        print(f'Epoch {epoch}: Val Loss={loss:.4f}, Perplexity={ppl:.2f}')

    # Plot Perplexity curves
    os.makedirs(out_dir, exist_ok=True)
    baseline_curve = [bl_ppl] * num_epochs
    plt.plot(epoch_list, baseline_curve, label='Baseline')  # aligned x-axis
    plt.plot(epoch_list, ppl_values,  label='LoRA')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity over Epochs')
    plt.legend()
    plt.xlim(epoch_list[0], epoch_list[-1])  # ensure x-axis starts at epoch 1
    plt.savefig(os.path.join(out_dir, 'ppl_curve.png'))
    print('PPL curve saved to', os.path.join(out_dir, 'ppl_curve.png'), '\n')

    # Qualitative comparison for 5 prompts
    print('=== Qualitative Comparison ===')
    combined_lines = val_lines + train_lines
    random.shuffle(combined_lines)
    prompts = combined_lines[:5]
    for idx, prompt in enumerate(prompts, 1):
        print(f'--- Prompt {idx} ---')
        print(prompt)
        for name, model in [('Baseline', base_model), ('LoRA', lora_model)]:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            out_ids = model.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            print(f'{name}:', tokenizer.decode(out_ids[0], skip_special_tokens=True))
        print()

if __name__ == '__main__':
    main()