import config
import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on medical transcription summaries (mtsamples).")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Pretrained model name or path (default: gpt2)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size per device (default: 2)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size for testing/quick training")
    parser.add_argument("--output_dir", type=str, default=str(config.LOCAL_SUMMARIZER_DIR), help="Output directory for saved model")
    return parser.parse_args()

class CausalLMDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for f in features:
            padding_length = max_len - len(f["input_ids"])
            # Pad input_ids and attention_mask using pad_token_id and 0
            padded_input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * padding_length)
            padded_attention_mask.append(f["attention_mask"] + [0] * padding_length)
            # Pad labels using -100 so they are ignored by CrossEntropyLoss
            padded_labels.append(f["labels"] + [-100] * padding_length)
            
        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
            "labels": torch.tensor(padded_labels)
        }

def train():
    args = parse_args()
    print("==================================================")
    print(f"Starting Fine-Tuning of {args.base_model} on mtsamples")
    print("==================================================")
    
    # 1. Load Hugging Face dataset
    print("\nStep 1: Downloading/Loading harishnair04/mtsamples from Hugging Face...")
    try:
        # Load from Hugging Face Hub (verify=False handles custom SSL environments if needed, datasets respects config/env settings)
        dataset = load_dataset("harishnair04/mtsamples", split="train")
        print(f"Dataset loaded. Total records: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face Hub: {e}")
        print("Attempting to load from local cached file if available...")
        raise e
        
    # Limit dataset if requested (great for dry runs/testing)
    if args.limit is not None:
        print(f"Limiting dataset to the first {args.limit} samples.")
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Filter out empty records
    dataset = dataset.filter(lambda x: x["transcription"] and x["description"])
    print(f"Records after filtering empty rows: {len(dataset)}")

    # Split dataset into train and validation (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 2. Load tokenizer and model
    print(f"\nStep 2: Initializing tokenizer and model for {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Configure padding token if it doesn't exist (e.g. for GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # 3. Tokenize dataset
    print("\nStep 3: Tokenizing medical notes and summaries...")
    
    def tokenize_function(examples):
        prompts = [
            f"System: You are a clinical summarization assistant. Summarize the following medical transcription.\n"
            f"Transcription: {t.strip()}\n"
            f"Summary:"
            for t in examples["transcription"]
        ]
        # Include target summary and EOS token
        targets = [f" {d.strip()}{tokenizer.eos_token}" for d in examples["description"]]
        
        # Tokenize separately to construct inputs and target labels
        prompt_inputs = tokenizer(prompts, truncation=True, max_length=400)
        target_inputs = tokenizer(targets, truncation=True, max_length=112)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for p_ids, t_ids in zip(prompt_inputs["input_ids"], target_inputs["input_ids"]):
            combined_ids = p_ids + t_ids
            # Set labels: prompt tokens get -100 (ignored in loss), target tokens get actual IDs
            label_ids = [-100] * len(p_ids) + t_ids
            
            input_ids.append(combined_ids)
            attention_mask.append([1] * len(combined_ids))
            labels.append(label_ids)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
    train_tokenized = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training dataset"
    )
    val_tokenized = val_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset"
    )

    # 4. Set up Trainer
    print("\nStep 4: Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./tmp_summarizer_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        disable_tqdm=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=CausalLMDataCollator(tokenizer)
    )

    # 5. Run training
    print("\nStep 5: Executing model training (Fine-Tuning)...")
    trainer.train()

    # 6. Save model and tokenizer
    print(f"\nStep 6: Saving fine-tuned model and tokenizer to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed successfully! Local model is ready for usage.")
    print("==================================================")

if __name__ == "__main__":
    train()
