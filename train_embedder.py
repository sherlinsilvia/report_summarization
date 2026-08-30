import config
import os
import argparse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer embedder on medical texts.")
    parser.add_argument("--base_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer base model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size for testing/quick training")
    parser.add_argument("--output_dir", type=str, default=str(config.LOCAL_EMBEDDER_DIR), help="Output directory for saved embedder model")
    return parser.parse_args()

def train():
    args = parse_args()
    print("==================================================")
    print(f"Starting Fine-Tuning of {args.base_model} on mtsamples")
    print("==================================================")
    
    # 1. Load dataset
    print("\nStep 1: Downloading/Loading harishnair04/mtsamples from Hugging Face...")
    try:
        dataset = load_dataset("harishnair04/mtsamples", split="train")
        print(f"Dataset loaded. Total records: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face Hub: {e}")
        raise e
        
    # Limit dataset if requested (great for dry runs/testing)
    if args.limit is not None:
        print(f"Limiting dataset to the first {args.limit} samples.")
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Filter out empty records
    dataset = dataset.filter(lambda x: x["transcription"] and x["description"])
    print(f"Records after filtering empty rows: {len(dataset)}")

    # 2. Initialize SentenceTransformer
    print(f"\nStep 2: Initializing SentenceTransformer model {args.base_model}...")
    model = SentenceTransformer(args.base_model)

    # 3. Create training dataset of InputExamples
    print("\nStep 3: Preparing query-document training pairs (MultipleNegativesRankingLoss)...")
    train_examples = []
    for trans, desc in zip(dataset["transcription"], dataset["description"]):
        # MNRL maps query-like description and document-like transcription close to each other
        train_examples.append(InputExample(texts=[desc.strip(), trans.strip()]))

    print(f"Total training pairs: {len(train_examples)}")
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # Use MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 4. Tune the model
    print(f"\nStep 4: Training (Fine-Tuning) the embedder model for {args.epochs} epoch(s)...")
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)  # 10% warmup
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True
    )

    # 5. Save fine-tuned model
    print(f"\nStep 5: Saving the fine-tuned embedder to {args.output_dir}...")
    model.save(args.output_dir)
    print("Embedding model fine-tuning completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    train()
