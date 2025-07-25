import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from typing import List, Dict, Any
import json
import os
import sys
from tqdm import tqdm

# Add project root to Python path to import from utils and config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.dataloader_utils import ConversationDataset, CollateFn
from config import (
    MODEL_CONFIG, 
    DATA_CONFIG, 
    TRAINING_CONFIG, 
    DATALOADER_CONFIG,
    TOKENIZER_CONFIG,
    OUTPUT_CONFIG,
    get_device, 
    get_data_paths,
    print_config
)

# Print configuration
print("🚀 Starting Training Setup...")
print_config()

print(f"\n📁 Project Directories:")
print(f"✅ All necessary directories have been created!")
print(f"Check PROJECT_STRUCTURE.md for detailed information.")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CONFIG["model_id"], 
    cache_dir=MODEL_CONFIG["local_dir"]
)

# Configure tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = TOKENIZER_CONFIG["padding_side"]

# Get data paths
data_paths = get_data_paths()

# Create datasets using configuration
train_dataset = ConversationDataset(
    data_paths["train"], 
    tokenizer, 
    max_length=DATA_CONFIG["max_length"]
)
val_dataset = ConversationDataset(
    data_paths["val"], 
    tokenizer, 
    max_length=DATA_CONFIG["max_length"]
)

# Create picklable collate function
collate_func = CollateFn(tokenizer)

# Create dataloaders using configuration
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAINING_CONFIG["batch_size"],
    shuffle=DATALOADER_CONFIG["shuffle_train"],
    num_workers=DATALOADER_CONFIG["num_workers"],
    collate_fn=collate_func,
    pin_memory=DATALOADER_CONFIG["pin_memory"]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=TRAINING_CONFIG["batch_size"],
    shuffle=DATALOADER_CONFIG["shuffle_val"],
    num_workers=DATALOADER_CONFIG["num_workers"],
    collate_fn=collate_func,
    pin_memory=DATALOADER_CONFIG["pin_memory"]
)

# Test the dataloader
print(f"\n📊 Dataset Info:")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Get a sample batch
batch = next(iter(train_loader))
print(f"\n🔍 Batch Info:")
print(f"Batch keys: {batch.keys()}")
print(f"Input shape: {batch['input_ids'].shape}")
print(f"Labels shape: {batch['labels'].shape}")
print(f"Attention mask shape: {batch['attention_mask'].shape}")
print(f"Device: {get_device()}")



# Initialize model
print(f"\n🤖 Initializing model...")
device = get_device()
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG["model_id"], 
    cache_dir=MODEL_CONFIG["local_dir"],
    torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]) if isinstance(MODEL_CONFIG["torch_dtype"], str) else MODEL_CONFIG["torch_dtype"],
    device_map=MODEL_CONFIG["device_map"]
)

# Print model info
print(f"Model loaded: {MODEL_CONFIG['model_id']}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ===============================
# TRAINING SETUP
# ===============================
print(f"\n⚙️ Setting up training...")
optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
model.train()

# Training configuration variables
NUM_EPOCHS = TRAINING_CONFIG["num_epochs"]
GRADIENT_ACCUMULATION_STEPS = TRAINING_CONFIG["gradient_accumulation_steps"]
SAVE_EVERY = TRAINING_CONFIG["save_steps"]
OUTPUT_DIR = OUTPUT_CONFIG["output_dir"]

# Calculate total steps
total_steps = len(train_loader) * NUM_EPOCHS
print(f"Total training steps: {total_steps}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")

# ===============================
# TRAINING LOOP
# ===============================
global_step = 0
total_loss = 0
epoch_losses = []

print(f"\n🚀 Starting training...")
try:
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        epoch_loss = 0

        for step, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                epoch_loss += loss.item()

                # Backward pass
                loss.backward()

                # Update weights every N steps
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAINING_CONFIG["max_grad_norm"])
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Update progress bar
                avg_loss = epoch_loss / (step + 1)
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "step": global_step
                })

                # Save checkpoint
                if global_step > 0 and global_step % SAVE_EVERY == 0:
                    checkpoint_dir = os.path.join(OUTPUT_CONFIG["checkpoint_dir"], f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print(f"\n💾 Saved checkpoint at step {global_step}")

            except Exception as e:
                print(f"\n❌ Error in training step {step}: {e}")
                continue

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        print(f"✅ Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

except KeyboardInterrupt:
    print("\n⏹️ Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training error: {e}")

# ===============================
# SAVE FINAL MODEL
# ===============================
print(f"\n💾 Saving final model...")
final_model_dir = os.path.join(OUTPUT_CONFIG["output_dir"], "final_model")
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"✅ Model saved to {final_model_dir}")

# Print training summary
print(f"\n📊 Training Summary:")
print(f"Total epochs: {NUM_EPOCHS}")
print(f"Total steps: {global_step}")
print(f"Final loss: {epoch_losses[-1] if epoch_losses else 'N/A':.4f}")
print(f"Model saved to: {final_model_dir}")
