"""
Configuration file for training setup
"""
import os

# Get project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Model Configuration
MODEL_CONFIG = {
    "model_id": "unsloth/Llama-3.2-1B-Instruct",
    "local_dir": os.path.join(PROJECT_ROOT, "models", "Llama-3.2-1B-Instruct"),
    "torch_dtype": "bfloat16",
    "device_map": "auto"
}

# Data Configuration
DATA_CONFIG = {
    "data_dir": os.path.join(PROJECT_ROOT, "data"),
    "train_file": "train.jsonl",
    "val_file": "val.jsonl",
    "test_file": "test.jsonl",
    "max_length": 1024
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 4,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100
}

# DataLoader Configuration
DATALOADER_CONFIG = {
    "num_workers": 0,  # Set to 0 to avoid multiprocessing issues
    "pin_memory": False,
    "shuffle_train": True,
    "shuffle_val": False
}

# Tokenizer Configuration
TOKENIZER_CONFIG = {
    "padding_side": "left",
    "truncation": True,
    "padding": True
}

# Device Configuration
DEVICE_CONFIG = {
    "use_cuda": True,
    "use_mps": True,  # For Apple Silicon
    "fallback_cpu": True
}

# Output Configuration
OUTPUT_CONFIG = {
    "output_dir": os.path.join(PROJECT_ROOT, "outputs"),
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "logs_dir": os.path.join(PROJECT_ROOT, "logs"),
    "save_total_limit": 3,
    "save_strategy": "steps"
}

def create_project_directories():
    """Create all necessary project directories"""
    directories_to_create = [
        # Output directories
        OUTPUT_CONFIG["output_dir"],
        OUTPUT_CONFIG["checkpoint_dir"],
        OUTPUT_CONFIG["logs_dir"],
        
        # Model cache directory
        MODEL_CONFIG["local_dir"],
        
        # Utils directory (now at root level)
        os.path.join(PROJECT_ROOT, "utils"),
        
        # Additional useful directories
        os.path.join(PROJECT_ROOT, "scripts"),
        os.path.join(PROJECT_ROOT, "experiments"),
        os.path.join(PROJECT_ROOT, "plots"),
    ]
    
    created_dirs = []
    for directory in directories_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created_dirs.append(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        os.path.join(PROJECT_ROOT, "utils", "__init__.py"),
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""Utils package for training utilities"""\n')
            print(f"✅ Created __init__.py: {init_file}")
    
    return created_dirs

def get_device():
    """Get the best available device"""
    import torch
    
    if DEVICE_CONFIG["use_cuda"] and torch.cuda.is_available():
        return "cuda"
    elif DEVICE_CONFIG["use_mps"] and torch.backends.mps.is_available():
        return "mps"
    elif DEVICE_CONFIG["fallback_cpu"]:
        return "cpu"
    else:
        return "cpu"

def get_data_paths():
    """Get full paths to data files"""
    return {
        "train": os.path.join(DATA_CONFIG["data_dir"], DATA_CONFIG["train_file"]),
        "val": os.path.join(DATA_CONFIG["data_dir"], DATA_CONFIG["val_file"]),
        "test": os.path.join(DATA_CONFIG["data_dir"], DATA_CONFIG["test_file"])
    }

def print_config():
    """Print current configuration"""
    print("=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Model: {MODEL_CONFIG['model_id']}")
    print(f"Device: {get_device()}")
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Max Length: {DATA_CONFIG['max_length']}")
    print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"Data Directory: {DATA_CONFIG['data_dir']}")
    print(f"Output Directory: {OUTPUT_CONFIG['output_dir']}")
    print("=" * 50)

# Create all project directories when config is imported
create_project_directories()

if __name__ == "__main__":
    print_config()
    print("\nData paths:")
    for key, path in get_data_paths().items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {key}: {path} {exists}")
