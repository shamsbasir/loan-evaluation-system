from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import json
import torch
from transformers import AutoTokenizer

def split_prompt_completion(messages):
    """
    Given a list of messages with roles (system, user, assistant),
    return a tuple: (messages_without_assistant, assistant_response)
    """
    messages_without_assistant = []
    assistant_response = None

    for message in messages:
        if message["role"] == "assistant":
            assistant_response = message["content"]
        else:
            messages_without_assistant.append(message)

    return messages_without_assistant, assistant_response


def generate_input_output_pair(prompt, target_responses, tokenizer):
    # Apply the chat template to each prompt
    chat_templates = tokenizer.apply_chat_template(prompt, continue_final_message=True,tokenize=False)

    # Append assistant response + EOS token to each prompt
    full_response_text = [
        chat_template + " " + target_response + tokenizer.eos_token
        for chat_template, target_response in zip(chat_templates, target_responses)
    ]

    # Tokenize the full input (prompt + response)
    input_ids_tokenized = tokenizer(
        full_response_text,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"]

    # Tokenize only the responses (with EOS)
    labels_tokenized = tokenizer(
        [" " + response + tokenizer.eos_token for response in target_responses],
        add_special_tokens=False,
        return_tensors="pt",
        padding="max_length",
        max_length=input_ids_tokenized.shape[1]
    )["input_ids"]
    

    # Replace padding tokens in labels with -100 so they are ignored in loss
    labels_tokenized_fixed = torch.where(
        labels_tokenized != tokenizer.pad_token_id,
        labels_tokenized,
        torch.tensor(-100)
    )
    labels_tokenized_fixed[:,-1] = tokenizer.pad_token_id  # Ensure last token is pad token

    # Left shift input_ids (remove last token) & right shift labels (remove first token)
    input_ids_tokenized_left_shifted = input_ids_tokenized[:, :-1]
    labels_tokenized_right_shifted = labels_tokenized_fixed[:, 1:]

    # Create attention mask for the shifted input
    attention_mask = (input_ids_tokenized_left_shifted != tokenizer.pad_token_id)

    return {
        "input_ids": input_ids_tokenized_left_shifted,
        "labels": labels_tokenized_right_shifted,
        "attention_mask": attention_mask
    }


class ConversationDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data_point = json.loads(line.strip())
                    messages, response = split_prompt_completion(data_point["messages"])
                    if messages and response:
                        self.data.append((messages, response))
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(self.data)} conversation pairs from {jsonl_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        messages, response = self.data[idx]
        
        # Use your existing function to generate the input-output pair
        batch = generate_input_output_pair([messages], [response], self.tokenizer)
        
        # Extract the first (and only) item from the batch
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        attention_mask = batch["attention_mask"][0]
        
        # Truncate to max_length if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        return {
            "input_ids": input_ids,
            "labels": labels, 
            "attention_mask": attention_mask
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer):
    """Custom collate function to handle variable length sequences."""
    # Find max length in batch
    max_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids = []
    labels = []
    attention_mask = []
    
    for item in batch:
        # Current lengths
        curr_len = len(item["input_ids"])
        pad_len = max_len - curr_len
        
        if pad_len > 0:
            # Pad input_ids with pad_token_id
            padded_input = torch.cat([
                item["input_ids"],
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=item["input_ids"].dtype)
            ])
            
            # Pad attention_mask with 0s
            padded_attention = torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=item["attention_mask"].dtype)
            ])
            
            # Pad labels with -100
            padded_labels = torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=item["labels"].dtype)
            ])
        else:
            padded_input = item["input_ids"]
            padded_attention = item["attention_mask"] 
            padded_labels = item["labels"]
        
        input_ids.append(padded_input)
        labels.append(padded_labels)
        attention_mask.append(padded_attention)
    
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }


class CollateFn:
    """Picklable collate function class."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        return collate_fn(batch, self.tokenizer)
