#!/usr/bin/env python3
"""
Model Inference Script
Loads a trained model and performs inference on new data
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import os
import sys
from typing import List, Dict, Any, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import (
    MODEL_CONFIG, 
    DATA_CONFIG, 
    OUTPUT_CONFIG,
    get_device, 
    get_data_paths
)

class LoanEvaluatorInference:
    """Inference class for loan evaluation model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the inference model
        
        Args:
            model_path: Path to trained model. If None, uses base model.
        """
        self.device = get_device()
        self.model_path = model_path or MODEL_CONFIG["model_id"]
        
        print(f"🤖 Loading model from: {self.model_path}")
        print(f"📱 Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir=MODEL_CONFIG["local_dir"]
        )
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=MODEL_CONFIG["local_dir"]
        )
        
        self.model.eval()
        
        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        print("✅ Model loaded successfully!")
    
    def create_chat_prompt(self, user_input: str) -> List[Dict[str, str]]:
        """Create a chat template prompt"""
        return [
            {
                "role": "system",
                "content": """You are a loan evaluator following these rules strictly:

- Applicant age must be at least 18 years old → else REJECT
- Credit score must be at least 670 → else REJECT
- Annual income must be at least $30,000 → else REJECT
- Debt-to-income ratio must be at most 40% → else REJECT
- Employment status must be one of employed_full_time, employed_part_time, self_employed, retired → else REJECT
- Employment duration in current role must be at least 6 months → else FLAG_REVIEW
- Residency status must be US_Citizen or Permanent_Resident → else REJECT
- Applicant must not have filed for bankruptcy in the last 7 years → else REJECT
- Requested loan amount must be at most 50% of annual income → else FLAG_REVIEW
- Applicant must have a verifiable bank account → else REJECT

Given an application, output a JSON object with:

{
  "result": "APPROVE", "REJECT", or "FLAG_REVIEW",
  "reasoning": "Explain which rules passed or failed and why the result was chosen."
}

Do not output anything else."""
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    
    def predict_single(self, user_input: str, max_new_tokens: int = 256) -> str:
        """
        Make a single prediction
        
        Args:
            user_input: The input text/JSON
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response
        """
        # Create chat messages
        messages = self.create_chat_prompt(user_input)
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        with torch.no_grad():
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for consistent outputs
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False  # Only return generated part
            )
        
        return outputs[0]["generated_text"].strip()
    
    def predict_batch(self, inputs: List[str], max_new_tokens: int = 256) -> List[str]:
        """
        Make predictions on a batch of inputs
        
        Args:
            inputs: List of input texts
            max_new_tokens: Maximum tokens to generate per input
            
        Returns:
            List of generated responses
        """
        results = []
        print(f"🔄 Processing {len(inputs)} inputs...")
        
        for i, user_input in enumerate(inputs):
            try:
                result = self.predict_single(user_input, max_new_tokens)
                results.append(result)
                print(f"✅ Completed {i+1}/{len(inputs)}")
            except Exception as e:
                print(f"❌ Error processing input {i+1}: {e}")
                results.append(f"Error: {str(e)}")
        
        return results
    
    def evaluate_loan_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a loan application
        
        Args:
            application_data: Dictionary with loan application data
            
        Returns:
            Evaluation result
        """
        # Convert application to JSON string
        json_input = json.dumps(application_data)
        
        # Get prediction
        raw_prediction = self.predict_single(json_input)
        
        try:
            # Try to parse as JSON
            prediction = json.loads(raw_prediction)
            return {
                "input": application_data,
                "prediction": prediction,
                "raw_output": raw_prediction,
                "status": "success"
            }
        except json.JSONDecodeError:
            return {
                "input": application_data,
                "prediction": None,
                "raw_output": raw_prediction,
                "status": "parse_error"
            }

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"❌ Test file not found: {file_path}")
    return data

def main():
    """Main inference function"""
    print("🚀 Starting Loan Evaluator Inference")
    print("=" * 50)
    
    # Model paths - you can change these
    TRAINED_MODEL_PATH = os.path.join(OUTPUT_CONFIG["output_dir"], "final_model")
    
    # Check if trained model exists
    if os.path.exists(TRAINED_MODEL_PATH):
        print(f"📁 Found trained model at: {TRAINED_MODEL_PATH}")
        model_path = TRAINED_MODEL_PATH
    else:
        print(f"⚠️  Trained model not found. Using base model: {MODEL_CONFIG['model_id']}")
        model_path = None
    
    # Initialize inference
    inferencer = LoanEvaluatorInference(model_path)
    
    # Example 1: Single prediction
    print("\n🔍 Example 1: Single Prediction")
    print("-" * 30)
    
    example_application = {
        "age": 35,
        "credit_score": 720,
        "annual_income_usd": 75000,
        "debt_to_income_ratio_percent": 25.0,
        "employment_status": "employed_full_time",
        "current_employment_duration_months": 24,
        "residency_status": "US_Citizen",
        "has_bankruptcy_recent": False,
        "requested_amount_usd": 30000,
        "has_verifiable_bank_account": True
    }
    
    result = inferencer.evaluate_loan_application(example_application)
    print(f"Input: {json.dumps(example_application, indent=2)}")
    print(f"Result: {json.dumps(result['prediction'], indent=2)}")
    
    # Example 2: Test data evaluation
    print("\n🔍 Example 2: Test Data Evaluation")
    print("-" * 30)
    
    data_paths = get_data_paths()
    test_data = load_test_data(data_paths["test"])
    
    if test_data:
        print(f"📊 Evaluating {min(5, len(test_data))} test samples...")
        
        for i, sample in enumerate(test_data[:5]):  # Evaluate first 5 samples
            if "messages" in sample:
                # Extract the user message (loan application)
                user_message = None
                for msg in sample["messages"]:
                    if msg["role"] == "user":
                        user_message = msg["content"]
                        break
                
                if user_message:
                    try:
                        app_data = json.loads(user_message)
                        result = inferencer.evaluate_loan_application(app_data)
                        
                        print(f"\n📋 Test Sample {i+1}:")
                        print(f"Prediction: {result['prediction']}")
                        
                        # Compare with ground truth if available
                        for msg in sample["messages"]:
                            if msg["role"] == "assistant":
                                print(f"Ground Truth: {msg['content']}")
                                break
                        
                    except Exception as e:
                        print(f"❌ Error processing sample {i+1}: {e}")
    
    print("\n✅ Inference completed!")

if __name__ == "__main__":
    main()
