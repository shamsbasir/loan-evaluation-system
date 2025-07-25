#!/usr/bin/env python3
"""
Simple Model Inference Script
Quick and easy way to test your trained model
"""
import sys
import os
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def quick_inference():
    """Quick inference with example data"""
    try:
        from src.inference import LoanEvaluatorInference
        from config import OUTPUT_CONFIG
        
        print("🚀 Quick Inference Test")
        print("=" * 30)
        
        # Try to load trained model
        trained_model_path = os.path.join(OUTPUT_CONFIG["output_dir"], "final_model")
        
        if os.path.exists(trained_model_path):
            print(f"✅ Using trained model")
            inferencer = LoanEvaluatorInference(trained_model_path)
        else:
            print(f"⚠️  Using base model (no training found)")
            inferencer = LoanEvaluatorInference()
        
        # Test with sample application
        sample_app = {
            "age": 30,
            "credit_score": 750,
            "annual_income_usd": 60000,
            "debt_to_income_ratio_percent": 20.0,
            "employment_status": "employed_full_time",
            "current_employment_duration_months": 18,
            "residency_status": "US_Citizen",
            "has_bankruptcy_recent": False,
            "requested_amount_usd": 25000,
            "has_verifiable_bank_account": True
        }
        
        print(f"\n📋 Testing with sample application:")
        print(json.dumps(sample_app, indent=2))
        
        result = inferencer.evaluate_loan_application(sample_app)
        
        print(f"\n🎯 Result:")
        if result["status"] == "success":
            print(json.dumps(result["prediction"], indent=2))
        else:
            print(f"❌ Error: {result['raw_output']}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have all dependencies installed!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_inference()
