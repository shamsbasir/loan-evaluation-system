import json
import random
from collections import defaultdict
import os
from typing import Dict, List, Any

# Allowed enum values for categorical fields
EMPLOYMENT_STATUSES = ["employed_full_time", "employed_part_time", "self_employed", "retired", "unemployed"]
RESIDENCY_STATUSES = ["US_Citizen", "Permanent_Resident", "Temporary_Resident", "Other"]

def evaluate_application(app):
    reasoning = []
    result = "APPROVE"

    # Age check
    if app["age"] < 18:
        reasoning.append(f"Applicant age {app['age']} is below minimum 18.")
        result = "REJECT"

    # Credit score check
    if app["credit_score"] < 670:
        reasoning.append(f"Credit score {app['credit_score']} is below minimum 670.")
        result = "REJECT"

    # Annual income check
    if app["annual_income_usd"] < 30000:
        reasoning.append(f"Annual income ${app['annual_income_usd']:,} is below minimum 30,000.")
        result = "REJECT"

    # Debt-to-income ratio
    if app["debt_to_income_ratio_percent"] > 40:
        reasoning.append(f"Debt-to-income ratio {app['debt_to_income_ratio_percent']}% exceeds max 40%.")
        result = "REJECT"

    # Employment status
    if app["employment_status"] not in EMPLOYMENT_STATUSES[:-1]:  # exclude "unemployed" as invalid
        reasoning.append(f"Employment status '{app['employment_status']}' is not allowed.")
        result = "REJECT"

    # Employment duration (flag review if less than 6 months)
    if app["current_employment_duration_months"] < 6 and result != "REJECT":
        reasoning.append(f"Employment duration {app['current_employment_duration_months']} months less than 6.")
        if result != "REJECT":
            result = "FLAG_REVIEW"

    # Residency status
    if app["residency_status"] not in ["US_Citizen", "Permanent_Resident"]:
        reasoning.append(f"Residency status '{app['residency_status']}' is not eligible.")
        result = "REJECT"

    # Bankruptcy recent
    if app["has_bankruptcy_recent"] is True:
        reasoning.append("Applicant has recent bankruptcy.")
        result = "REJECT"

    # Requested amount ≤ 0.5 * income (flag review if violated)
    if app["requested_amount_usd"] > 0.5 * app["annual_income_usd"] and result != "REJECT":
        reasoning.append(
            f"Requested loan amount ${app['requested_amount_usd']:,} exceeds 50% of annual income ${app['annual_income_usd']:,}."
        )
        if result != "REJECT":
            result = "FLAG_REVIEW"

    # Verifiable bank account
    if not app["has_verifiable_bank_account"]:
        reasoning.append("Applicant does not have a verifiable bank account.")
        result = "REJECT"

    if not reasoning:
        reasoning.append("All criteria met for approval.")

    return {
        "result": result,
        "reasoning": " ".join(reasoning)
    }

def generate_application_biased(target_result=None):
    """Generate application with bias towards a specific result"""
    if target_result == "APPROVE":
        return {
            "age": random.randint(25, 65),
            "credit_score": random.randint(670, 850),
            "annual_income_usd": random.randint(35000, 150000),
            "debt_to_income_ratio_percent": round(random.uniform(10, 35), 2),
            "employment_status": random.choice(EMPLOYMENT_STATUSES[:-1]),  # Exclude unemployed
            "current_employment_duration_months": random.randint(12, 60),
            "residency_status": random.choice(["US_Citizen", "Permanent_Resident"]),
            "has_bankruptcy_recent": False,
            "requested_amount_usd": lambda income: random.randint(5000, int(income * 0.4)),
            "has_verifiable_bank_account": True
        }
    elif target_result == "REJECT":
        reject_reasons = [
            # Age violation
            {"age": random.randint(16, 17)},
            # Credit score violation
            {"credit_score": random.randint(300, 669)},
            # Income violation
            {"annual_income_usd": random.randint(10000, 29999)},
            # DTI violation
            {"debt_to_income_ratio_percent": round(random.uniform(41, 80), 2)},
            # Employment violation
            {"employment_status": "unemployed"},
            # Residency violation
            {"residency_status": random.choice(["Temporary_Resident", "Other"])},
            # Bankruptcy violation
            {"has_bankruptcy_recent": True},
            # Bank account violation
            {"has_verifiable_bank_account": False}
        ]

        base_app = {
            "age": random.randint(22, 65),
            "credit_score": random.randint(670, 800),
            "annual_income_usd": random.randint(35000, 120000),
            "debt_to_income_ratio_percent": round(random.uniform(15, 35), 2),
            "employment_status": random.choice(EMPLOYMENT_STATUSES[:-1]),
            "current_employment_duration_months": random.randint(12, 48),
            "residency_status": random.choice(["US_Citizen", "Permanent_Resident"]),
            "has_bankruptcy_recent": False,
            "requested_amount_usd": 50000,
            "has_verifiable_bank_account": True
        }

        # Apply one or more rejection reasons
        violation = random.choice(reject_reasons)
        base_app.update(violation)
        return base_app

    elif target_result == "FLAG_REVIEW":
        base_app = {
            "age": random.randint(22, 65),
            "credit_score": random.randint(670, 800),
            "annual_income_usd": random.randint(35000, 120000),
            "debt_to_income_ratio_percent": round(random.uniform(15, 35), 2),
            "employment_status": random.choice(EMPLOYMENT_STATUSES[:-1]),
            "current_employment_duration_months": random.randint(0, 5),  # Short employment
            "residency_status": random.choice(["US_Citizen", "Permanent_Resident"]),
            "has_bankruptcy_recent": False,
            "requested_amount_usd": lambda income: random.randint(int(income * 0.51), int(income * 0.8)),  # High loan amount
            "has_verifiable_bank_account": True
        }
        return base_app
    else:
        # Random generation
        return {
            "age": random.randint(16, 75),
            "credit_score": random.randint(300, 850),
            "annual_income_usd": random.randint(15000, 200000),
            "debt_to_income_ratio_percent": round(random.uniform(5, 80), 2),
            "employment_status": random.choice(EMPLOYMENT_STATUSES),
            "current_employment_duration_months": random.randint(0, 60),
            "residency_status": random.choice(RESIDENCY_STATUSES),
            "has_bankruptcy_recent": random.choice([True, False]),
            "requested_amount_usd": random.randint(5000, 150000),
            "has_verifiable_bank_account": random.choice([True, False])
        }

def finalize_application(app_template):
    """Convert template with functions to final application"""
    app = app_template.copy()

    # Handle requested_amount_usd if it's a function
    if callable(app.get("requested_amount_usd")):
        app["requested_amount_usd"] = app["requested_amount_usd"](app["annual_income_usd"])
    elif "requested_amount_usd" not in app:
        app["requested_amount_usd"] = random.randint(5000, 100000)

    return app

system_prompt = """You are a loan evaluator following these rules strictly:

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

def generate_balanced_dataset(
    samples_per_class: int = 5000,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    output_dir: str = "loan_data",
    seed: int = 42
):
    """Generate balanced dataset with train/val/test splits"""

    random.seed(seed)

    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    os.makedirs(output_dir, exist_ok=True)

    classes = ["APPROVE", "REJECT", "FLAG_REVIEW"]
    all_data = []
    class_counts = defaultdict(int)
    generation_attempts = defaultdict(int)

    print("Generating balanced dataset...")
    print(f"Target: {samples_per_class} samples per class")

    # Generate data for each class
    for target_class in classes:
        print(f"\nGenerating {target_class} samples...")
        class_data = []

        while len(class_data) < samples_per_class:
            generation_attempts[target_class] += 1

            # Use biased generation 80% of the time, random 20%
            if random.random() < 0.8:
                app_template = generate_application_biased(target_class)
            else:
                app_template = generate_application_biased(None)

            app = finalize_application(app_template)
            eval_result = evaluate_application(app)

            if eval_result["result"] == target_class:
                sample = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(app)},
                        {"role": "assistant", "content": json.dumps(eval_result)}
                    ]
                }
                class_data.append(sample)
                class_counts[target_class] += 1

                if len(class_data) % 1000 == 0:
                    print(f"  Generated {len(class_data)}/{samples_per_class} {target_class} samples")

        all_data.extend(class_data)
        efficiency = (len(class_data) / generation_attempts[target_class]) * 100
        print(f"  Completed {target_class}: {len(class_data)} samples ({efficiency:.1f}% efficiency)")

    # Shuffle all data
    random.shuffle(all_data)

    # Calculate split sizes
    total_samples = len(all_data)
    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    test_size = total_samples - train_size - val_size

    # Split data
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    # Save datasets
    datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    for split_name, split_data in datasets.items():
        filepath = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(filepath, "w") as f:
            for sample in split_data:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(split_data)} samples to {filepath}")

    # Generate statistics
    stats = {
        "total_samples": total_samples,
        "class_distribution": dict(class_counts),
        "split_sizes": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "generation_efficiency": {
            cls: f"{(samples_per_class / generation_attempts[cls]) * 100:.1f}%"
            for cls in classes
        }
    }

    # Save statistics
    stats_file = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset Generation Complete!")
    print(f"Total samples: {total_samples}")
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Statistics saved to: {stats_file}")

    return datasets, stats

def generate_simple_dataset(samples_per_class: int = 1000, filename: str = "loan_data.jsonl"):
    """Simple generation for quick testing"""
    classes = ["APPROVE", "REJECT", "FLAG_REVIEW"]
    counts = defaultdict(int)
    data = []

    print(f"Generating {samples_per_class} samples per class...")

    while min(counts.values()) < samples_per_class:
        # Bias generation towards needed classes
        needed_classes = [cls for cls in classes if counts[cls] < samples_per_class]
        target_class = random.choice(needed_classes)

        if random.random() < 0.7:  # 70% biased generation
            app_template = generate_application_biased(target_class)
        else:
            app_template = generate_application_biased(None)

        app = finalize_application(app_template)
        eval_result = evaluate_application(app)

        if counts[eval_result["result"]] < samples_per_class:
            sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(app)},
                    {"role": "assistant", "content": json.dumps(eval_result)}
                ]
            }
            data.append(sample)
            counts[eval_result["result"]] += 1

            if sum(counts.values()) % 1000 == 0:
                print(f"Progress: {dict(counts)}")

    # Shuffle and save
    random.shuffle(data)
    with open(filename, "w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {len(data)} samples saved to {filename}")
    print(f"Final distribution: {dict(counts)}")
    return data

if __name__ == "__main__":
    # Generate full dataset with train/val/test splits
    datasets, stats = generate_balanced_dataset(
        samples_per_class=5000,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        output_dir="../data"
    )