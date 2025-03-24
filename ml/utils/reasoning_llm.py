import os
import json
import re
from transformers import pipeline

# Load once and reuse
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

REASONS = [
    "Price Mismatch",
    "Quantity Mismatch",
    "GL vs IHub Difference",
    "Timing Issue",
    "Unknown Reason"
]

EXPLANATION_MAP_PATH = "/shared/explanation_map.json"

def extract_reason(comment=None, features=None):
    # Prepare the sentence for LLM
    if comment:
        prompt = comment
    elif features:
        prompt = " ".join([f"{k} is {v}" for k, v in features.items()])
    else:
        prompt = "Unknown anomaly"

    try:
        result = classifier(prompt, REASONS)
        reason = result["labels"][0]
        return reason
    except Exception as e:
        print("❌ LLM failed:", e)
        return "Unknown Reason"

def generate_reasoning_map(data, save_path=EXPLANATION_MAP_PATH):
    mapping = {}
    for row in data:
        key = str(tuple(round(float(row[k]), 3) for k in row if isinstance(row[k], (int, float))))
        comment = row.get("COMMENT", "")
        reason = extract_reason(comment, row)
        mapping[key] = reason
    with open(save_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Saved explanation map to {save_path}")
