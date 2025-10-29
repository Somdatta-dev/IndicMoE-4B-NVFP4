"""
Quick test to verify the tokenizer training script fixes
Tests data loading without full training
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
import dotenv

dotenv.load_dotenv()

def test_indiccorp():
    """Test IndicCorpV2 loading"""
    print("Testing IndicCorpV2 loading...")
    
    lang_to_split = {
        "hi": "hin_Deva",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "bn": "ben_Beng",
        "mr": "mar_Deva",
        "gu": "guj_Gujr",
        "kn": "kan_Knda"
    }
    
    for lang, split_name in lang_to_split.items():
        try:
            print(f"  Testing {lang} ({split_name})...", end=" ")
            dataset = load_dataset(
                "ai4bharat/IndicCorpV2",
                name="indiccorp_v2",
                split=split_name,
                streaming=True,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            # Get first sample
            sample = next(iter(dataset))
            text = sample.get('text', '')
            print(f"✓ OK ({len(text)} chars)")
        except Exception as e:
            print(f"✗ FAILED: {e}")

def test_code_dataset():
    """Test code dataset loading"""
    print("\nTesting code dataset loading...")
    
    datasets_to_try = [
        ("bigcode/the-stack-v2", "The Stack v2 (BEST - 900B tokens)"),
        ("bigcode/starcoderdata", "StarCoderData"),
        ("codeparrot/github-code", "GitHub Code"),
    ]
    
    for dataset_name, display_name in datasets_to_try:
        try:
            print(f"  Testing {display_name}...", end=" ")
            dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            sample = next(iter(dataset))
            content = sample.get('content', '') or sample.get('code', '') or sample.get('text', '')
            print(f"✓ OK ({len(content)} chars)")
            return dataset_name  # Return first working dataset
        except Exception as e:
            print(f"✗ FAILED: {str(e)[:50]}")
    
    return None

def main():
    print("=" * 60)
    print("TOKENIZER DATA LOADING TEST")
    print("=" * 60)
    print()
    
    # Check token
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token or token == "your_token_here":
        print("❌ HUGGINGFACE_TOKEN not set!")
        print("Please add your token to .env file")
        return
    
    print(f"✓ HuggingFace token found: {token[:10]}...")
    print()
    
    # Test Indic languages
    test_indiccorp()
    
    # Test code dataset
    working_dataset = test_code_dataset()
    
    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if working_dataset:
        print(f"✓ Use code dataset: {working_dataset}")
    else:
        print("⚠️  No code dataset accessible, will skip code data")
    
    print()
    print("Ready to train tokenizer!")

if __name__ == "__main__":
    main()

