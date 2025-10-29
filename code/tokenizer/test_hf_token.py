"""
Test HuggingFace token authentication
"""
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, whoami
from datasets import load_dataset

# Load .env file
load_dotenv()

print("=" * 80)
print("HUGGINGFACE TOKEN DIAGNOSIS")
print("=" * 80)

# 1. Check if token exists
token = os.getenv("HUGGINGFACE_TOKEN")
print(f"\n1. Token loaded from .env:")
if token:
    print(f"   ✓ Token found: {token[:10]}...{token[-5:]} (length: {len(token)})")
    print(f"   Full token: {token}")
else:
    print("   ✗ NO TOKEN FOUND")
    exit(1)

# 2. Check token format
print(f"\n2. Token format check:")
if token.startswith("hf_"):
    print(f"   ✓ Token starts with 'hf_' (correct format)")
else:
    print(f"   ✗ Token does NOT start with 'hf_' (incorrect format)")
    print(f"   Token starts with: {token[:10]}")

# 3. Test token with HuggingFace API
print(f"\n3. Testing token with HuggingFace API:")
try:
    api = HfApi(token=token)
    user_info = whoami(token=token)
    print(f"   ✓ Token is VALID")
    print(f"   User: {user_info.get('name', 'N/A')}")
    print(f"   Type: {user_info.get('type', 'N/A')}")
    print(f"   Auth: {user_info.get('auth', {}).get('type', 'N/A')}")
except Exception as e:
    print(f"   ✗ Token VALIDATION FAILED: {e}")
    exit(1)

# 4. Test dataset access - IndicCorpV2
print(f"\n4. Testing IndicCorpV2 access:")
try:
    ds = load_dataset(
        "ai4bharat/IndicCorpV2",
        "indiccorp_v2",
        split="hin_Deva",
        streaming=True,
        token=token
    )
    sample = next(iter(ds))
    print(f"   ✓ IndicCorpV2 accessible")
    print(f"   Sample keys: {list(sample.keys())}")
except Exception as e:
    print(f"   ✗ IndicCorpV2 FAILED: {e}")

# 5. Test dataset access - StarCoderData
print(f"\n5. Testing StarCoderData access:")
try:
    ds = load_dataset(
        "bigcode/starcoderdata",
        data_dir="python",
        split="train",
        streaming=True,
        token=token
    )
    sample = next(iter(ds))
    print(f"   ✓ StarCoderData accessible")
    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   Content length: {len(sample.get('content', ''))}")
except Exception as e:
    print(f"   ✗ StarCoderData FAILED: {e}")

# 6. Test dataset access - FineWeb-Edu
print(f"\n6. Testing FineWeb-Edu access:")
try:
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
        token=token
    )
    sample = next(iter(ds))
    print(f"   ✓ FineWeb-Edu accessible")
    print(f"   Sample keys: {list(sample.keys())}")
except Exception as e:
    print(f"   ✗ FineWeb-Edu FAILED: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

