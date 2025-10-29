"""
Configuration for IndicMoE-4B data processing pipeline
"""

import os
from pathlib import Path

# Data directories
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "/workspace/data/raw")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "/workspace/data/processed")
CACHE_DIR = os.getenv("CACHE_DIR", "/workspace/data/cache")

# Tokenizer configuration
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "meta-llama/Llama-3.2-1B")
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "2048"))

# Processing configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# Dataset configuration with streaming and language priorities
DATA_CONFIG = {
    "raw_data_dir": RAW_DATA_DIR,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "cache_dir": CACHE_DIR,
    
    "datasets": {
        # Phase 1: Pre-training (Multilingual + Coding)
        "phase1_pretraining": {
            # Indian languages - IndicCorpV2 (largest Indic corpus)
            "indiccorp_v2": {
                "name": "ai4bharat/IndicCorpV2",
                "subsets": ["hi", "ta", "te", "bn", "mr", "gu", "kn"],  # 7 major languages
                "streaming": True,
                "text_field": "text",
                "weight": 0.15,  # 15% of training data
            },
            
            # Wikipedia - Multilingual (high quality)
            "wikipedia": {
                "name": "wikimedia/wikipedia",
                "subsets": [
                    "20231101.en",  # English
                    "20231101.hi",  # Hindi
                    "20231101.ta",  # Tamil
                    "20231101.te",  # Telugu
                    "20231101.bn",  # Bengali
                    "20231101.mr",  # Marathi
                ],
                "streaming": True,
                "text_field": "text",
                "weight": 0.10,  # 10% of training data
            },
            
            # mC4 - Multilingual C4 (web crawl)
            "mc4": {
                "name": "mc4",
                "subsets": ["en", "hi", "ta", "te", "bn", "mr"],
                "streaming": True,
                "text_field": "text",
                "weight": 0.20,  # 20% of training data
            },
            
            # FineWeb-Edu - High quality English (education focused)
            "fineweb_edu": {
                "name": "HuggingFaceFW/fineweb-edu",
                "subset": "sample-10BT",  # 10B token sample
                "streaming": True,
                "text_field": "text",
                "weight": 0.35,  # 35% (English as main language)
            },
            
    # Code - ONLY Python, JavaScript, HTML, CSS (no TypeScript, Java, etc.)
    # Using StarCoderData: pre-filtered, decontaminated, deduplicated (250B tokens)
    "code_starcoderdata": {
        "name": "bigcode/starcoderdata",
        "languages": ["python", "javascript", "html", "css"],  # Language-separated dirs
        "streaming": True,
        "text_field": "content",
        "weight": 0.20,  # 20% coding knowledge
    },
        },
        
        # Phase 2: Instruction Tuning (Optional - for later)
        "phase2_instruction": {
            "indic_instruct": {
                "name": "Cognitive-Lab/indic-instruct",
                "subset": "all",
                "streaming": True,
                "text_field": "output",
                "weight": 0.50,
            },
            
            "openorca": {
                "name": "Open-Orca/OpenOrca",
                "streaming": True,
                "text_field": "response",
                "weight": 0.30,
            },
            
            "code_alpaca": {
                "name": "sahil2801/CodeAlpaca-20k",
                "streaming": False,
                "text_field": "output",
                "weight": 0.20,
            },
        },
        
        # Phase 3: Function Calling (Optional - for later)
        "phase3_function_calling": {
            "glaive_function_calling": {
                "name": "glaiveai/glaive-function-calling-v2",
                "streaming": True,
                "text_field": "text",
                "weight": 1.0,
            },
        },
    },
    
    # Language priority weights
    "language_weights": {
        "en": 0.55,      # English (main language)
        "hi": 0.10,      # Hindi
        "ta": 0.05,      # Tamil
        "te": 0.05,      # Telugu
        "bn": 0.05,      # Bengali
        "mr": 0.05,      # Marathi
        "gu": 0.05,      # Gujarati
        "kn": 0.05,      # Kannada
        "code": 0.05,    # Programming languages
    },
    
    # Processing configuration
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    
    # Parquet output configuration
    "parquet_row_group_size": 10000,
    "parquet_compression": "snappy",
}

