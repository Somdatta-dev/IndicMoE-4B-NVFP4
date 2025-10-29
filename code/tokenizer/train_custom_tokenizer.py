"""
Train custom tokenizer for IndicMoE-4B
Optimized for Indian languages, English, and code

Based on research:
- "Multilingual Tokenization through the Lens of Indian Languages" (2024)
- "Scaling Laws with Vocabulary" (2024)
- Sarvam-1 tokenizer design (68k vocab, 1.4-2.1 fertility)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC, Sequence as NormalizerSequence
from transformers import PreTrainedTokenizerFast
import datasets
from dotenv import load_dotenv

# Load environment variables from /workspace/.env
env_path = Path("/workspace/.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✓ Loaded .env from {env_path}")
else:
    load_dotenv()  # Fallback to default search
    print(f"⚠ Using default .env search (not found at {env_path})")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicMoETokenizerTrainer:
    """
    Train custom tokenizer optimized for:
    - Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada)
    - English
    - Programming languages
    
    Algorithm: Byte-level BPE (best for multilingual + code)
    Vocab Size: 128k (expandable to 256k)
    """
    
    def __init__(
        self,
        vocab_size: int = 128000,
        min_frequency: int = 2,
        output_dir: str = "/workspace/tokenizers/indicmoe_tokenizer",
        cache_dir: str = "/workspace/data/cache"
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Special tokens
        self.special_tokens = [
            "<|endoftext|>",
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|unk|>",
            # Language markers
            "<|english|>",
            "<|hindi|>",
            "<|tamil|>",
            "<|telugu|>",
            "<|bengali|>",
            "<|marathi|>",
            "<|gujarati|>",
            "<|kannada|>",
            # Content type markers
            "<|code|>",
            "<|python|>",
            "<|javascript|>",
            "<|html|>",
            "<|css|>",
            # Instruction markers
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ]
        
        logger.info(f"Initializing tokenizer trainer:")
        logger.info(f"  Vocab size: {vocab_size:,}")
        logger.info(f"  Special tokens: {len(self.special_tokens)}")
    
    def create_training_corpus(
        self,
        data_config: Dict,
        max_samples_per_dataset: Optional[int] = 1_000_000,
        english_ratio: float = 0.35,
        indic_ratio: float = 0.55,
        code_ratio: float = 0.10
    ):
        """
        Create balanced training corpus from datasets
        
        Target distribution (optimized for Indic languages):
        - English: 35% (main language, but reduced for better Indic fertility)
        - Indic languages: 55% (7.86% each for 7 languages) - Higher for better tokenization
        - Code: 10% (Python, JavaScript, HTML, CSS)
        """
        logger.info("=" * 80)
        logger.info("CREATING TRAINING CORPUS")
        logger.info("=" * 80)
        
        corpus_file = self.cache_dir / "tokenizer_training_corpus.txt"
        
        if corpus_file.exists():
            logger.info(f"✓ Corpus already exists: {corpus_file}")
            return str(corpus_file)
        
        # Calculate samples per category
        total_samples = max_samples_per_dataset
        english_samples = int(total_samples * english_ratio)
        indic_samples = int(total_samples * indic_ratio)
        code_samples = int(total_samples * code_ratio)
        indic_per_lang = indic_samples // 7  # 7 Indic languages
        
        logger.info(f"Target samples:")
        logger.info(f"  English: {english_samples:,}")
        logger.info(f"  Indic (total): {indic_samples:,} ({indic_per_lang:,} per language)")
        logger.info(f"  Code: {code_samples:,}")
        
        corpus_stats = defaultdict(int)
        
        with open(corpus_file, 'w', encoding='utf-8') as f:
            # 1. English data
            logger.info("\n[1/3] Collecting English data...")
            english_count = 0
            
            # FineWeb-Edu (high quality English)
            try:
                dataset = datasets.load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train",
                    streaming=True,
                    cache_dir=str(self.cache_dir)
                )
                
                for i, sample in enumerate(dataset):
                    if english_count >= english_samples:
                        break
                    
                    text = sample.get('text', '')
                    if text and len(text) > 100:
                        f.write(text.strip() + '\n')
                        english_count += 1
                        corpus_stats['english'] += 1
                    
                    if (i + 1) % 10000 == 0:
                        logger.info(f"  English: {english_count:,} / {english_samples:,}")
                
                logger.info(f"✓ English data collected: {english_count:,} samples")
            
            except Exception as e:
                logger.error(f"Error loading English data: {e}")
            
            # 2. Indic languages data
            logger.info("\n[2/3] Collecting Indic languages data...")
            
            languages = {
                "hi": "Hindi",
                "ta": "Tamil",
                "te": "Telugu",
                "bn": "Bengali",
                "mr": "Marathi",
                "gu": "Gujarati",
                "kn": "Kannada"
            }
            
            # Map language codes to correct split names
            lang_to_split = {
                "hi": "hin_Deva",
                "ta": "tam_Taml",
                "te": "tel_Telu",
                "bn": "ben_Beng",
                "mr": "mar_Deva",
                "gu": "guj_Gujr",
                "kn": "kan_Knda"
            }
            
            for lang_code, lang_name in languages.items():
                logger.info(f"  Processing {lang_name}...")
                lang_count = 0
                
                try:
                    # IndicCorpV2 with correct split names
                    split_name = lang_to_split.get(lang_code)
                    if not split_name:
                        logger.warning(f"  Skipping {lang_name} - no split mapping")
                        continue
                    
                    dataset = datasets.load_dataset(
                        "ai4bharat/IndicCorpV2",
                        name="indiccorp_v2",
                        split=split_name,
                        streaming=True,
                        cache_dir=str(self.cache_dir),
                        token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                    
                    for i, sample in enumerate(dataset):
                        if lang_count >= indic_per_lang:
                            break
                        
                        text = sample.get('text', '')
                        if text and len(text) > 100:
                            f.write(text.strip() + '\n')
                            lang_count += 1
                            corpus_stats[lang_code] += 1
                        
                        if (i + 1) % 5000 == 0:
                            logger.info(f"    {lang_name}: {lang_count:,} / {indic_per_lang:,}")
                    
                    logger.info(f"  ✓ {lang_name}: {lang_count:,} samples")
                
                except Exception as e:
                    logger.error(f"  Error loading {lang_name} data: {e}")
            
            # 3. Code data (ONLY: Python, HTML, CSS, JavaScript)
            logger.info("\n[3/3] Collecting code data (Python, HTML, CSS, JS only)...")
            code_count = 0
            
            # StarCoderData - High-quality code dataset (250B tokens, 86 languages)
            # Pre-filtered, decontaminated, PII-removed, deduplicated
            # Language-separated for efficient loading
            languages_to_load = ['python', 'javascript', 'html', 'css']
            
            for lang in languages_to_load:
                try:
                    logger.info(f"  Loading {lang.upper()}...")
                    dataset = datasets.load_dataset(
                        "bigcode/starcoderdata",
                        data_dir=lang,
                        split="train",
                        streaming=True,
                        cache_dir=str(self.cache_dir),
                        token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                    
                    lang_count = 0
                    target_per_lang = code_samples // 4  # Split evenly among 4 languages
                    
                    for i, sample in enumerate(dataset):
                        if lang_count >= target_per_lang:
                            break
                        
                        content = sample.get('content', '')
                        
                        # Quality filter: minimum length
                        if content and len(content) > 100:
                            f.write(content.strip() + '\n')
                            code_count += 1
                            lang_count += 1
                            corpus_stats['code'] += 1
                            corpus_stats[f'code_{lang}'] = corpus_stats.get(f'code_{lang}', 0) + 1
                        
                        if (lang_count + 1) % 5000 == 0:
                            logger.info(f"    {lang.upper()}: {lang_count:,} / {target_per_lang:,}")
                    
                    logger.info(f"  ✓ {lang.upper()}: {lang_count:,} samples")
                
                except Exception as e:
                    logger.error(f"  Error loading {lang} data: {e}")
            
            logger.info(f"✓ Total code data collected: {code_count:,} samples")
        
        # Print statistics
        logger.info("\n" + "=" * 80)
        logger.info("CORPUS STATISTICS")
        logger.info("=" * 80)
        total = sum(corpus_stats.values())
        logger.info(f"Total samples: {total:,}")
        
        for category, count in sorted(corpus_stats.items()):
            pct = (count / total * 100) if total > 0 else 0
            logger.info(f"  {category}: {count:,} ({pct:.1f}%)")
        
        # Show code language breakdown if available
        code_langs = {k: v for k, v in corpus_stats.items() if k.startswith('code_')}
        if code_langs:
            logger.info("\nCode language breakdown:")
            for lang, count in sorted(code_langs.items()):
                logger.info(f"  {lang}: {count:,}")
        
        logger.info(f"\n✓ Corpus saved to: {corpus_file}")
        return str(corpus_file)
    
    def train_tokenizer(self, corpus_file: str):
        """
        Train Byte-level BPE tokenizer
        
        Based on best practices:
        - Byte-level for Indic script support
        - BPE for multilingual + code
        - NFKC normalization for unicode
        """
        logger.info("=" * 80)
        logger.info("TRAINING TOKENIZER")
        logger.info("=" * 80)
        
        # Initialize Byte-level BPE
        tokenizer = Tokenizer(models.BPE(
            unk_token="<|unk|>",
            byte_fallback=True  # Essential for Indic scripts
        ))
        
        # Normalization: NFKC for unicode compatibility
        tokenizer.normalizer = NormalizerSequence([NFKC()])
        
        # Pre-tokenizer: ByteLevel for robust handling
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True
        )
        
        # Decoder
        tokenizer.decoder = decoders.ByteLevel()
        
        # Post-processor for special tokens
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        logger.info(f"Training with:")
        logger.info(f"  Vocabulary size: {self.vocab_size:,}")
        logger.info(f"  Min frequency: {self.min_frequency}")
        logger.info(f"  Special tokens: {len(self.special_tokens)}")
        logger.info(f"  Corpus: {corpus_file}")
        logger.info("")
        logger.info("This may take 30-60 minutes for large corpus...")
        
        # Train from file
        def corpus_iterator():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip()
        
        tokenizer.train_from_iterator(
            corpus_iterator(),
            trainer=trainer,
            length=None  # Unknown length for iterator
        )
        
        logger.info("✓ Tokenizer training complete!")
        
        return tokenizer
    
    def save_tokenizer(self, tokenizer: Tokenizer):
        """Save tokenizer in HuggingFace format"""
        logger.info("=" * 80)
        logger.info("SAVING TOKENIZER")
        logger.info("=" * 80)
        
        # Save raw tokenizer
        tokenizer_file = self.output_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_file))
        logger.info(f"✓ Raw tokenizer saved: {tokenizer_file}")
        
        # Wrap in HuggingFace PreTrainedTokenizerFast
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<|unk|>",
            bos_token="<|bos|>",
            eos_token="<|eos|>",
            pad_token="<|pad|>",
            model_max_length=2048,
            padding_side="right",
            truncation_side="right"
        )
        
        # Save HuggingFace format
        hf_tokenizer.save_pretrained(str(self.output_dir))
        logger.info(f"✓ HuggingFace tokenizer saved: {self.output_dir}")
        
        # Save metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "algorithm": "Byte-level BPE",
            "special_tokens": self.special_tokens,
            "languages": ["en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "code"],
            "max_length": 2048,
            "created_for": "IndicMoE-4B-FP4"
        }
        
        with open(self.output_dir / "tokenizer_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved")
        
        return hf_tokenizer
    
    def evaluate_tokenizer(self, tokenizer: Tokenizer):
        """Evaluate tokenizer fertility (tokens per word) for different languages"""
        logger.info("=" * 80)
        logger.info("TOKENIZER EVALUATION")
        logger.info("=" * 80)
        
        test_sentences = {
            "English": "The quick brown fox jumps over the lazy dog.",
            "Hindi": "भारत एक विशाल और विविधतापूर्ण देश है।",
            "Tamil": "இந்தியா ஒரு பெரிய மற்றும் பன்முக நாடு.",
            "Telugu": "భారతదేశం ఒక గొప్ప మరియు వైవిధ్యమైన దేశం.",
            "Bengali": "ভারত একটি বিশাল এবং বৈচিত্র্যময় দেশ।",
            "Marathi": "भारत एक विशाल आणि वैविध्यपूर्ण देश आहे.",
            "Gujarati": "ભારત એક વિશાળ અને વૈવિધ્યસભર દેશ છે.",
            "Kannada": "ಭಾರತವು ಒಂದು ದೊಡ್ಡ ಮತ್ತು ವೈವಿಧ್ಯಮಯ ದೇಶವಾಗಿದೆ.",
            "Python": "def hello_world():\n    print('Hello, World!')\n    return True",
            "JavaScript": "function helloWorld() {\n  console.log('Hello, World!');\n  return true;\n}"
        }
        
        logger.info("Fertility Scores (tokens per word):")
        logger.info("-" * 80)
        
        results = {}
        for lang, text in test_sentences.items():
            encoded = tokenizer.encode(text)
            num_tokens = len(encoded.tokens)
            num_words = len(text.split())
            fertility = num_tokens / num_words if num_words > 0 else 0
            
            results[lang] = {
                "text": text[:50] + "..." if len(text) > 50 else text,
                "tokens": num_tokens,
                "words": num_words,
                "fertility": fertility
            }
            
            logger.info(f"  {lang:12} | Tokens: {num_tokens:3} | Words: {num_words:3} | Fertility: {fertility:.2f}")
        
        logger.info("-" * 80)
        
        # Target: English ~1.4, Indic ~1.5-2.5, Code ~2-3
        avg_indic_fertility = sum(
            results[lang]["fertility"] 
            for lang in ["Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", "Kannada"]
        ) / 7
        
        logger.info(f"\nAverage Indic fertility: {avg_indic_fertility:.2f}")
        logger.info(f"Target: 1.5-2.5 (Sarvam-1 achieved 1.4-2.1)")
        
        if avg_indic_fertility <= 2.5:
            logger.info("✓ EXCELLENT: Fertility within target range!")
        elif avg_indic_fertility <= 3.5:
            logger.info("⚠ GOOD: Fertility acceptable but could be improved")
        else:
            logger.info("✗ POOR: Fertility too high, consider increasing vocab size")
        
        return results


def main():
    import argparse
    import shutil
    
    parser = argparse.ArgumentParser(description="Train custom tokenizer for IndicMoE-4B")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128000,
        choices=[65536, 98304, 128000, 196608, 256000],
        help="Vocabulary size (default: 128k)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1_000_000,
        help="Max samples for training corpus (default: 1M)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/tokenizers/indicmoe_tokenizer",
        help="Output directory"
    )
    parser.add_argument(
        "--skip-corpus-creation",
        action="store_true",
        help="Skip corpus creation if already exists"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip automatic cleanup of previous tokenizer and corpus"
    )
    
    args = parser.parse_args()
    
    # Auto-cleanup: Remove old tokenizer and corpus unless --no-cleanup
    if not args.no_cleanup:
        logger.info("=" * 80)
        logger.info("CLEANUP: Removing previous artifacts")
        logger.info("=" * 80)
        
        # Remove old tokenizer directory
        tokenizer_path = Path(args.output_dir)
        if tokenizer_path.exists():
            logger.info(f"  Removing old tokenizer: {tokenizer_path}")
            shutil.rmtree(tokenizer_path)
        
        # Remove old corpus file
        corpus_path = Path("/workspace/data/cache/tokenizer_training_corpus.txt")
        if corpus_path.exists():
            logger.info(f"  Removing old corpus: {corpus_path}")
            corpus_path.unlink()
        
        logger.info("✓ Cleanup complete\n")
    
    logger.info("=" * 80)
    logger.info("INDICMOE-4B TOKENIZER TRAINING")
    logger.info("=" * 80)
    logger.info(f"Vocabulary size: {args.vocab_size:,}")
    logger.info(f"Max samples: {args.max_samples:,}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Initialize trainer
    trainer = IndicMoETokenizerTrainer(
        vocab_size=args.vocab_size,
        output_dir=args.output_dir
    )
    
    # Create training corpus
    if not args.skip_corpus_creation:
        corpus_file = trainer.create_training_corpus(
            data_config={},  # Will use default datasets
            max_samples_per_dataset=args.max_samples
        )
    else:
        corpus_file = str(trainer.cache_dir / "tokenizer_training_corpus.txt")
        logger.info(f"Using existing corpus: {corpus_file}")
    
    # Train tokenizer
    tokenizer = trainer.train_tokenizer(corpus_file)
    
    # Save tokenizer
    hf_tokenizer = trainer.save_tokenizer(tokenizer)
    
    # Evaluate tokenizer
    trainer.evaluate_tokenizer(tokenizer)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ TOKENIZER TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Tokenizer saved to: {args.output_dir}")
    logger.info("\nTo use the tokenizer:")
    logger.info(f"  from transformers import AutoTokenizer")
    logger.info(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()

