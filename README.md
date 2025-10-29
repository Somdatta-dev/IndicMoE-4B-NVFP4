# IndicMoE-4B-FP4: Multilingual MoE Language Model

A 4B parameter Mixture-of-Experts language model optimized for Indian languages with strong English and coding capabilities.

## Features

- **Multilingual**: Support for 8 languages (English + 7 major Indian languages)
- **Coding**: Programming knowledge across Python, JavaScript, TypeScript, etc.
- **Efficient**: FP4 quantization with Megatron-LM + Transformer Engine
- **Scalable**: Streaming data pipeline for large-scale training
- **Failsafe**: Automatic checkpointing and resume - never lose progress on multi-day processing

## Quick Start

### 1. Setup Environment

```powershell
# Clone repository
git clone <your-repo>
cd Indic_moe_4b

# Copy and configure environment
cp env.example .env
# Edit .env and add your HuggingFace token
```

### 2. Start Docker Container

```powershell
# Run setup script (Windows)
.\setup.ps1

# Or manually start container
docker compose up -d
docker exec -it indicmoe-training bash
```

### 3. Setup Data Directories

```bash
# Inside container
bash /workspace/setup_data_dirs.sh
```

### 4. Process Training Data

```bash
# Test mode (5000 samples per dataset)
python code/data/pipeline.py --test --phases phase1

# Full processing (Phase 1 only)
python code/data/pipeline.py --phases phase1

# All phases with language selection
python code/data/pipeline.py \
    --phases phase1 phase2 phase3 \
    --languages en hi ta te \
    --batch-size 1000
```

## Data Pipeline

### Architecture

```
HuggingFace Datasets (Streaming)
    ↓
Tokenization (Llama-3.2-1B tokenizer)
    ↓
Checkpointing (every 1,000 samples)
    ↓
Parquet Files (organized by phase/dataset)
    ↓
Training DataLoader
```

### Failsafe Features

The pipeline is designed for **multi-day processing** with zero data loss:

✅ **Automatic checkpointing** every 1,000 samples  
✅ **Resume from any interruption** (network, crash, CTRL+C)  
✅ **Graceful shutdown** handling  
✅ **Automatic retry** on transient failures  
✅ **Atomic file operations** prevent corruption  

See [FAILSAFE_FEATURES.md](FAILSAFE_FEATURES.md) for complete documentation.

### Supported Datasets

**Phase 1: Pre-training** (Multilingual + Code)
- IndicCorpV2: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada
- Wikipedia: Multilingual high-quality text
- mC4: Multilingual web crawl
- FineWeb-Edu: High-quality English educational content
- GitHub Code: Python, JavaScript, TypeScript, HTML, CSS, Java

**Phase 2: Instruction Tuning** (Optional)
- IndicInstruct: Multilingual instruction following
- OpenOrca: English instruction tuning
- CodeAlpaca: Coding instruction tuning

**Phase 3: Function Calling** (Optional)
- Glaive Function Calling v2

### Data Configuration

Edit `code/config.py` to:
- Adjust dataset weights
- Add/remove datasets
- Configure language priorities
- Modify processing parameters

## Project Structure

```
Indic_moe_4b/
├── code/
│   ├── config.py                 # Dataset & processing configuration
│   ├── requirements.txt          # Python dependencies
│   └── data/
│       ├── downloader.py         # Dataset metadata downloader
│       ├── tokenizer_wrapper.py  # Tokenizer with language markers
│       ├── streaming_processor.py # Stream → Tokenize → Parquet
│       └── pipeline.py           # Main orchestrator
├── data/
│   ├── raw/                      # Original dataset metadata
│   ├── processed/                # Tokenized parquet files
│   └── cache/                    # HuggingFace cache
├── docker-compose.yml
├── setup.ps1
├── setup_data_dirs.sh
└── .env
```

## Environment Variables

```bash
# Required
HUGGINGFACE_TOKEN=your_token_here

# Optional (with defaults)
TOKENIZER_NAME=meta-llama/Llama-3.2-1B
MAX_SEQUENCE_LENGTH=2048
BATCH_SIZE=1000
NUM_WORKERS=4
```

## Usage Examples

### Process Specific Languages

```bash
# Only English and Hindi
python code/data/pipeline.py \
    --phases phase1 \
    --languages en hi \
    --max-samples 100000
```

### Test Pipeline with Small Sample

```bash
python code/data/pipeline.py --test
```

### Process All Data (Production)

```bash
python code/data/pipeline.py \
    --phases phase1 phase2 phase3 \
    --batch-size 2000
```

## Output Format

Processed data is saved as Parquet files with the following schema:

```python
{
    'input_ids': List[int],        # Tokenized input
    'attention_mask': List[int],   # Attention mask
    'labels': List[int],           # Labels (same as input_ids for CLM)
    'length': int,                 # Actual sequence length
    'language': str,               # Language code (en, hi, etc.)
    'dataset': str,                # Source dataset name
}
```

## Monitoring

Access services at:
- **TensorBoard**: http://localhost:6006
- **Jupyter Lab**: http://localhost:8888

## Next Steps

After processing data:

1. **Train Model**: Implement training script with Megatron-LM
2. **Evaluation**: Add evaluation scripts for different languages
3. **Deployment**: Export model for inference

## Requirements

- Docker with GPU support (nvidia-docker2)
- NVIDIA GPU with 24GB+ VRAM
- ~500GB storage for processed data
- HuggingFace account with access token

## License

[Your License]

## Citation

```bibtex
@software{indicmoe4b,
  title={IndicMoE-4B-FP4: Multilingual MoE Language Model},
  author={Your Name},
  year={2024}
}
```

