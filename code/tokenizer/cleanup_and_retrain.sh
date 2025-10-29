#!/bin/bash
# Cleanup bad tokenizer training and retrain with fixes

echo "🧹 Cleaning up previous training..."
rm -rf /workspace/tokenizers/indicmoe_tokenizer
rm -f /workspace/data/cache/tokenizer_training_corpus.txt

echo "✓ Cleanup complete"
echo ""
echo "🚀 Starting fresh tokenizer training..."
echo ""

# Make sure HuggingFace token is set
if [ -z "$HUGGINGFACE_TOKEN" ] || [ "$HUGGINGFACE_TOKEN" = "your_token_here" ]; then
    echo "❌ ERROR: HUGGINGFACE_TOKEN not set!"
    echo "Please set your token in .env file"
    exit 1
fi

# Train with corrected script
python code/tokenizer/train_custom_tokenizer.py \
    --vocab-size 128000 \
    --max-samples 1000000 \
    --output-dir /workspace/tokenizers/indicmoe_128k

echo ""
echo "✓ Tokenizer training complete!"

