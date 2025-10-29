"""
Utility script to verify and manage checkpoints
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from data.checkpoint_manager import CheckpointManager


def list_checkpoints(checkpoint_dir: str):
    """List all checkpoints with their status"""
    manager = CheckpointManager(checkpoint_dir)
    
    print("\n" + "=" * 100)
    print("CHECKPOINT STATUS")
    print("=" * 100)
    
    checkpoint_files = sorted(Path(checkpoint_dir).glob("*.json"))
    
    if not checkpoint_files:
        print("No checkpoints found.")
        return
    
    for cp_file in checkpoint_files:
        if cp_file.name.endswith('.tmp'):
            continue
        
        try:
            with open(cp_file, 'r') as f:
                data = json.load(f)
            
            status = "✓ Complete" if data.get("completed") else ("✗ Failed" if data.get("failed") else "⏳ In Progress")
            samples = data.get("total_samples_processed", 0)
            tokens = data.get("total_tokens_processed", 0)
            retries = data.get("retry_count", 0)
            
            print(f"\n{status}")
            print(f"  Dataset: {data.get('dataset_name')} ({data.get('subset', 'default')})")
            print(f"  Phase: {data.get('phase')}")
            print(f"  Samples: {samples:,} | Tokens: {tokens:,}")
            print(f"  Files: {data.get('files_created', 0)} | Retries: {retries}")
            
            if data.get("started_at"):
                print(f"  Started: {data['started_at']}")
            if data.get("last_checkpoint_at"):
                print(f"  Last Checkpoint: {data['last_checkpoint_at']}")
            if data.get("error_message"):
                print(f"  Error: {data['error_message']}")
        
        except Exception as e:
            print(f"\n✗ Error reading {cp_file.name}: {e}")
    
    print("\n" + "=" * 100)


def reset_checkpoint(checkpoint_dir: str, dataset_name: str, subset: str = None, phase: str = "phase1"):
    """Reset a checkpoint to retry processing"""
    manager = CheckpointManager(checkpoint_dir)
    checkpoint_key = manager._get_checkpoint_key(dataset_name, subset, phase)
    checkpoint_path = manager._get_checkpoint_path(checkpoint_key)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_key}")
        return
    
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n🔄 Resetting checkpoint: {dataset_name} ({subset})")
        print(f"   Previous state: {data.get('total_samples_processed', 0):,} samples")
        
        # Reset counters
        data["total_samples_processed"] = 0
        data["total_tokens_processed"] = 0
        data["files_created"] = 0
        data["current_file_idx"] = 0
        data["samples_in_current_file"] = 0
        data["completed"] = False
        data["failed"] = False
        data["error_message"] = None
        data["retry_count"] = 0
        data["started_at"] = datetime.now().isoformat()
        
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print("✓ Checkpoint reset successfully")
    
    except Exception as e:
        print(f"❌ Error resetting checkpoint: {e}")


def delete_checkpoint(checkpoint_dir: str, dataset_name: str, subset: str = None, phase: str = "phase1"):
    """Delete a checkpoint file"""
    manager = CheckpointManager(checkpoint_dir)
    checkpoint_key = manager._get_checkpoint_key(dataset_name, subset, phase)
    checkpoint_path = manager._get_checkpoint_path(checkpoint_key)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_key}")
        return
    
    try:
        checkpoint_path.unlink()
        print(f"✓ Deleted checkpoint: {dataset_name} ({subset})")
    except Exception as e:
        print(f"❌ Error deleting checkpoint: {e}")


def clear_failed_checkpoints(checkpoint_dir: str):
    """Delete all failed checkpoints"""
    checkpoint_files = Path(checkpoint_dir).glob("*.json")
    deleted = 0
    
    for cp_file in checkpoint_files:
        if cp_file.name.endswith('.tmp'):
            continue
        
        try:
            with open(cp_file, 'r') as f:
                data = json.load(f)
            
            if data.get("failed"):
                cp_file.unlink()
                print(f"✓ Deleted failed checkpoint: {cp_file.name}")
                deleted += 1
        
        except Exception as e:
            print(f"⚠️  Error processing {cp_file.name}: {e}")
    
    print(f"\n✓ Deleted {deleted} failed checkpoint(s)")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Manager Utility")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/workspace/data/processed/checkpoints",
        help="Checkpoint directory"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    subparsers.add_parser("list", help="List all checkpoints")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset a checkpoint")
    reset_parser.add_argument("dataset_name", help="Dataset name")
    reset_parser.add_argument("--subset", help="Dataset subset")
    reset_parser.add_argument("--phase", default="phase1", help="Processing phase")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a checkpoint")
    delete_parser.add_argument("dataset_name", help="Dataset name")
    delete_parser.add_argument("--subset", help="Dataset subset")
    delete_parser.add_argument("--phase", default="phase1", help="Processing phase")
    
    # Clear failed command
    subparsers.add_parser("clear-failed", help="Delete all failed checkpoints")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_checkpoints(args.checkpoint_dir)
    elif args.command == "reset":
        reset_checkpoint(args.checkpoint_dir, args.dataset_name, args.subset, args.phase)
    elif args.command == "delete":
        delete_checkpoint(args.checkpoint_dir, args.dataset_name, args.subset, args.phase)
    elif args.command == "clear-failed":
        clear_failed_checkpoints(args.checkpoint_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

