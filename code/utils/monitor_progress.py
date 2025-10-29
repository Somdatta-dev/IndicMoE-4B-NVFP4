"""
Real-time progress monitoring for data pipeline
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

sys.path.append(str(Path(__file__).parent.parent))


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds to human-readable duration"""
    return str(timedelta(seconds=int(seconds)))


def load_checkpoints(checkpoint_dir: str) -> Dict:
    """Load all checkpoint files"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = {}
    
    for cp_file in checkpoint_dir.glob("*.json"):
        if cp_file.name.endswith('.tmp'):
            continue
        
        try:
            with open(cp_file, 'r') as f:
                data = json.load(f)
                checkpoints[cp_file.stem] = data
        except Exception:
            pass
    
    return checkpoints


def calculate_stats(checkpoints: Dict) -> Dict:
    """Calculate overall statistics"""
    total_samples = sum(cp.get("total_samples_processed", 0) for cp in checkpoints.values())
    total_tokens = sum(cp.get("total_tokens_processed", 0) for cp in checkpoints.values())
    
    completed = sum(1 for cp in checkpoints.values() if cp.get("completed", False))
    failed = sum(1 for cp in checkpoints.values() if cp.get("failed", False))
    in_progress = len(checkpoints) - completed - failed
    
    # Calculate processing speed
    total_time = 0
    for cp in checkpoints.values():
        if cp.get("started_at") and cp.get("last_checkpoint_at"):
            start = datetime.fromisoformat(cp["started_at"])
            end = datetime.fromisoformat(cp["last_checkpoint_at"])
            total_time += (end - start).total_seconds()
    
    samples_per_sec = total_samples / total_time if total_time > 0 else 0
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "total_datasets": len(checkpoints),
        "completed": completed,
        "failed": failed,
        "in_progress": in_progress,
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "total_time": total_time
    }


def print_progress_report(checkpoint_dir: str, data_dir: str):
    """Print comprehensive progress report"""
    checkpoints = load_checkpoints(checkpoint_dir)
    stats = calculate_stats(checkpoints)
    
    # Calculate disk usage
    data_path = Path(data_dir)
    total_size = sum(f.stat().st_size for f in data_path.rglob('*.parquet') if f.is_file())
    num_parquet_files = len(list(data_path.rglob('*.parquet')))
    
    # Print report
    print("\n" + "=" * 100)
    print("DATA PIPELINE PROGRESS REPORT")
    print("=" * 100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Dataset Status
    print("📊 DATASET STATUS")
    print("-" * 100)
    print(f"  Total Datasets: {stats['total_datasets']}")
    print(f"  ✅ Completed: {stats['completed']}")
    print(f"  ⏳ In Progress: {stats['in_progress']}")
    print(f"  ❌ Failed: {stats['failed']}")
    print()
    
    # Processing Stats
    print("📈 PROCESSING STATS")
    print("-" * 100)
    print(f"  Total Samples: {stats['total_samples']:,}")
    print(f"  Total Tokens: {stats['total_tokens']:,}")
    print(f"  Processing Time: {format_duration(stats['total_time'])}")
    print(f"  Speed: {stats['samples_per_sec']:.1f} samples/sec, {stats['tokens_per_sec']:,.0f} tokens/sec")
    print()
    
    # Storage Stats
    print("💾 STORAGE STATS")
    print("-" * 100)
    print(f"  Parquet Files: {num_parquet_files}")
    print(f"  Total Size: {format_size(total_size)}")
    print(f"  Avg File Size: {format_size(total_size / num_parquet_files) if num_parquet_files > 0 else '0 B'}")
    print()
    
    # Active Datasets
    if stats['in_progress'] > 0:
        print("🔄 IN PROGRESS DATASETS")
        print("-" * 100)
        
        for name, cp in checkpoints.items():
            if not cp.get("completed", False) and not cp.get("failed", False):
                samples = cp.get("total_samples_processed", 0)
                tokens = cp.get("total_tokens_processed", 0)
                files = cp.get("files_created", 0)
                
                # Calculate time elapsed
                if cp.get("started_at"):
                    start = datetime.fromisoformat(cp["started_at"])
                    elapsed = (datetime.now() - start).total_seconds()
                    elapsed_str = format_duration(elapsed)
                else:
                    elapsed_str = "Unknown"
                
                print(f"  📦 {cp.get('dataset_name', 'Unknown')} ({cp.get('subset', 'default')})")
                print(f"     Samples: {samples:,} | Tokens: {tokens:,} | Files: {files}")
                print(f"     Elapsed: {elapsed_str}")
                print()
    
    # Failed Datasets
    if stats['failed'] > 0:
        print("❌ FAILED DATASETS")
        print("-" * 100)
        
        for name, cp in checkpoints.items():
            if cp.get("failed", False):
                error = cp.get("error_message", "Unknown error")
                retries = cp.get("retry_count", 0)
                
                print(f"  ❌ {cp.get('dataset_name', 'Unknown')} ({cp.get('subset', 'default')})")
                print(f"     Error: {error}")
                print(f"     Retries: {retries}/3")
                print()
    
    print("=" * 100)
    print()


def watch_progress(checkpoint_dir: str, data_dir: str, interval: int = 30):
    """Watch progress in real-time"""
    try:
        while True:
            # Clear screen (works on Unix and Windows)
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print_progress_report(checkpoint_dir, data_dir)
            print(f"Refreshing every {interval} seconds... (Press CTRL+C to exit)")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor data pipeline progress")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/workspace/data/processed/checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/data/processed",
        help="Processed data directory"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode (auto-refresh)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    if args.watch:
        watch_progress(args.checkpoint_dir, args.data_dir, args.interval)
    else:
        print_progress_report(args.checkpoint_dir, args.data_dir)


if __name__ == "__main__":
    main()

