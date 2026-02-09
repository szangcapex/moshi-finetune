#!/usr/bin/env python3
"""
Modify Moshi training config file's batch_size based on GPU type.
Only modifies necessary lines, preserves all formatting and comments.
"""

import sys
import re
from pathlib import Path


def modify_yaml_config(yaml_path: str, gpu_type: str):
    """
    Modify batch_size and max_steps in YAML config file.
    Preserves original formatting and comments.
    
    Args:
        yaml_path: Path to YAML config file
        gpu_type: GPU type ('4090' or 'H100')
    """
    # Validate input
    if gpu_type.upper() not in ['4090', 'H100']:
        print(f"Error: Unsupported GPU type '{gpu_type}'")
        print("Supported types: 4090, H100")
        sys.exit(1)
    
    # Check if file exists
    config_path = Path(yaml_path)
    if not config_path.exists():
        print(f"Error: File not found: {yaml_path}")
        sys.exit(1)
    
    # Set batch_size based on GPU type
    gpu_config = {
        '4090': {'batch_size': 1, 'description': 'NVIDIA RTX 4090'},
        'H100': {'batch_size': 2, 'description': 'NVIDIA H100'}
    }
    
    gpu_type_upper = gpu_type.upper()
    new_batch_size = gpu_config[gpu_type_upper]['batch_size']
    gpu_desc = gpu_config[gpu_type_upper]['description']
    
    # Calculate new max_steps (keep total samples constant)
    # Baseline: batch_size=16, max_steps=2000 => 32000 total samples
    baseline_total_samples = 16 * 2000
    baseline_batch_size = 16
    new_max_steps = baseline_total_samples // new_batch_size
    
    # Calculate new checkpoint and eval frequencies (keep same sample intervals)
    # Original: ckpt_freq=100, eval_freq=100 with batch_size=16
    # => checkpoint every 100*16=1600 samples
    new_ckpt_freq = 100 * baseline_batch_size // new_batch_size
    new_eval_freq = 100 * baseline_batch_size // new_batch_size
    
    # Read file content
    print(f"Reading config file: {yaml_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract original values for reporting
    old_batch_size = re.search(r'^batch_size:\s*(\d+)', content, re.MULTILINE)
    old_max_steps = re.search(r'^max_steps:\s*(\d+)', content, re.MULTILINE)
    old_ckpt_freq = re.search(r'^ckpt_freq:\s*(\d+)', content, re.MULTILINE)
    old_eval_freq = re.search(r'^eval_freq:\s*(\d+)', content, re.MULTILINE)
    
    old_batch_size_val = old_batch_size.group(1) if old_batch_size else 'N/A'
    old_max_steps_val = old_max_steps.group(1) if old_max_steps else 'N/A'
    old_ckpt_freq_val = old_ckpt_freq.group(1) if old_ckpt_freq else 'N/A'
    old_eval_freq_val = old_eval_freq.group(1) if old_eval_freq else 'N/A'
    
    # Backup original file
    backup_path = config_path.with_suffix('.yaml.bak')
    print(f"Backing up to: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Replace only batch_size, max_steps, ckpt_freq, and eval_freq lines (preserve formatting)
    content = re.sub(
        r'^batch_size:\s*\d+',
        f'batch_size: {new_batch_size}',
        content,
        flags=re.MULTILINE
    )
    content = re.sub(
        r'^max_steps:\s*\d+',
        f'max_steps: {new_max_steps}',
        content,
        flags=re.MULTILINE
    )
    content = re.sub(
        r'^ckpt_freq:\s*\d+',
        f'ckpt_freq: {new_ckpt_freq}',
        content,
        flags=re.MULTILINE
    )
    content = re.sub(
        r'^eval_freq:\s*\d+',
        f'eval_freq: {new_eval_freq}',
        content,
        flags=re.MULTILINE
    )
    
    # Write modified content
    print(f"Writing modified config to: {yaml_path}")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Print modification summary
    print("\n" + "=" * 60)
    print(f"Config modified - GPU type: {gpu_desc}")
    print("=" * 60)
    print(f"batch_size:   {old_batch_size_val} → {new_batch_size}")
    print(f"max_steps:    {old_max_steps_val} → {new_max_steps}")
    print(f"ckpt_freq:    {old_ckpt_freq_val} → {new_ckpt_freq}")
    print(f"eval_freq:    {old_eval_freq_val} → {new_eval_freq}")
    print(f"\nTotal training samples: {new_batch_size * new_max_steps:,}")
    print(f"Checkpoint interval: every {new_ckpt_freq * new_batch_size:,} samples")
    print("=" * 60)


def main():
    if len(sys.argv) != 3:
        print("Usage: python modify_moshi_config.py <yaml_path> <gpu_type>")
        print("\nArguments:")
        print("  yaml_path  - Path to YAML config file (e.g., example/moshi_7B.yaml)")
        print("  gpu_type   - GPU type (4090 or H100)")
        print("\nExamples:")
        print("  python modify_moshi_config.py example/moshi_7B.yaml 4090")
        print("  python modify_moshi_config.py example/moshi_7B.yaml H100")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    gpu_type = sys.argv[2]
    
    modify_yaml_config(yaml_path, gpu_type)


if __name__ == "__main__":
    main()