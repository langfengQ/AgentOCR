#!/usr/bin/env python3
"""
Plot ablation study results comparing three cache modes.
Generates publication-quality figures for batch_time_ms and cache_memory_mb.

Requirements:
    pip install matplotlib numpy

Usage:
    python plot_ablation.py
    python plot_ablation.py --separate  # Save separate figures
    python plot_ablation.py --data-dir /path/to/data --output results.pdf
"""

import json
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print("Error: matplotlib is required but not installed.")
    print("Please install it with: pip install matplotlib numpy")
    sys.exit(1)

# Use a publication-quality style
plt.style.use('seaborn-v0_8-paper')
# Alternative: plt.style.use('seaborn-v0_8-whitegrid')

# Define beautiful color palette (suitable for papers)
# Using a professional color scheme that works well in both color and grayscale
COLORS = {
    'none': '#E74C3C',        # Red - vibrant, stands out
    'naive': '#3498DB',       # Blue - professional
    'segment': '#2ECC71'     # Green - success/optimization
}

# Alternative color scheme (more muted, academic)
# COLORS = {
#     'none': '#D32F2F',        # Deep red
#     'full_image': '#1976D2',  # Deep blue
#     'master': '#388E3C'      # Deep green
# }

# Line styles for better distinction
LINE_STYLES = {
    'none': '-',
    'naive': '--',
    'segment': '-.'
}

# Labels for legend
LABELS = {
    'none': 'No Cache',
    'naive': 'Naive Cache',
    'segment': 'Incremental Optical Cache'
}


def load_jsonl_data(filepath):
    """
    Load data from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
    
    Returns:
        Tuple of (cache_mode, list of step data dictionaries)
    """
    steps = []
    cache_mode = None
    
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('_type') == 'metadata':
                cache_mode = data.get('cache_mode', 'unknown')
            elif data.get('_type') == 'step':
                steps.append(data)
    
    return cache_mode, steps


def extract_metrics(steps):
    """
    Extract metrics from step data.
    
    Args:
        steps: List of step dictionaries
    
    Returns:
        Tuple of (steps, batch_times, cache_memory)
    """
    steps_list = [s['step'] for s in steps]
    batch_times = [s['batch_time_ms'] for s in steps]
    cache_memory = [s['cache_memory_mb'] for s in steps]
    
    return steps_list, batch_times, cache_memory


def plot_ablation_results(data_dir=None, output_file='ocr_ablation_results.pdf', 
                          figsize=(12, 5), dpi=300):
    """
    Plot ablation study results comparing three cache modes.
    
    Args:
        data_dir: Directory containing the JSONL files (default: logs/ocr_stats)
        output_file: Output filename for the figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output figure
    """
    # Find all JSONL files
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "logs" / "ocr_stats"
    else:
        data_dir = Path(data_dir)
    jsonl_files = list(data_dir.glob('ocr_stats_*.jsonl'))
    
    if len(jsonl_files) == 0:
        print(f"Error: No JSONL files found in {data_dir}")
        return
    
    # Load data from all files
    all_data = {}
    for filepath in sorted(jsonl_files):
        cache_mode, steps = load_jsonl_data(filepath)
        if cache_mode and steps:
            all_data[cache_mode] = steps
            print(f"Loaded {len(steps)} steps from {filepath.name} (mode: {cache_mode})")
    
    if len(all_data) == 0:
        print("Error: No valid data found")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # Plot order: none, naive, segment (for consistent legend order)
    plot_order = ['none', 'naive', 'segment']
    
    # Plot 1: Batch Time
    for mode in plot_order:
        if mode not in all_data:
            continue
        
        steps_list, batch_times, _ = extract_metrics(all_data[mode])
        
        # Use fewer markers if there are many data points
        marker_freq = max(1, len(steps_list) // 20) if len(steps_list) > 20 else 1
        
        ax1.plot(steps_list, batch_times, 
                color=COLORS[mode], 
                linestyle=LINE_STYLES[mode],
                linewidth=2.5,
                marker='o' if len(steps_list) <= 50 else None,
                markersize=4 if len(steps_list) <= 50 else 0,
                markerfacecolor=COLORS[mode],
                markeredgecolor='white',
                markeredgewidth=0.5,
                label=LABELS[mode],
                alpha=0.9,
                zorder=3)
    
    ax1.set_xlabel('Step', fontsize=13, fontweight='medium')
    ax1.set_ylabel('Batch Time (ms)', fontsize=13, fontweight='medium')
    ax1.set_title('Batch Processing Time', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    legend1 = ax1.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')
    legend1.get_frame().set_linewidth(0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Cache Memory
    for mode in plot_order:
        if mode not in all_data:
            continue
        
        steps_list, _, cache_memory = extract_metrics(all_data[mode])
        
        ax2.plot(steps_list, cache_memory,
                color=COLORS[mode],
                linestyle=LINE_STYLES[mode],
                linewidth=2.5,
                marker='s' if len(steps_list) <= 50 else None,
                markersize=4 if len(steps_list) <= 50 else 0,
                markerfacecolor=COLORS[mode],
                markeredgecolor='white',
                markeredgewidth=0.5,
                label=LABELS[mode],
                alpha=0.9,
                zorder=3)
    
    ax2.set_xlabel('Step', fontsize=13, fontweight='medium')
    ax2.set_ylabel('Cache Memory (MB)', fontsize=13, fontweight='medium')
    ax2.set_title('Cache Memory Usage', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    legend2 = ax2.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')
    legend2.get_frame().set_linewidth(0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout(pad=2.5)
    
    # Save figure
    output_path = data_dir / output_file
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nFigure saved to: {output_path}")
    
    # Also save as PNG for easy viewing
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Figure saved to: {png_path}")
    
    plt.close()


def plot_ablation_results_separate(data_dir=None, output_dir=None, 
                                   figsize=(10, 6), dpi=300):
    """
    Plot ablation study results in separate figures (one for each metric).
    
    Args:
        data_dir: Directory containing the JSONL files (default: logs/ocr_stats)
        output_dir: Directory to save output figures (default: same as data_dir)
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output figure
    """
    # Find all JSONL files
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "logs" / "ocr_stats"
    else:
        data_dir = Path(data_dir)
    
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = list(data_dir.glob('ocr_stats_*.jsonl'))
    
    if len(jsonl_files) == 0:
        print(f"Error: No JSONL files found in {data_dir}")
        return
    
    # Load data from all files
    all_data = {}
    for filepath in sorted(jsonl_files):
        cache_mode, steps = load_jsonl_data(filepath)
        if cache_mode and steps:
            all_data[cache_mode] = steps
            print(f"Loaded {len(steps)} steps from {filepath.name} (mode: {cache_mode})")
    
    if len(all_data) == 0:
        print("Error: No valid data found")
        return
    
    plot_order = ['none', 'naive', 'segment']
    
    # Plot 1: Batch Time
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig1.patch.set_facecolor('white')
    
    for mode in plot_order:
        if mode not in all_data:
            continue
        
        steps_list, batch_times, _ = extract_metrics(all_data[mode])
        
        ax1.plot(steps_list, batch_times,
                color=COLORS[mode],
                linestyle=LINE_STYLES[mode],
                linewidth=2.5,
                marker='o' if len(steps_list) <= 50 else None,
                markersize=5 if len(steps_list) <= 50 else 0,
                markerfacecolor=COLORS[mode],
                markeredgecolor='white',
                markeredgewidth=0.8,
                label=LABELS[mode],
                alpha=0.9,
                zorder=3)
    
    ax1.set_xlabel('Step', fontsize=14, fontweight='medium')
    ax1.set_ylabel('Batch Time (ms)', fontsize=14, fontweight='medium')
    ax1.set_title('Batch Processing Time Comparison', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    legend1 = ax1.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='gray')
    legend1.get_frame().set_linewidth(0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path1 = output_dir / 'ocr_ablation_batch_time.pdf'
    fig1.savefig(output_path1, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    png_path1 = output_path1.with_suffix('.png')
    fig1.savefig(png_path1, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved batch time figure to: {output_path1}")
    plt.close(fig1)
    
    # Plot 2: Cache Memory
    fig2, ax2 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig2.patch.set_facecolor('white')
    
    for mode in plot_order:
        if mode not in all_data:
            continue
        
        steps_list, _, cache_memory = extract_metrics(all_data[mode])
        
        ax2.plot(steps_list, cache_memory,
                color=COLORS[mode],
                linestyle=LINE_STYLES[mode],
                linewidth=2.5,
                marker='s' if len(steps_list) <= 50 else None,
                markersize=5 if len(steps_list) <= 50 else 0,
                markerfacecolor=COLORS[mode],
                markeredgecolor='white',
                markeredgewidth=0.8,
                label=LABELS[mode],
                alpha=0.9,
                zorder=3)
    
    ax2.set_xlabel('Step', fontsize=14, fontweight='medium')
    ax2.set_ylabel('Cache Memory (MB)', fontsize=14, fontweight='medium')
    ax2.set_title('Cache Memory Usage Comparison', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    legend2 = ax2.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='gray')
    legend2.get_frame().set_linewidth(0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path2 = output_dir / 'ocr_ablation_cache_memory.pdf'
    fig2.savefig(output_path2, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    png_path2 = output_path2.with_suffix('.png')
    fig2.savefig(png_path2, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved cache memory figure to: {output_path2}")
    plt.close(fig2)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot OCR ablation study results')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing JSONL files (default: logs/ocr_stats)')
    parser.add_argument('--output', type=str, default='ocr_ablation_results.pdf',
                       help='Output filename (default: ocr_ablation_results.pdf)')
    parser.add_argument('--separate', action='store_true',
                       help='Save separate figures for each metric')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure resolution (default: 300)')
    
    args = parser.parse_args()
    
    if args.separate:
        plot_ablation_results_separate(data_dir=args.data_dir, output_dir=args.data_dir, dpi=args.dpi)
    else:
        plot_ablation_results(data_dir=args.data_dir, output_file=args.output, dpi=args.dpi)

