#!/usr/bin/env python3
"""
Generate LaTeX table comparing OCR cache ablation study results.
Outputs a publication-ready table with average and peak metrics.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


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


def calculate_statistics(steps):
    """
    Calculate statistics from step data, including growth rates from linear regression.
    
    Args:
        steps: List of step dictionaries
    
    Returns:
        Dictionary with statistics including growth rates
    """
    if not steps:
        return None
    
    batch_times = [s['batch_time_ms'] for s in steps]
    cache_memory = [s['cache_memory_mb'] for s in steps]
    step_numbers = [s['step'] for s in steps]
    
    # Filter steps 1-49 for growth calculation (as mentioned in caption)
    filtered_indices = [i for i, step_num in enumerate(step_numbers) if 1 <= step_num <= 49]
    
    # Calculate growth rates using linear regression on steps 1-49
    batch_time_grow = 0.0
    cache_memory_grow = 0.0
    if len(filtered_indices) > 1:
        filtered_steps = [step_numbers[i] for i in filtered_indices]
        filtered_batch_times = [batch_times[i] for i in filtered_indices]
        filtered_cache_memory = [cache_memory[i] for i in filtered_indices]
        
        # Linear regression for batch time growth
        coeffs = np.polyfit(filtered_steps, filtered_batch_times, 1)
        batch_time_grow = coeffs[0]
        
        # Linear regression for cache memory growth
        if any(filtered_cache_memory):  # Only if there's non-zero cache memory
            coeffs = np.polyfit(filtered_steps, filtered_cache_memory, 1)
            cache_memory_grow = coeffs[0]
    
    stats = {
        'avg_batch_time_ms': sum(batch_times) / len(batch_times),
        'avg_cache_memory_mb': sum(cache_memory) / len(cache_memory),
        'peak_cache_memory_mb': max(cache_memory) if cache_memory else 0.0,
        'batch_time_grow_ms_per_step': batch_time_grow,
        'cache_memory_grow_mb_per_step': cache_memory_grow,
        'num_steps': len(steps),
    }
    
    return stats


def generate_latex_table(all_stats, output_file=None):
    """
    Generate LaTeX table code matching the reference format.
    
    Args:
        all_stats: Dictionary mapping cache_mode to statistics
        output_file: Optional file path to save the table
    """
    
    # Define display names
    display_names = {
        'none': 'No Cache',
        'naive': 'Naive Cache',
        'segment': 'Ours'
    }
    
    # Order for table rows
    order = ['none', 'naive', 'segment']
    
    # Calculate baselines for speedup and memory savings
    baseline_none = all_stats.get('none')
    baseline_naive = all_stats.get('naive')
    
    # Find best values for bold formatting
    # Best batch time (lowest) - only for cache methods
    cache_methods = [m for m in order if m != 'none']
    best_avg_batch = min([all_stats[m]['avg_batch_time_ms'] for m in cache_methods if m in all_stats], default=None)
    best_batch_grow = min([all_stats[m]['batch_time_grow_ms_per_step'] for m in cache_methods if m in all_stats], default=None)
    best_peak_mem = min([all_stats[m]['peak_cache_memory_mb'] for m in cache_methods if m in all_stats], default=None)
    best_mem_grow = min([all_stats[m]['cache_memory_grow_mb_per_step'] for m in cache_methods if m in all_stats], default=None)
    
    # Generate LaTeX code
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Cache mechanism ablation. \\emph{Growth/step} is the slope from a least-squares linear fit over steps 1--49. \\emph{Speedup} is relative to ``no cache''. \\emph{Mem Saving} is relative to the peak cache memory of ``naive cache''.}")
    latex_lines.append("\\label{tab:cache_ablation}")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{lccccc c}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\multirow{2}{*}{\\textbf{Method}} &")
    latex_lines.append("\\multicolumn{2}{c}{\\textbf{Render Time}$\\downarrow$} &")
    latex_lines.append("\\multirow{2}{*}{\\textbf{Speedup}$\\uparrow$} &")
    latex_lines.append("\\multicolumn{2}{c}{\\textbf{Cache Mem}$\\downarrow$} &")
    latex_lines.append("\\multirow{2}{*}{\\textbf{Mem Save}$\\uparrow$} \\\\")
    latex_lines.append("\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}")
    latex_lines.append("& \\textbf{Avg (ms)} & \\textbf{Grow (ms/step)} & &")
    latex_lines.append("\\textbf{Peak (MB)} & \\textbf{Grow (MB/step)} & \\\\")
    latex_lines.append("\\midrule")
    
    # Add data rows
    for mode in order:
        if mode not in all_stats:
            continue
        
        stats = all_stats[mode]
        method_name = display_names.get(mode, mode)
        
        # Calculate speedup (relative to "no cache")
        if baseline_none and mode != 'none':
            speedup = baseline_none['avg_batch_time_ms'] / stats['avg_batch_time_ms']
            speedup_str = f"{speedup:.2f}x"
        elif mode == 'none':
            speedup_str = "1.00x"
        else:
            speedup_str = "--"
        
        # Calculate memory savings (relative to naive cache peak)
        if baseline_naive and mode != 'none':
            if mode == 'naive':
                mem_save_str = "0.00\\%"
            else:
                mem_save = ((baseline_naive['peak_cache_memory_mb'] - stats['peak_cache_memory_mb']) / 
                           baseline_naive['peak_cache_memory_mb']) * 100
                mem_save_str = f"{mem_save:.2f}\\%"
        else:
            mem_save_str = "--"
        
        # Format batch time
        avg_batch = stats['avg_batch_time_ms']
        avg_batch_str = f"{avg_batch:.2f}"
        if mode != 'none' and best_avg_batch and abs(avg_batch - best_avg_batch) < 0.01:
            avg_batch_str = f"\\textbf{{{avg_batch_str}}}"
        
        # Format batch time growth
        batch_grow = stats.get('batch_time_grow_ms_per_step', 0.0)
        if mode == 'none':
            batch_grow_str = f"{batch_grow:.2f}"
        else:
            batch_grow_str = f"{batch_grow:.2f}"
            if best_batch_grow and abs(batch_grow - best_batch_grow) < 0.01:
                batch_grow_str = f"\\textbf{{{batch_grow_str}}}"
        
        # Format peak cache memory
        if mode == 'none':
            peak_mem_str = "--"
            mem_grow_str = "--"
        else:
            peak_mem = stats['peak_cache_memory_mb']
            peak_mem_str = f"{peak_mem:.2f}"
            if best_peak_mem and abs(peak_mem - best_peak_mem) < 0.01:
                peak_mem_str = f"\\textbf{{{peak_mem_str}}}"
            
            # Format cache memory growth
            mem_grow = stats.get('cache_memory_grow_mb_per_step', 0.0)
            mem_grow_str = f"{mem_grow:.2f}"
            if best_mem_grow and abs(mem_grow - best_mem_grow) < 0.01:
                mem_grow_str = f"\\textbf{{{mem_grow_str}}}"
        
        # Format speedup
        if mode != 'none' and baseline_none:
            if best_avg_batch and abs(avg_batch - best_avg_batch) < 0.01:
                speedup_str = f"\\textbf{{{speedup_str}}}"
        
        # Format memory save
        if mode == 'segment' and baseline_naive:
            mem_save = ((baseline_naive['peak_cache_memory_mb'] - stats['peak_cache_memory_mb']) / 
                       baseline_naive['peak_cache_memory_mb']) * 100
            mem_save_str = f"\\textbf{{{mem_save:.2f}\\%}}"
        
        # Format row - add rowcolor for "Ours" and match reference spacing
        if mode == 'segment':
            latex_lines.append("\\rowcolor{OursBlue}")
            method_padded = f"{method_name}        "  # "Ours        " (12 chars total)
        elif mode == 'naive':
            method_padded = f"{method_name} "  # "Naive Cache " (12 chars total)
        else:
            method_padded = f"{method_name}    "  # "No Cache    " (12 chars total)
        
        # Build the row matching reference format exactly
        if mode == 'none':
            # "No Cache    & 3876.79 & 124.71 & 1.00x  & --      & --     & -- \\"
            latex_lines.append(f"{method_padded} & {avg_batch_str} & {batch_grow_str} & {speedup_str}  & {peak_mem_str}      & {mem_grow_str}     & {mem_save_str} \\\\")
        else:
            # "Naive Cache & \textbf{369.35} & 0.57 & \textbf{10.50x} & 4917.65 & 102.70 & 0.00\% \\"
            # "Ours        & 374.91 & \textbf{0.50} & 10.34x & \textbf{189.91} & \textbf{3.57} & \textbf{96.14\%} \\"
            latex_lines.append(f"{method_padded} & {avg_batch_str} & {batch_grow_str} & {speedup_str} & {peak_mem_str} & {mem_grow_str} & {mem_save_str} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")
    
    latex_code = "\n".join(latex_lines)
    
    # Print to console
    print("\n" + "="*70)
    print("LaTeX Table Code (Reference Format):")
    print("="*70)
    print(latex_code)
    print("="*70 + "\n")
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_code)
        print(f"Table saved to: {output_file}\n")
    
    return latex_code


def main():
    """Main function to process data and generate tables."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LaTeX table from OCR ablation study results')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing JSONL files (default: logs/ocr_stats)')
    parser.add_argument('--output', type=str, default='ocr_ablation_table.tex',
                       help='Output LaTeX file (default: ocr_ablation_table.tex)')
    
    args = parser.parse_args()
    
    # Find all JSONL files
    if args.data_dir is None:
        # Default to logs/ocr_stats relative to script location
        data_dir = Path(__file__).parent.parent.parent / "logs" / "ocr_stats"
    else:
        data_dir = Path(args.data_dir)
    jsonl_files = list(data_dir.glob('ocr_stats_*.jsonl'))
    
    if len(jsonl_files) == 0:
        print(f"Error: No JSONL files found in {data_dir}")
        sys.exit(1)
    
    # Load data from all files
    all_stats = {}
    for filepath in sorted(jsonl_files):
        cache_mode, steps = load_jsonl_data(filepath)
        if cache_mode and steps:
            stats = calculate_statistics(steps)
            if stats:
                all_stats[cache_mode] = stats
                print(f"Loaded {stats['num_steps']} steps from {filepath.name} (mode: {cache_mode})")
                print(f"  Avg Batch Time: {stats['avg_batch_time_ms']:.2f} ms")
                print(f"  Batch Time Growth: {stats['batch_time_grow_ms_per_step']:.4f} ms/step")
                print(f"  Peak Cache Memory: {stats['peak_cache_memory_mb']:.2f} MB")
                print(f"  Cache Memory Growth: {stats['cache_memory_grow_mb_per_step']:.4f} MB/step")
    
    if len(all_stats) == 0:
        print("Error: No valid data found")
        sys.exit(1)
    
    # Generate table
    output_path = data_dir / args.output
    generate_latex_table(all_stats, output_file=str(output_path))


if __name__ == '__main__':
    main()

