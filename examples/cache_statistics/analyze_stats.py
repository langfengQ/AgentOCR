import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_jsonl_data(path: str | Path) -> Tuple[Dict, List[Dict]]:
    """加载JSONL文件，返回metadata和step数据列表"""
    path = Path(path)
    meta = {}
    steps = []
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("_type") == "metadata":
                meta = obj
            elif obj.get("_type") == "step":
                steps.append(obj)
    
    return meta, steps


def calculate_metrics(meta: Dict, steps: List[Dict]) -> Dict:
    """计算各种指标"""
    if not steps:
        raise ValueError("No step records found")
    
    # 提取数据
    batch_times = [float(s["batch_time_ms"]) for s in steps]
    cache_mems = [float(s["cache_memory_mb"]) for s in steps]
    steps_num = [int(s["step"]) for s in steps]
    
    # 提取所有计时组件
    render_times = [float(s.get("render_time_ms", 0)) for s in steps]
    cache_lookup_times = [float(s.get("cache_lookup_time_ms", 0)) for s in steps]
    cache_update_times = [float(s.get("cache_update_time_ms", 0)) for s in steps]
    array_conversion_times = [float(s.get("array_conversion_time_ms", 0)) for s in steps]
    array_append_times = [float(s.get("array_append_time_ms", 0)) for s in steps]
    text_preprocess_times = [float(s.get("text_preprocess_time_ms", 0)) for s in steps]
    image_assembly_times = [float(s.get("image_assembly_time_ms", 0)) for s in steps]
    batch_preprocess_times = [float(s.get("batch_preprocess_time_ms", 0)) for s in steps]
    postprocess_times = [float(s.get("postprocess_time_ms", 0)) for s in steps]
    cache_loop_times = [float(s.get("cache_loop_time_ms", 0)) for s in steps]
    
    # 1. 平均batch_time_ms
    avg_batch_time_ms = np.mean(batch_times)
    
    # 2. batch_time_ms grow (ms/step) - 使用线性回归计算增长率
    if len(steps_num) > 1:
        # 使用线性回归计算斜率
        coeffs = np.polyfit(steps_num, batch_times, 1)
        batch_time_grow_ms_per_step = coeffs[0]
    else:
        batch_time_grow_ms_per_step = 0.0
    
    # 3. peak Cache Memory
    peak_cache_memory_mb = max(cache_mems) if cache_mems else 0.0
    
    # 4. Cache Memory Grow (MB/step) - 使用线性回归计算增长率
    if len(steps_num) > 1:
        coeffs = np.polyfit(steps_num, cache_mems, 1)
        cache_memory_grow_mb_per_step = coeffs[0]
    else:
        cache_memory_grow_mb_per_step = 0.0
    
    # 5. 计算 batch_time 的完整 breakdown
    avg_render = np.mean(render_times)
    avg_cache_lookup = np.mean(cache_lookup_times)
    avg_cache_update = np.mean(cache_update_times)
    avg_array_conversion = np.mean(array_conversion_times)
    avg_array_append = np.mean(array_append_times)
    avg_text_preprocess = np.mean(text_preprocess_times)
    avg_image_assembly = np.mean(image_assembly_times)
    avg_batch_preprocess = np.mean(batch_preprocess_times)
    avg_postprocess = np.mean(postprocess_times)
    avg_cache_loop = np.mean(cache_loop_times)
    
    # 计算已统计时间和未统计时间
    # 注意：cache_loop_time 包含了 render, cache_lookup, cache_update, text_preprocess
    # 所以我们有两种计算方式：
    # 方式1 (细粒度): 所有独立计时的总和
    avg_accounted_detailed = (avg_render + avg_cache_lookup + avg_cache_update + 
                              avg_array_conversion + avg_array_append + 
                              avg_text_preprocess + avg_image_assembly +
                              avg_batch_preprocess + avg_postprocess)
    # 方式2 (粗粒度): 使用 cache_loop_time (包含内部所有时间)
    avg_accounted_coarse = (avg_batch_preprocess + avg_cache_loop + 
                            avg_image_assembly + avg_array_append + avg_postprocess)
    # 使用细粒度计算（向后兼容旧数据）
    avg_accounted = avg_accounted_detailed
    avg_unaccounted = avg_batch_time_ms - avg_accounted
    
    return {
        "cache_mode": meta.get("cache_mode", "unknown"),
        "avg_batch_time_ms": avg_batch_time_ms,
        "batch_time_grow_ms_per_step": batch_time_grow_ms_per_step,
        "peak_cache_memory_mb": peak_cache_memory_mb,
        "cache_memory_grow_mb_per_step": cache_memory_grow_mb_per_step,
        "num_steps": len(steps),
        # Breakdown components
        "avg_render_time_ms": avg_render,
        "avg_cache_lookup_time_ms": avg_cache_lookup,
        "avg_cache_update_time_ms": avg_cache_update,
        "avg_array_conversion_time_ms": avg_array_conversion,
        "avg_array_append_time_ms": avg_array_append,
        "avg_text_preprocess_time_ms": avg_text_preprocess,
        "avg_image_assembly_time_ms": avg_image_assembly,
        "avg_batch_preprocess_time_ms": avg_batch_preprocess,
        "avg_postprocess_time_ms": avg_postprocess,
        "avg_cache_loop_time_ms": avg_cache_loop,
        "avg_accounted_time_ms": avg_accounted,
        "avg_accounted_coarse_time_ms": avg_accounted_coarse,
        "avg_unaccounted_time_ms": avg_unaccounted,
    }


def calculate_speedup_and_mem_save(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """计算Speedup率和Mem Save"""
    # 确保有none模式作为基准
    if "none" not in results:
        print("Warning: 'none' mode not found, cannot calculate speedup")
        return results
    
    none_avg_time = results["none"]["avg_batch_time_ms"]
    
    # 确保有naive模式作为内存基准（如果没有naive，使用segment）
    mem_baseline_mode = None
    if "naive" in results:
        mem_baseline_mode = "naive"
    elif "segment" in results:
        mem_baseline_mode = "segment"
    else:
        print("Warning: No baseline mode found for mem save calculation")
        return results
    
    baseline_peak_mem = results[mem_baseline_mode]["peak_cache_memory_mb"]
    
    # 为每个模式计算speedup和mem_save
    for mode, data in results.items():
        if mode == "none":
            data["speedup"] = 1.0
            data["mem_save_mb"] = 0.0
            data["mem_save_percent"] = 0.0
        else:
            # Speedup率 = none的batch_time / 当前模式的batch_time
            data["speedup"] = none_avg_time / data["avg_batch_time_ms"]
            
            # Mem Save = baseline的peak_mem - 当前模式的peak_mem
            if mode == mem_baseline_mode:
                data["mem_save_mb"] = 0.0
                data["mem_save_percent"] = 0.0
            else:
                data["mem_save_mb"] = baseline_peak_mem - data["peak_cache_memory_mb"]
                if baseline_peak_mem > 0:
                    data["mem_save_percent"] = (data["mem_save_mb"] / baseline_peak_mem) * 100
                else:
                    data["mem_save_percent"] = 0.0
    
    return results


def print_results(results: Dict[str, Dict]):
    """打印结果表格"""
    print("\n" + "="*100)
    print("OCR Statistics Analysis Results")
    print("="*100)
    
    # 表头
    header = f"{'Mode':<15} {'Avg Batch Time (ms)':<20} {'Batch Time Grow (ms/step)':<25} "
    header += f"{'Speedup':<12} {'Peak Cache (MB)':<18} {'Cache Grow (MB/step)':<20} "
    header += f"{'Mem Save (MB)':<15} {'Mem Save (%)':<12}"
    print(header)
    print("-"*100)
    
    # 按模式顺序打印
    mode_order = ["none", "naive", "segment"]
    for mode in mode_order:
        if mode not in results:
            continue
        
        data = results[mode]
        row = f"{data['cache_mode']:<15} "
        row += f"{data['avg_batch_time_ms']:<20.2f} "
        row += f"{data['batch_time_grow_ms_per_step']:<25.4f} "
        row += f"{data.get('speedup', 0.0):<12.2f}x "
        row += f"{data['peak_cache_memory_mb']:<18.2f} "
        row += f"{data['cache_memory_grow_mb_per_step']:<20.4f} "
        row += f"{data.get('mem_save_mb', 0.0):<15.2f} "
        row += f"{data.get('mem_save_percent', 0.0):<12.2f}%"
        print(row)
    
    print("="*100)
    
    # 打印 batch_time 的完整 breakdown
    print("\n" + "="*120)
    print("Batch Time Breakdown (Average ms per step)")
    print("="*120)
    
    header2 = f"{'Mode':<10} {'Render':<10} {'CacheLkp':<10} {'CacheUpd':<10} "
    header2 += f"{'ArrConv':<10} {'ArrAppnd':<10} {'TextPrep':<10} {'ImgAssm':<10} "
    header2 += f"{'BatchPre':<10} {'PostProc':<10} {'CacheLoop':<10} {'Accounted':<11} {'Unacct':<10} {'Total':<10} {'%':<6}"
    print(header2)
    print("-"*120)
    
    for mode in mode_order:
        if mode not in results:
            continue
        
        data = results[mode]
        total = data['avg_batch_time_ms']
        accounted = data.get('avg_accounted_time_ms', 0.0)
        unaccounted = data.get('avg_unaccounted_time_ms', 0.0)
        unaccounted_pct = (unaccounted / total * 100) if total > 0 else 0
        
        row2 = f"{data['cache_mode']:<10} "
        row2 += f"{data.get('avg_render_time_ms', 0.0):<10.2f} "
        row2 += f"{data.get('avg_cache_lookup_time_ms', 0.0):<10.2f} "
        row2 += f"{data.get('avg_cache_update_time_ms', 0.0):<10.2f} "
        row2 += f"{data.get('avg_array_conversion_time_ms', 0.0):<10.2f} "
        row2 += f"{data.get('avg_array_append_time_ms', 0.0):<10.2f} "
        row2 += f"{data.get('avg_text_preprocess_time_ms', 0.0):<10.2f} "
        row2 += f"{data.get('avg_image_assembly_time_ms', 0.0):<10.2f} "
        row2 += f"{data.get('avg_batch_preprocess_time_ms', 0.0):<10.2f} "
        row2 += f"{data.get('avg_postprocess_time_ms', 0.0):<10.2f} "
        row2 += f"{data.get('avg_cache_loop_time_ms', 0.0):<10.2f} "
        row2 += f"{accounted:<11.2f} "
        row2 += f"{unaccounted:<10.2f} "
        row2 += f"{total:<10.2f} "
        row2 += f"{unaccounted_pct:<6.1f}%"
        print(row2)
    
    print("="*120)
    print("\nNotes:")
    print("- Speedup: Compared to 'none' mode (higher is better)")
    print("- Mem Save: Compared to baseline cache mode (higher is better)")
    print("- Batch Time Grow: Positive means increasing, negative means decreasing")
    print("- Cache Grow: Positive means increasing, negative means decreasing")
    print("\nBreakdown Legend:")
    print("  Render     = Text rendering time (PIL ImageDraw)")
    print("  CacheLkp   = Cache lookup time")
    print("  CacheUpd   = Cache update/insert time")
    print("  ArrConv    = Array conversion time (PIL -> numpy)")
    print("  ArrAppnd   = Array append to result list time")
    print("  TextPrep   = Text preprocessing time (splitting, etc.)")
    print("  ImgAssm    = Image assembly time (np.vstack)")
    print("  BatchPre   = Batch-level preprocessing (masks, blank array creation)")
    print("  PostProc   = Post-processing (array reconstruction, compression, padding)")
    print("  CacheLoop  = Total time in segment cache loop (includes Render, CacheLkp, CacheUpd, TextPrep)")
    print("  Unacct     = Other overhead not explicitly tracked (Python overhead, etc.)")
    print()
    print("Note: 'Accounted' uses fine-grained times. CacheLoop is shown for validation.")


def main():
    # 数据目录路径
    data_dir = Path(__file__).parent.parent.parent / "logs" / "ocr_stats"
    
    # 查找所有JSONL文件
    jsonl_files = list(data_dir.glob("ocr_stats_*.jsonl"))
    
    if not jsonl_files:
        print(f"Error: No JSONL files found in {data_dir}")
        return
    
    results = {}
    
    # 处理每个文件
    for filepath in sorted(jsonl_files):
        meta, steps = load_jsonl_data(filepath)
        if not steps:
            print(f"Warning: No step records found in {filepath}")
            continue
        
        cache_mode = meta.get("cache_mode", "unknown")
        metrics = calculate_metrics(meta, steps)
        results[cache_mode] = metrics
        print(f"Loaded {len(steps)} steps from {filepath.name} (mode: {cache_mode})")
    
    # 计算speedup和mem_save
    results = calculate_speedup_and_mem_save(results)
    
    # 打印结果
    print_results(results)
    
    # 可选：保存到JSON文件
    output_json = Path(__file__).parent / "analysis_results.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()

