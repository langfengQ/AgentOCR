import json
from pathlib import Path
import matplotlib.pyplot as plt


def summarize_jsonl(path: str | Path):
    path = Path(path)
    meta = {}
    batch_times = []
    cache_mems = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("_type") == "metadata":
                meta = obj
                continue
            # 只统计 step 行
            if obj.get("_type") != "step":
                continue

            # 关键指标（你的 jsonl 里就是这两个 key）
            batch_times.append(float(obj["batch_time_ms"]))
            cache_mems.append(float(obj["cache_memory_mb"]))

    if not batch_times:
        raise ValueError(f"No step records found in {path}")

    avg_batch_ms = sum(batch_times) / len(batch_times)
    peak_cache_mb = max(cache_mems) if cache_mems else 0.0
    label = meta.get("cache_mode", path.stem)

    return label, avg_batch_ms, peak_cache_mb


def plot_pareto(points, title="Pareto Scatter: Latency vs Cache Memory", output_path=None):
    """
    points: list of (label, avg_batch_ms, peak_cache_mb)
    output_path: path to save the figure. If None, saves as 'pareto_plot.png' in current directory.
    """
    plt.figure(figsize=(7, 5))

    # 画点 + 标注
    for label, avg_ms, peak_mb in points:
        plt.scatter(peak_mb, avg_ms, s=90, marker="o")
        plt.annotate(
            f"{label}\n({peak_mb:.2f}MB, {avg_ms:.2f}ms)",
            (peak_mb, avg_ms),
            textcoords="offset points",
            xytext=(8, 8),
            ha="left",
            fontsize=9,
        )

    # x 轴用 symlog：既能看 0，也能拉开 190 vs 4900 的差距
    plt.xscale("symlog", linthresh=1.0)
    plt.grid(True, which="both", linewidth=0.6)

    plt.xlabel("Peak Cache Memory (MB) ↓")
    plt.ylabel("Avg Batch Time (ms) ↓")
    plt.title(title)

    # 视觉提示：左下角更好
    plt.text(
        0.02, 0.02, "Better →", transform=plt.gca().transAxes, fontsize=10
    )

    plt.tight_layout()
    
    if output_path is None:
        output_path = "pareto_plot.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # 数据目录路径
    data_dir = Path(__file__).parent.parent.parent / "logs" / "ocr_stats"
    
    # 查找所有JSONL文件
    jsonl_files = list(data_dir.glob("ocr_stats_*.jsonl"))
    
    if not jsonl_files:
        print(f"Error: No JSONL files found in {data_dir}")
        exit(1)
    
    points = [summarize_jsonl(p) for p in sorted(jsonl_files)]
    # 可选：把标签更"论文味"一点
    rename = {
        "none": "No Cache",
        "naive": "Naive Cache",
        "segment": "Ours (Segment Cache)",
    }
    points = [(rename.get(lbl, lbl), avg, peak) for (lbl, avg, peak) in points]

    output_path = data_dir / "pareto_plot.png"
    plot_pareto(points, output_path=str(output_path))
