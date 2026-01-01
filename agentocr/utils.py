from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict, Any
import re
from functools import lru_cache
import time
import os
from dataclasses import dataclass, field

import numpy as np

# Optional psutil import for CPU/memory monitoring
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None

# Global font cache
_FONT_CACHE = {}


@dataclass
class StepStats:
    """Statistics snapshot for a single step/batch."""
    step: int = 0
    timestamp: float = 0.0
    
    # Time stats for this step
    render_time: float = 0.0
    render_count: int = 0
    cache_lookup_time: float = 0.0
    cache_update_time: float = 0.0
    batch_time: float = 0.0
    
    # Additional detailed timings
    array_conversion_time: float = 0.0  # PIL to numpy conversion
    array_append_time: float = 0.0      # Appending to result list
    text_preprocess_time: float = 0.0   # Text splitting and preprocessing
    image_assembly_time: float = 0.0    # Image stacking/assembly
    batch_preprocess_time: float = 0.0  # Batch-level preprocessing (masks, blank creation)
    postprocess_time: float = 0.0       # Post-processing (compression, padding, etc.)
    cache_loop_time: float = 0.0        # Total time in cache processing loop (per env)
    
    # Cache stats for this step
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Memory stats at this step (in bytes)
    cache_memory_bytes: int = 0
    process_memory_bytes: int = 0
    
    # Cache counts at this step
    master_image_count: int = 0
    full_image_count: int = 0
    segment_count: int = 0
    
    # CPU at this step
    cpu_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'step': self.step,
            'timestamp': round(self.timestamp, 4),
            'render_time_ms': round(self.render_time * 1000, 4),
            'render_count': self.render_count,
            'cache_lookup_time_ms': round(self.cache_lookup_time * 1000, 4),
            'cache_update_time_ms': round(self.cache_update_time * 1000, 4),
            'batch_time_ms': round(self.batch_time * 1000, 4),
            'array_conversion_time_ms': round(self.array_conversion_time * 1000, 4),
            'array_append_time_ms': round(self.array_append_time * 1000, 4),
            'text_preprocess_time_ms': round(self.text_preprocess_time * 1000, 4),
            'image_assembly_time_ms': round(self.image_assembly_time * 1000, 4),
            'batch_preprocess_time_ms': round(self.batch_preprocess_time * 1000, 4),
            'postprocess_time_ms': round(self.postprocess_time * 1000, 4),
            'cache_loop_time_ms': round(self.cache_loop_time * 1000, 4),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_memory_mb': round(self.cache_memory_bytes / (1024 * 1024), 4),
            'process_memory_mb': round(self.process_memory_bytes / (1024 * 1024), 4),
            'master_image_count': self.master_image_count,
            'full_image_count': self.full_image_count,
            'segment_count': self.segment_count,
            'cpu_percent': round(self.cpu_percent, 2),
        }


@dataclass
class PerformanceStats:
    """
    Performance statistics for OCR operations.
    Tracks time, memory (cache-specific), and CPU usage.
    """
    # Time statistics (in seconds)
    total_render_time: float = 0.0
    total_cache_lookup_time: float = 0.0
    total_cache_update_time: float = 0.0
    total_array_conversion_time: float = 0.0  # PIL to numpy conversion
    total_array_append_time: float = 0.0      # Appending to result list
    total_text_preprocess_time: float = 0.0   # Text splitting and preprocessing
    total_image_assembly_time: float = 0.0    # Image stacking/assembly
    total_batch_preprocess_time: float = 0.0  # Batch-level preprocessing
    total_postprocess_time: float = 0.0       # Post-processing
    total_cache_loop_time: float = 0.0        # Total time in cache processing loop
    render_count: int = 0
    cache_lookup_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    
    # Cache memory statistics (in bytes) - ONLY for OCR cache data structures
    master_image_cache_bytes: int = 0      # Variant 3: master image cache size
    full_image_cache_bytes: int = 0        # Variant 2: full image cache size
    segment_cache_bytes: int = 0           # Segment cache size
    compact_cache_bytes: int = 0           # Compact mode cache size
    peak_cache_bytes: int = 0              # Peak cache size across all types
    
    # Cache memory samples for averaging
    cache_memory_samples: List[int] = field(default_factory=list)  # OCR cache memory samples
    process_memory_samples: List[int] = field(default_factory=list)  # Process memory samples
    
    # Cache entry counts
    master_image_count: int = 0            # Number of environments with master images
    full_image_count: int = 0              # Number of cached full images
    segment_count: int = 0                 # Number of cached segments
    
    # Process memory (optional, for reference)
    process_memory_bytes: int = 0
    peak_process_memory_bytes: int = 0     # Peak process memory
    
    # CPU statistics
    cpu_percent_samples: List[float] = field(default_factory=list)
    
    # Batch statistics
    batch_times: List[float] = field(default_factory=list)
    
    # Per-step statistics for detailed logging
    step_stats: List[StepStats] = field(default_factory=list)
    
    # Tracking for per-step deltas
    _last_render_time: float = field(default=0.0, repr=False)
    _last_render_count: int = field(default=0, repr=False)
    _last_cache_lookup_time: float = field(default=0.0, repr=False)
    _last_cache_update_time: float = field(default=0.0, repr=False)
    _last_array_conversion_time: float = field(default=0.0, repr=False)
    _last_array_append_time: float = field(default=0.0, repr=False)
    _last_text_preprocess_time: float = field(default=0.0, repr=False)
    _last_image_assembly_time: float = field(default=0.0, repr=False)
    _last_batch_preprocess_time: float = field(default=0.0, repr=False)
    _last_postprocess_time: float = field(default=0.0, repr=False)
    _last_cache_loop_time: float = field(default=0.0, repr=False)
    _last_cache_hits: int = field(default=0, repr=False)
    _last_cache_misses: int = field(default=0, repr=False)
    
    def reset(self):
        """Reset all statistics."""
        self.total_render_time = 0.0
        self.total_cache_lookup_time = 0.0
        self.total_cache_update_time = 0.0
        self.total_array_conversion_time = 0.0
        self.total_array_append_time = 0.0
        self.total_text_preprocess_time = 0.0
        self.total_image_assembly_time = 0.0
        self.total_batch_preprocess_time = 0.0
        self.total_postprocess_time = 0.0
        self.total_cache_loop_time = 0.0
        self.render_count = 0
        self.cache_lookup_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.master_image_cache_bytes = 0
        self.full_image_cache_bytes = 0
        self.segment_cache_bytes = 0
        self.compact_cache_bytes = 0
        self.peak_cache_bytes = 0
        self.cache_memory_samples = []
        self.process_memory_samples = []
        self.master_image_count = 0
        self.full_image_count = 0
        self.segment_count = 0
        self.process_memory_bytes = 0
        self.peak_process_memory_bytes = 0
        self.cpu_percent_samples = []
        self.batch_times = []
        self.step_stats = []
        self._last_render_time = 0.0
        self._last_render_count = 0
        self._last_cache_lookup_time = 0.0
        self._last_cache_update_time = 0.0
        self._last_array_conversion_time = 0.0
        self._last_array_append_time = 0.0
        self._last_text_preprocess_time = 0.0
        self._last_image_assembly_time = 0.0
        self._last_batch_preprocess_time = 0.0
        self._last_postprocess_time = 0.0
        self._last_cache_loop_time = 0.0
        self._last_cache_hits = 0
        self._last_cache_misses = 0
    
    def record_step(self, step: int, batch_time: float, cpu_percent: float = 0.0):
        """
        Record statistics snapshot for a step.
        
        Args:
            step: Current step number
            batch_time: Time taken for this batch
            cpu_percent: CPU usage for this step
        """
        step_stat = StepStats(
            step=step,
            timestamp=time.time(),
            render_time=self.total_render_time - self._last_render_time,
            render_count=self.render_count - self._last_render_count,
            cache_lookup_time=self.total_cache_lookup_time - self._last_cache_lookup_time,
            cache_update_time=self.total_cache_update_time - self._last_cache_update_time,
            batch_time=batch_time,
            array_conversion_time=self.total_array_conversion_time - self._last_array_conversion_time,
            array_append_time=self.total_array_append_time - self._last_array_append_time,
            text_preprocess_time=self.total_text_preprocess_time - self._last_text_preprocess_time,
            image_assembly_time=self.total_image_assembly_time - self._last_image_assembly_time,
            batch_preprocess_time=self.total_batch_preprocess_time - self._last_batch_preprocess_time,
            postprocess_time=self.total_postprocess_time - self._last_postprocess_time,
            cache_loop_time=self.total_cache_loop_time - self._last_cache_loop_time,
            cache_hits=self.cache_hit_count - self._last_cache_hits,
            cache_misses=self.cache_miss_count - self._last_cache_misses,
            cache_memory_bytes=self.total_cache_bytes,
            process_memory_bytes=self.process_memory_bytes,
            master_image_count=self.master_image_count,
            full_image_count=self.full_image_count,
            segment_count=self.segment_count,
            cpu_percent=cpu_percent,
        )
        self.step_stats.append(step_stat)
        
        # Update last values for next delta calculation
        self._last_render_time = self.total_render_time
        self._last_render_count = self.render_count
        self._last_cache_lookup_time = self.total_cache_lookup_time
        self._last_cache_update_time = self.total_cache_update_time
        self._last_array_conversion_time = self.total_array_conversion_time
        self._last_array_append_time = self.total_array_append_time
        self._last_text_preprocess_time = self.total_text_preprocess_time
        self._last_image_assembly_time = self.total_image_assembly_time
        self._last_batch_preprocess_time = self.total_batch_preprocess_time
        self._last_postprocess_time = self.total_postprocess_time
        self._last_cache_loop_time = self.total_cache_loop_time
        self._last_cache_hits = self.cache_hit_count
        self._last_cache_misses = self.cache_miss_count
    
    def save_step_stats(self, filepath: str, cache_mode: str = "unknown"):
        """
        Save per-step statistics to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            cache_mode: Cache mode string to include in metadata
        """
        import json
        
        data = {
            'metadata': {
                'cache_mode': cache_mode,
                'total_steps': len(self.step_stats),
                'summary': self.get_summary(),
            },
            'steps': [step.to_dict() for step in self.step_stats]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def append_step_to_json(self, filepath: str, step_stat: 'StepStats', is_first: bool = False, cache_mode: str = "unknown"):
        """
        Append a single step's statistics to a JSON file (JSON Lines format).
        
        Args:
            filepath: Path to the JSON file
            step_stat: StepStats object to append
            is_first: Whether this is the first write (writes metadata)
            cache_mode: Cache mode string for metadata
        """
        import json
        
        mode = 'w' if is_first else 'a'
        with open(filepath, mode) as f:
            if is_first:
                # Write metadata as first line
                metadata = {
                    "_type": "metadata",
                    "cache_mode": cache_mode,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                f.write(json.dumps(metadata) + '\n')
            # Append step data
            step_data = step_stat.to_dict()
            step_data["_type"] = "step"
            f.write(json.dumps(step_data) + '\n')
    
    @property
    def total_cache_bytes(self) -> int:
        """Total cache memory usage across all cache types."""
        return self.master_image_cache_bytes + self.full_image_cache_bytes + self.segment_cache_bytes + self.compact_cache_bytes
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance statistics."""
        avg_render_time = self.total_render_time / max(1, self.render_count)
        avg_cache_lookup_time = self.total_cache_lookup_time / max(1, self.cache_lookup_count)
        avg_cpu = sum(self.cpu_percent_samples) / max(1, len(self.cpu_percent_samples))
        avg_batch_time = sum(self.batch_times) / max(1, len(self.batch_times))
        total_cache_mb = self.total_cache_bytes / (1024 * 1024)
        
        # Average memory calculations
        avg_cache_memory = sum(self.cache_memory_samples) / max(1, len(self.cache_memory_samples))
        avg_process_memory = sum(self.process_memory_samples) / max(1, len(self.process_memory_samples))
        
        return {
            # Time stats
            'total_render_time_s': round(self.total_render_time, 4),
            'avg_render_time_ms': round(avg_render_time * 1000, 4),
            'render_count': self.render_count,
            'total_cache_lookup_time_s': round(self.total_cache_lookup_time, 4),
            'avg_cache_lookup_time_ms': round(avg_cache_lookup_time * 1000, 4),
            'cache_lookup_count': self.cache_lookup_count,
            'total_cache_update_time_s': round(self.total_cache_update_time, 4),
            # Cache stats
            'cache_hit_count': self.cache_hit_count,
            'cache_miss_count': self.cache_miss_count,
            # Cache memory stats (OCR cache ONLY) - current
            'master_image_cache_mb': round(self.master_image_cache_bytes / (1024 * 1024), 2),
            'full_image_cache_mb': round(self.full_image_cache_bytes / (1024 * 1024), 2),
            'segment_cache_mb': round(self.segment_cache_bytes / (1024 * 1024), 2),
            'compact_cache_mb': round(self.compact_cache_bytes / (1024 * 1024), 2),
            'total_cache_mb': round(total_cache_mb, 2),
            'peak_cache_mb': round(self.peak_cache_bytes / (1024 * 1024), 2),
            'avg_cache_mb': round(avg_cache_memory / (1024 * 1024), 2),
            # Cache entry counts
            'master_image_count': self.master_image_count,
            'full_image_count': self.full_image_count,
            'segment_count': self.segment_count,
            # Process memory (reference)
            'process_memory_mb': round(self.process_memory_bytes / (1024 * 1024), 2),
            'peak_process_memory_mb': round(self.peak_process_memory_bytes / (1024 * 1024), 2),
            'avg_process_memory_mb': round(avg_process_memory / (1024 * 1024), 2),
            # CPU/Batch stats
            'avg_cpu_percent': round(avg_cpu, 2),
            'avg_batch_time_ms': round(avg_batch_time * 1000, 4),
            'total_batches': len(self.batch_times),
        }


class PerformanceMonitor:
    """
    Context manager and utilities for monitoring performance.
    Focuses on OCR cache memory, not general process memory.
    """
    
    def __init__(self, stats: PerformanceStats):
        self.stats = stats
        self._process = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        self._start_time = None
        self._json_filepath = None
        self._json_initialized = False
        self._cache_mode = "unknown"
    
    def set_json_filepath(self, filepath: str, cache_mode: str = "unknown"):
        """
        Set the JSON filepath for real-time step saving.
        
        Args:
            filepath: Path to the JSON file
            cache_mode: Cache mode string for metadata
        """
        self._json_filepath = filepath
        self._json_initialized = False
        self._cache_mode = cache_mode
    
    def start_batch(self):
        """Start timing a batch operation."""
        self._start_time = time.perf_counter()
        
        if self._process is not None:
            try:
                # Sample CPU at start
                cpu_percent = self._process.cpu_percent(interval=None)
                if cpu_percent > 0:
                    self.stats.cpu_percent_samples.append(cpu_percent)
            except:
                pass
    
    def end_batch(self, step: int = -1):
        """
        End timing a batch operation and record stats.
        
        Args:
            step: Current step number for per-step logging (-1 to auto-increment)
        """
        batch_time = 0.0
        if self._start_time is not None:
            batch_time = time.perf_counter() - self._start_time
            self.stats.batch_times.append(batch_time)
            self._start_time = None
        
        cpu_percent = 0.0
        if self._process is not None:
            try:
                # Update process memory (for reference only)
                current_process_mem = self._process.memory_info().rss
                self.stats.process_memory_bytes = current_process_mem
                self.stats.process_memory_samples.append(current_process_mem)
                
                # Update peak process memory
                if current_process_mem > self.stats.peak_process_memory_bytes:
                    self.stats.peak_process_memory_bytes = current_process_mem
                
                # Sample CPU
                cpu_percent = self._process.cpu_percent(interval=None)
                if cpu_percent > 0:
                    self.stats.cpu_percent_samples.append(cpu_percent)
            except:
                pass
        
        # Record cache memory sample
        current_cache_mem = self.stats.total_cache_bytes
        self.stats.cache_memory_samples.append(current_cache_mem)
        
        # Record per-step statistics (this completes the timing measurement)
        if step == -1:
            step = len(self.stats.step_stats)
        self.stats.record_step(step, batch_time, cpu_percent)
        
        # Real-time save to JSON if filepath is set
        # Note: This happens AFTER recording stats, so save time is NOT included in measurements
        if self._json_filepath and self.stats.step_stats:
            latest_step = self.stats.step_stats[-1]
            is_first = not self._json_initialized
            self.stats.append_step_to_json(
                self._json_filepath, 
                latest_step, 
                is_first=is_first,
                cache_mode=self._cache_mode
            )
            self._json_initialized = True
    
    def time_render(self):
        """Context manager for timing render operations."""
        return _TimingContext(self.stats, 'render')
    
    def time_cache_lookup(self):
        """Context manager for timing cache lookup operations."""
        return _TimingContext(self.stats, 'cache_lookup')
    
    def time_cache_update(self):
        """Context manager for timing cache update operations."""
        return _TimingContext(self.stats, 'cache_update')
    
    def time_array_conversion(self):
        """Context manager for timing array conversion operations."""
        return _TimingContext(self.stats, 'array_conversion')
    
    def time_array_append(self):
        """Context manager for timing array append operations."""
        return _TimingContext(self.stats, 'array_append')
    
    def time_text_preprocess(self):
        """Context manager for timing text preprocessing operations."""
        return _TimingContext(self.stats, 'text_preprocess')
    
    def time_image_assembly(self):
        """Context manager for timing image assembly operations."""
        return _TimingContext(self.stats, 'image_assembly')
    
    def time_batch_preprocess(self):
        """Context manager for timing batch-level preprocessing."""
        return _TimingContext(self.stats, 'batch_preprocess')
    
    def time_postprocess(self):
        """Context manager for timing post-processing operations."""
        return _TimingContext(self.stats, 'postprocess')
    
    def time_cache_loop(self):
        """Context manager for timing cache processing loop."""
        return _TimingContext(self.stats, 'cache_loop')
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.stats.cache_hit_count += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.stats.cache_miss_count += 1
    
    def record_render(self, elapsed_time: float):
        """Record a render operation with timing."""
        self.stats.total_render_time += elapsed_time
        self.stats.render_count += 1
    
    def update_segment_cache_memory(self, segment_caches: dict):
        """
        Update memory stats for segment cache.
        
        Args:
            segment_caches: Dict of {env_idx: SegmentCache}
        """
        if segment_caches is None:
            self.stats.segment_cache_bytes = 0
            self.stats.segment_count = 0
            return
        
        total_bytes = 0
        total_segments = 0
        
        for env_idx, segment_cache in segment_caches.items():
            if hasattr(segment_cache, '_cache'):
                for cached_img in segment_cache._cache.values():
                    if isinstance(cached_img, np.ndarray):
                        total_bytes += cached_img.nbytes
                total_segments += len(segment_cache._cache)
        
        self.stats.segment_cache_bytes = total_bytes
        self.stats.segment_count = total_segments
        self._update_peak_cache()
    
    def update_master_image_cache_memory(self, master_images: dict):
        """
        Update memory stats for master image cache (Variant 3).
        
        Args:
            master_images: Dict of {env_idx: {'master_img': np.ndarray, 'segments': [...], ...}}
        """
        if master_images is None:
            self.stats.master_image_cache_bytes = 0
            self.stats.master_image_count = 0
            self.stats.segment_count = 0
            return
        
        total_bytes = 0
        total_segments = 0
        
        for env_idx, data in master_images.items():
            if data.get('master_img') is not None:
                total_bytes += data['master_img'].nbytes
            if 'segments' in data:
                total_segments += len(data['segments'])
                # Estimate segment metadata (~200 bytes per segment)
                total_bytes += len(data['segments']) * 200
        
        self.stats.master_image_cache_bytes = total_bytes
        self.stats.master_image_count = len(master_images)
        self.stats.segment_count = total_segments
        self._update_peak_cache()
    
    def update_full_image_cache_memory(self, full_image_cache: dict):
        """
        Update memory stats for full image cache (Variant 2).
        
        Args:
            full_image_cache: Dict of {env_idx: {context_hash: {'text': str, 'image': np.ndarray}}}
        """
        if full_image_cache is None:
            self.stats.full_image_cache_bytes = 0
            self.stats.full_image_count = 0
            return
        
        total_bytes = 0
        total_images = 0
        
        for env_idx, env_cache in full_image_cache.items():
            for context_hash, cache_entry in env_cache.items():
                if isinstance(cache_entry, dict):
                    # New format: {'text': str, 'image': np.ndarray}
                    img_array = cache_entry.get('image')
                    text = cache_entry.get('text', '')
                    if img_array is not None:
                        total_bytes += img_array.nbytes
                        total_images += 1
                    # Estimate text memory (~1 byte per char)
                    total_bytes += len(text)
                elif isinstance(cache_entry, np.ndarray):
                    # Old format fallback: just np.ndarray
                    total_bytes += cache_entry.nbytes
                    total_images += 1
        
        self.stats.full_image_cache_bytes = total_bytes
        self.stats.full_image_count = total_images
        self._update_peak_cache()
    
    def update_compact_cache_memory(self, compact_cache: dict):
        """
        Update memory stats for compact mode cache.
        
        Args:
            compact_cache: Dict of {env_idx: {'complete_lines_img': np.ndarray, ...}}
        """
        if compact_cache is None:
            self.stats.compact_cache_bytes = 0
            return
        
        total_bytes = 0
        
        for env_idx, data in compact_cache.items():
            if data.get('complete_lines_img') is not None:
                total_bytes += data['complete_lines_img'].nbytes
        
        self.stats.compact_cache_bytes = total_bytes
        self._update_peak_cache()
    
    def _update_peak_cache(self):
        """Update peak cache memory."""
        total = self.stats.total_cache_bytes
        if total > self.stats.peak_cache_bytes:
            self.stats.peak_cache_bytes = total


class _TimingContext:
    """Context manager for timing specific operations."""
    
    def __init__(self, stats: PerformanceStats, operation: str):
        self.stats = stats
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        if self.operation == 'render':
            self.stats.total_render_time += elapsed
            self.stats.render_count += 1
        elif self.operation == 'cache_lookup':
            self.stats.total_cache_lookup_time += elapsed
            self.stats.cache_lookup_count += 1
        elif self.operation == 'cache_update':
            self.stats.total_cache_update_time += elapsed
        elif self.operation == 'array_conversion':
            self.stats.total_array_conversion_time += elapsed
        elif self.operation == 'array_append':
            self.stats.total_array_append_time += elapsed
        elif self.operation == 'text_preprocess':
            self.stats.total_text_preprocess_time += elapsed
        elif self.operation == 'image_assembly':
            self.stats.total_image_assembly_time += elapsed
        elif self.operation == 'batch_preprocess':
            self.stats.total_batch_preprocess_time += elapsed
        elif self.operation == 'postprocess':
            self.stats.total_postprocess_time += elapsed
        elif self.operation == 'cache_loop':
            self.stats.total_cache_loop_time += elapsed
        return False

# Compact mode special symbol
COMPACT_NEWLINE_SYMBOL = "⏎"  # Return symbol to represent newlines


def apply_compact_mode(text: str, symbol: str = COMPACT_NEWLINE_SYMBOL) -> str:
    """
    Convert text to compact mode by replacing newlines with a special symbol.
    
    Args:
        text: Original text with newlines
        symbol: Symbol to replace newlines with (default: ⏎)
    
    Returns:
        Text with newlines replaced by the symbol, treated as single paragraph
    """
    if not text:
        return ""
    
    # Replace newlines with the symbol (add space around for readability)
    compact_text = text.replace('\n', f' {symbol} ')
    # Clean up multiple spaces
    compact_text = ' '.join(compact_text.split())
    return compact_text


def get_compact_symbol_positions(text: str, symbol: str = COMPACT_NEWLINE_SYMBOL) -> List[int]:
    """
    Find positions of compact mode symbols in text.
    
    Args:
        text: Text containing compact symbols
        symbol: The symbol to find
    
    Returns:
        List of character positions where the symbol appears
    """
    positions = []
    start = 0
    while True:
        pos = text.find(symbol, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions

def preprocess_trajectory_contexts(trajectory_contexts: List[str]) -> List[str]:
    """
    Preprocess trajectory contexts.
    """
    # replace \" with "
    return [context.replace('\\"', '\"') for context in trajectory_contexts]

def _get_cached_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    """
    Get or create a cached font object to avoid repeated font loading.
    This provides significant speedup for repeated text rendering.
    """
    cache_key = (font_path or "default", font_size)
    
    if cache_key not in _FONT_CACHE:
        font = None
        font_paths = []
        
        if font_path:
            font_paths.append(font_path)
        
        # Prioritize monospace fonts for better packing efficiency
        font_paths.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Monaco.ttf",  # macOS
            "C:\\Windows\\Fonts\\consola.ttf",   # Windows
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "Arial.ttf",
        ])
        
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        _FONT_CACHE[cache_key] = font
    
    return _FONT_CACHE[cache_key]


# @lru_cache(maxsize=128)
# def parse_trajectory_text(text: str) -> Tuple[Tuple[str, str, str], ...]:
#     """
#     Extract Observation and Action pairs from trajectory text.

#     Args:
#         text: The original trajectory text, can contain multiple rounds

#     Returns:
#         A tuple of tuples in the format ((obs_num, obs_text, action_text), ...)
#         (Changed to tuple for caching compatibility)
#     """
#     pairs = []
    
#     # Match [Observation N: '...', Action N: '...'] format
#     pattern = r"\[Observation\s+(\d+):\s*'(.*?)',\s*Action\s+\d+:\s*'(.*?)'\]"
#     matches = re.findall(pattern, text, re.DOTALL)
    
#     for match in matches:
#         obs_num, obs_text, action_text = match
#         # Unescape the text (handle \n, \t, etc.)
#         obs_text = obs_text.replace('\\n', '\n').replace('\\t', '\t')
#         action_text = action_text.replace('\\n', '\n').replace('\\t', '\t')
        
#         # Replace all newlines with spaces to keep text on single line
#         obs_text = obs_text.replace('\n', ' ').replace('\r', ' ')
#         action_text = action_text.replace('\n', ' ').replace('\r', ' ')
        
#         # Remove multiple consecutive spaces
#         obs_text = ' '.join(obs_text.split())
#         action_text = ' '.join(action_text.split())
        
#         pairs.append((obs_num, obs_text, action_text))
    
#     return tuple(pairs)


# def format_trajectory_compact(pairs: List[Tuple[str, str, str]]) -> str:
#     """
#     Format Observation-Action pairs into a compact format without empty lines
#     """
#     lines = []
#     for obs_num, obs_text, action_text in pairs:
#         lines.append(f"[Observation {obs_num}]: {obs_text}")
#         lines.append(f"[Action {obs_num}]: {action_text}")
    
#     result = "\n".join(lines)
#     return result


def wrap_text_fast(text: str, max_chars_per_line: int) -> List[Tuple[str, bool]]:
    """
    Fast text wrapping based on character count.
    Returns a list of tuples (line_text, is_paragraph_end) to track paragraph boundaries.
    Optimized for speed with early returns and minimal operations.
    """
    if not text:
        return []
    
    lines = []
    paragraphs = text.split('\n')
    num_paragraphs = len(paragraphs)
    
    for para_idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append(("", True))
            continue
        
        # Fast path for short paragraphs
        if len(paragraph) <= max_chars_per_line:
            is_last_para = para_idx == num_paragraphs - 1
            lines.append((paragraph, not is_last_para))
            continue
        
        words = paragraph.split()
        current_line = ""
        current_len = 0
        
        for word in words:
            word_len = len(word)
            test_len = current_len + (1 if current_len else 0) + word_len
            
            if test_len <= max_chars_per_line:
                if current_line:
                    current_line += " " + word
                    current_len = test_len
                else:
                    current_line = word
                    current_len = word_len
            else:
                if current_line:
                    lines.append((current_line, False))
                
                if word_len > max_chars_per_line:
                    # Split long words
                    for i in range(0, word_len, max_chars_per_line):
                        lines.append((word[i:i + max_chars_per_line], False))
                    current_line = ""
                    current_len = 0
                else:
                    current_line = word
                    current_len = word_len
        
        if current_line:
            is_last_para = para_idx == num_paragraphs - 1
            lines.append((current_line, not is_last_para))
    
    return lines


def wrap_text_compact(
    text: str, 
    max_chars_per_line: int
) -> Tuple[List[Tuple[str, bool]], int, str]:
    """
    Wrap text for compact mode (single paragraph, no newline splitting).
    
    This function is specifically designed for compact mode caching:
    - All text is treated as a single paragraph
    - Returns information about complete vs incomplete lines for caching
    
    Args:
        text: Text to wrap (should already have newlines replaced with symbols)
        max_chars_per_line: Maximum characters per line
    
    Returns:
        Tuple of:
        - lines: List of (line_text, is_complete) tuples
                 is_complete=True means the line filled the available width
        - complete_char_count: Number of characters in complete lines
        - incomplete_text: Text that doesn't fill a complete line (for next render)
    """
    if not text:
        return [], 0, ""
    
    text = text.strip()
    if not text:
        return [], 0, ""
    
    lines = []
    words = text.split()
    current_line = ""
    current_len = 0
    char_position = 0  # Track position in original text
    complete_char_count = 0
    
    for word_idx, word in enumerate(words):
        word_len = len(word)
        test_len = current_len + (1 if current_len else 0) + word_len
        
        if test_len <= max_chars_per_line:
            if current_line:
                current_line += " " + word
                current_len = test_len
            else:
                current_line = word
                current_len = word_len
        else:
            if current_line:
                # This line is complete (filled to capacity)
                lines.append((current_line, True))
                complete_char_count += len(current_line) + 1  # +1 for the space that would follow
            
            if word_len > max_chars_per_line:
                # Split long words
                for i in range(0, word_len, max_chars_per_line):
                    chunk = word[i:i + max_chars_per_line]
                    is_complete = (i + max_chars_per_line < word_len)
                    lines.append((chunk, is_complete))
                    if is_complete:
                        complete_char_count += len(chunk)
                current_line = ""
                current_len = 0
            else:
                current_line = word
                current_len = word_len
    
    # Handle the last line (incomplete - didn't fill the width)
    incomplete_text = ""
    if current_line:
        lines.append((current_line, False))  # Last line is never "complete"
        # The incomplete text is the last line's content
        incomplete_text = current_line
    
    return lines, complete_char_count, incomplete_text


def wrap_text_precise(text: str, max_width: int, font, font_size: int) -> List[Tuple[str, bool]]:
    """
    Precise text wrapping using actual font measurements for optimal packing.
    Returns a list of tuples (line_text, is_paragraph_end) to track paragraph boundaries.
    """
    lines = []
    paragraphs = text.split('\n')
    
    for para_idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append(("", True))
            continue
        
        words = paragraph.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            
            # Use actual font measurement
            try:
                bbox = font.getbbox(test_line)
                text_width = bbox[2] - bbox[0]
            except:
                # Fallback to character-based estimation
                text_width = len(test_line) * font_size * 0.6
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append((current_line, False))
                current_line = word
        
        if current_line:
            # Mark the last line of each paragraph
            is_last_para = para_idx == len(paragraphs) - 1
            lines.append((current_line, not is_last_para))
    
    return lines


# Cache for font metrics to avoid repeated calculations
_FONT_METRICS_CACHE = {}

def get_font_metrics(font, font_size: int) -> Tuple[float, int]:
    """
    Get accurate font metrics for optimal layout calculation.
    Returns (average_char_width, line_height)
    
    Optimized for maximum density: minimal line spacing while maintaining readability.
    Cached for performance.
    """
    # Use font object id and size as cache key
    cache_key = (id(font), font_size)
    
    if cache_key in _FONT_METRICS_CACHE:
        return _FONT_METRICS_CACHE[cache_key]
    
    # Test with a representative set of characters
    sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?()[]{}@#$%^&*-_=+/\\"
    
    try:
        bbox = font.getbbox(sample_text)
        total_width = bbox[2] - bbox[0]
        avg_char_width = total_width / len(sample_text)
        line_height = bbox[3] - bbox[1]
        # Ultra-compact: minimal spacing (1.05x instead of 1.2x)
        # This is the sweet spot between density and readability
        line_height = int(line_height * 1.2)
    except:
        # Fallback to estimates with compact spacing
        avg_char_width = font_size * 0.6  # Slightly more aggressive
        line_height = int(font_size * 1.2)
    
    result = (avg_char_width, line_height)
    _FONT_METRICS_CACHE[cache_key] = result
    return result


def find_fast_dimensions(
    text: str,
    font,
    font_size: int,
    padding: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    use_precise: bool = False
) -> Tuple[int, int, List[Tuple[str, bool]]]:
    """
    Fast dimension calculation - uses fixed width and calculates required height.
    Significantly faster than binary search approach, suitable for real-time use.
    
    Returns:
        (width, height, wrapped_lines) where wrapped_lines is List[Tuple[str, bool]]
    """
    # Use max_width for consistent, fast layout
    width = max_width
    available_width = width - 2 * padding
    
    # Get accurate font metrics
    avg_char_width, line_height = get_font_metrics(font, font_size)
    
    # Wrap text
    if use_precise:
        lines = wrap_text_precise(text, available_width, font, font_size)
    else:
        max_chars_per_line = int(available_width / avg_char_width)
        lines = wrap_text_fast(text, max_chars_per_line)
    
    # Calculate required height
    num_paragraph_breaks = sum(1 for _, is_para_end in lines if is_para_end)
    required_height = len(lines) * line_height + num_paragraph_breaks * int(line_height * 0.0) + 2 * padding
    
    # Clamp to min/max bounds
    height = max(min_height, min(max_height, required_height))
    
    # Truncate lines if needed
    if required_height > max_height:
        available_height = height - 2 * padding
        max_lines = int(available_height / line_height)
        lines = lines[:max_lines]
    
    return (width, height, lines)


def find_optimal_dimensions(
    text: str,
    font,
    font_size: int,
    padding: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    use_precise: bool = False
) -> Tuple[int, int, List[Tuple[str, bool]]]:
    """
    Find optimal image dimensions using binary search and precise font metrics.
    Goal: Maximum text coverage with minimum resolution while maintaining clarity.
    
    Returns:
        (width, height, wrapped_lines) where wrapped_lines is List[Tuple[str, bool]]
    """
    # Get accurate font metrics
    avg_char_width, line_height = get_font_metrics(font, font_size)
    
    # Calculate text length to estimate optimal starting point
    text_length = len(text.replace('\n', ' '))
    total_text_area = text_length * avg_char_width * line_height
    
    # Start with a square-ish aspect ratio for better packing
    aspect_ratio = 1.5  # Slightly wider than tall for better readability
    estimated_width = int((total_text_area * aspect_ratio) ** 0.5)
    estimated_width = max(min_width, min(max_width, estimated_width))
    
    def evaluate_width(width: int) -> Tuple[int, List[Tuple[str, bool]], bool]:
        """
        Evaluate a given width and return (height, lines, fits).
        Returns fits=True if text fits within max_height.
        """
        available_width = width - 2 * padding
        
        if use_precise:
            lines = wrap_text_precise(text, available_width, font, font_size)
        else:
            max_chars_per_line = int(available_width / avg_char_width)
            lines = wrap_text_fast(text, max_chars_per_line)
        
        # Calculate height considering paragraph spacing (minimal spacing for compact layout)
        num_paragraph_breaks = sum(1 for _, is_para_end in lines if is_para_end)
        required_height = len(lines) * line_height + num_paragraph_breaks * int(line_height * 0.0) + 2 * padding
        fits = required_height <= max_height
        
        return required_height, lines, fits
    
    # Binary search for minimum width that fits all text
    left, right = estimated_width, max_width
    best_solution = None
    best_area = float('inf')

    while left <= right:
        mid = (left + right) // 2
        if mid < left:
            mid = left
        if mid > right:
            mid = right
            
        required_height, lines, fits = evaluate_width(mid)
        
        if fits:
            # Text fits! Try to minimize area
            height = required_height
            height = max(min_height, min(max_height, height))
            area = mid * height
            
            if area < best_area:
                best_area = area
                best_solution = (mid, height, lines)
            
            # Try smaller width
            right = mid - 1
        else:
            # Doesn't fit, need wider
            left = mid + 1

    # If no solution found in binary search, use max dimensions with truncation
    if best_solution is None:
        width = max_width
        height = max_height
        available_width = width - 2 * padding
        
        if use_precise:
            lines = wrap_text_precise(text, available_width, font, font_size)
        else:
            max_chars_per_line = int(available_width / avg_char_width)
            lines = wrap_text_fast(text, max_chars_per_line)
        
        # Truncate lines that don't fit (considering paragraph spacing)
        available_height = height - 2 * padding
        max_lines = int(available_height / line_height)
        lines = lines[:max_lines]
        
        best_solution = (width, height, lines)

    # Final optimization: try to reduce height if there's too much empty space
    width, height, lines = best_solution
    num_paragraph_breaks = sum(1 for _, is_para_end in lines if is_para_end)
    actual_height_needed = len(lines) * line_height + num_paragraph_breaks * int(line_height * 0.0) + 2 * padding
    actual_height_needed = max(min_height, actual_height_needed)

    if actual_height_needed < height:
        height = actual_height_needed

    return (width, height, lines)


def text_to_adaptive_image(
    text: str,
    font_size: int = 8,
    padding: int = 8,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_path: Optional[str] = None,
    min_width: int = 28,
    max_width: int = 1024,
    min_height: int = 28,
    max_height: int = 1024,
    use_precise: bool = True,
    fast_mode: bool = True,
    compact_mode: bool = False,
    compact_symbol: str = COMPACT_NEWLINE_SYMBOL,
    highlight_configs: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Image.Image:
    """
    Convert text to image with ultimate optimization for maximum text coverage
    and minimum resolution while maintaining clarity.
    
    Args:
        text: Input text to render
        font_size: Font size (default 8 for dense packing)
        padding: Padding in pixels (optimized for minimal waste)
        bg_color: Background color RGB tuple
        text_color: Text color RGB tuple
        font_path: Custom font path (condensed fonts recommended)
        min_width: Minimum image width
        max_width: Maximum image width
        min_height: Minimum image height
        max_height: Maximum image height
        use_precise: Use precise font measurements (recommended, slightly slower but optimal)
        fast_mode: Use fast mode (fixed width) instead of binary search (much faster)
        compact_mode: Enable compact mode (replace newlines with symbols)
        compact_symbol: Symbol to use for newline replacement in compact mode
        highlight_configs: List of dicts specifying text contexts to highlight with colors.
                          To highlight compact_symbol, include it in highlight_configs.
                          Example: [{"context": "Action", "color": [255, 0, 0]}, 
                                   {"context": "⏎", "color": [128, 128, 128]}]
    
    Returns:
        PIL Image with optimally packed text
    """
    text = text.strip() if text else ""
    
    # Apply compact mode transformation if enabled
    if compact_mode:
        text = apply_compact_mode(text, compact_symbol)
    
    optimized_padding = padding

    min_width = max(min_width, 28)
    max_width = min(max_width, 1024)

    # Use cached font for significant speedup
    font = _get_cached_font(font_path, font_size)

    # Find dimensions - use fast mode for real-time performance
    if fast_mode:
        img_width, img_height, lines = find_fast_dimensions(
            text, font, font_size, optimized_padding, 
            min_width, max_width, min_height, max_height,
            use_precise=use_precise
        )
    else:
        img_width, img_height, lines = find_optimal_dimensions(
            text, font, font_size, optimized_padding, 
            min_width, max_width, min_height, max_height,
            use_precise=use_precise
        )
    
    # Get actual line height from font metrics
    _, line_height = get_font_metrics(font, font_size)

    # Create image with optimized dimensions
    img = Image.new('RGB', (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Render text with optimal spacing and 0.5x line spacing after paragraphs
    y_position = optimized_padding
    paragraph_spacing = int(line_height * 0.0)
    
    for line_text, is_paragraph_end in lines:
        if highlight_configs:
            # Render with highlighted contexts (also handles compact_symbol if defined in highlight_configs)
            _render_line_with_highlights(
                draw, line_text, optimized_padding, y_position,
                text_color, highlight_configs, font
            )
        else:
            draw.text((optimized_padding, y_position), line_text, fill=text_color, font=font)
        y_position += line_height
        # Add extra spacing after paragraph end
        if is_paragraph_end:
            y_position += paragraph_spacing
    
    return img


def _get_text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    """
    Get the actual rendering width of text using the most accurate method available.
    
    Args:
        font: Font to use
        text: Text to measure
    
    Returns:
        Width in pixels
    """
    if not text:
        return 0
    try:
        # getlength() is the most accurate method for text width (includes kerning)
        return int(font.getlength(text))
    except AttributeError:
        # Fallback for older PIL versions
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]
        except:
            return len(text) * 6  # Last resort fallback


def _render_line_with_highlights(
    draw: ImageDraw.ImageDraw,
    line_text: str,
    x: int,
    y: int,
    text_color: Tuple[int, int, int],
    highlight_configs: Optional[List[Dict[str, Any]]],
    font: ImageFont.FreeTypeFont
) -> None:
    """
    Render a line of text with multiple highlighted contexts in different colors.
    
    Uses cumulative prefix width calculation to ensure proper character spacing
    and kerning across segment boundaries.
    
    Args:
        draw: PIL ImageDraw object
        line_text: Text to render
        x: X position
        y: Y position
        text_color: Default color for regular text
        highlight_configs: List of dicts with 'context' and 'color' keys
                          Example: [{"context": "Action", "color": [255, 0, 0]}, 
                                   {"context": "Observation", "color": [0, 255, 0]}]
        font: Font to use
    """
    if not highlight_configs:
        # No highlights, render normally
        draw.text((x, y), line_text, fill=text_color, font=font)
        return
    
    # Build a list of (start_pos, end_pos, color) for all matches
    highlights = []
    for config in highlight_configs:
        context = config.get('context', '')
        color = tuple(config.get('color', text_color))
        if not context:
            continue
        
        # Find all occurrences of this context in the line
        start = 0
        while True:
            pos = line_text.find(context, start)
            if pos == -1:
                break
            highlights.append((pos, pos + len(context), color))
            start = pos + 1
    
    # Sort highlights by start position
    highlights.sort(key=lambda h: h[0])
    
    # Merge overlapping highlights (take the first one in case of overlap)
    merged_highlights = []
    for start, end, color in highlights:
        if merged_highlights and start < merged_highlights[-1][1]:
            # Overlapping, keep the existing one
            continue
        merged_highlights.append((start, end, color))
    
    # If no matches found, render normally
    if not merged_highlights:
        draw.text((x, y), line_text, fill=text_color, font=font)
        return
    
    # Build segments: list of (text, color, start_char_pos, end_char_pos)
    segments = []
    current_pos = 0
    
    for start, end, color in merged_highlights:
        # Add non-highlighted segment before this highlight
        if current_pos < start:
            segments.append((line_text[current_pos:start], text_color, current_pos, start))
        # Add highlighted segment
        segments.append((line_text[start:end], color, start, end))
        current_pos = end
    
    # Add remaining non-highlighted segment
    if current_pos < len(line_text):
        segments.append((line_text[current_pos:], text_color, current_pos, len(line_text)))
    
    # Render each segment using cumulative prefix width for positioning
    # This ensures proper kerning is considered
    for segment_text, segment_color, start_char_pos, end_char_pos in segments:
        # Calculate x position using prefix width (considers kerning)
        prefix = line_text[:start_char_pos]
        segment_x = x + _get_text_width(font, prefix)
        draw.text((segment_x, y), segment_text, fill=segment_color, font=font)


def text_to_adaptive_image_compact(
    text: str,
    font_size: int = 8,
    padding: int = 8,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_path: Optional[str] = None,
    min_width: int = 28,
    max_width: int = 1024,
    min_height: int = 28,
    max_height: int = 1024,
    use_precise: bool = False,
    compact_symbol: str = COMPACT_NEWLINE_SYMBOL,
    highlight_configs: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Tuple[Image.Image, int, str, List[Tuple[str, bool]]]:
    """
    Convert text to image in compact mode with caching metadata.
    
    This function is designed for incremental caching in compact mode:
    - Returns the number of complete lines and their character count
    - Returns the incomplete line's text for prepending to next render
    
    Args:
        text: Input text to render (newlines will be replaced with symbols)
        font_size: Font size
        padding: Padding in pixels
        bg_color: Background color RGB tuple
        text_color: Text color RGB tuple
        font_path: Custom font path
        min_width: Minimum image width
        max_width: Maximum image width
        min_height: Minimum image height
        max_height: Maximum image height
        use_precise: Use precise font measurements
        compact_symbol: Symbol to use for newline replacement
        highlight_configs: List of dicts specifying text contexts to highlight with colors.
                          To highlight compact_symbol, include it in highlight_configs.
                          Example: [{"context": "Action", "color": [255, 0, 0]}, 
                                   {"context": "⏎", "color": [128, 128, 128]}]
    
    Returns:
        Tuple of:
        - img: PIL Image with rendered text
        - num_complete_lines: Number of lines that filled the available width
        - incomplete_text: Text from the last incomplete line (for next render)
        - lines: List of (line_text, is_complete) tuples
    """
    text = text.strip() if text else ""
    
    # Apply compact mode transformation
    compact_text = apply_compact_mode(text, compact_symbol)
    
    optimized_padding = padding
    min_width = max(min_width, 28)
    max_width = min(max_width, 1024)
    
    # Use cached font
    font = _get_cached_font(font_path, font_size)
    
    # Get font metrics
    avg_char_width, line_height = get_font_metrics(font, font_size)
    
    # Use fixed width (fast mode) for compact rendering
    width = max_width
    available_width = width - 2 * padding
    max_chars_per_line = int(available_width / avg_char_width)
    
    # Wrap text with compact mode logic
    lines, complete_char_count, incomplete_text = wrap_text_compact(
        compact_text, max_chars_per_line
    )
    
    # Count complete lines
    num_complete_lines = sum(1 for _, is_complete in lines if is_complete)
    
    # Calculate required height
    required_height = len(lines) * line_height + 2 * padding
    height = max(min_height, min(max_height, required_height))
    
    # Truncate lines if needed
    if required_height > max_height:
        available_height = height - 2 * padding
        max_lines = int(available_height / line_height)
        lines = lines[:max_lines]
        # Recalculate complete lines and incomplete text
        num_complete_lines = sum(1 for _, is_complete in lines if is_complete)
        if lines:
            _, last_is_complete = lines[-1]
            if not last_is_complete:
                incomplete_text = lines[-1][0]
    
    # Create image
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Render text with colored symbols and highlights
    y_position = optimized_padding
    
    for line_text, _ in lines:
        if highlight_configs:
            # Render with highlighted contexts (also handles compact_symbol if defined in highlight_configs)
            _render_line_with_highlights(
                draw, line_text, optimized_padding, y_position,
                text_color, highlight_configs, font
            )
        else:
            draw.text((optimized_padding, y_position), line_text, fill=text_color, font=font)
        y_position += line_height
    
    return img, num_complete_lines, incomplete_text, lines


def trajectory_to_image(
    trajectory_text: str,
    font_size: int = 8,
    padding: int = 8,
    use_precise: bool = True,
    fast_mode: bool = True,
    compact_mode: bool = False,
    compact_symbol: str = COMPACT_NEWLINE_SYMBOL,
    highlight_configs: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Image.Image:
    """
    Transform trajectory text to image with ultimate optimization.
    Achieves maximum text coverage with minimum resolution while maintaining clarity.
    
    Args:
        trajectory_text: Trajectory text to render
        font_size: Font size (default 8 for optimal density)
        padding: Padding in pixels (optimized to 8 for minimal waste)
        use_precise: Use precise font measurements for optimal packing (recommended)
        fast_mode: Use fast mode (fixed width) for real-time performance (default True)
        compact_mode: Enable compact mode (replace newlines with symbols)
        compact_symbol: Symbol to use for newline replacement in compact mode
        highlight_configs: List of dicts specifying text contexts to highlight with colors.
                          To highlight compact_symbol, include it in highlight_configs.
                          Example: [{"context": "Action", "color": [255, 0, 0]}, 
                                   {"context": "⏎", "color": [128, 128, 128]}]
        **kwargs: Additional parameters passed to text_to_adaptive_image

    Returns:
        PIL Image object with optimally packed text
    """
    trajectory_text = trajectory_text.strip() if trajectory_text else ""
    formatted_text = trajectory_text

    return text_to_adaptive_image(
        formatted_text,
        font_size=font_size,
        padding=padding,
        use_precise=use_precise,
        fast_mode=fast_mode,
        compact_mode=compact_mode,
        compact_symbol=compact_symbol,
        highlight_configs=highlight_configs,
        **kwargs
    )