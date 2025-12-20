# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union, Optional, Dict, Any, Tuple
from PIL import Image, ImageOps
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from functools import lru_cache
import hashlib

from .base import BaseOCRTool
from .utils import (
    trajectory_to_image,
    text_to_adaptive_image_compact,
    apply_compact_mode,
    get_font_metrics,
    _get_cached_font,
    COMPACT_NEWLINE_SYMBOL,
    COMPACT_SYMBOL_COLOR
)


class OCRTool(BaseOCRTool):
    """
    OCR Tool for converting trajectory history records (text) into images.
    
    This tool is designed to be:
    - Highly flexible: Supports various trajectory formats and configurations
    - Decoupled: Works independently of the main pipeline
    - Easy to integrate: Minimal modifications needed to environment managers
    - Optimized for sliding windows: Segment-based caching supports non-contiguous history
    
    Caching Strategy (Segment-Based):
        - Instead of caching only full prefixes, we cache individual segments (lines split by \n)
        - Segments are split by newlines to match memory structure exactly
        - Each segment has its own content hash and height range in master image
        - Supports sliding window: Can match and reuse segments from any position
        - Format-agnostic: No dependency on specific patterns like "Observation X:"
        - Example: If context changes from "line 1-5" to "line 3-7", lines 3-5 are reused
    
    Master Image Structure:
        - master_img: Single concatenated image containing all cached segments
        - segments: List of segment metadata (content_hash, step, start_h, end_h, text)
        - indices: Dict for backward compatibility (full context hash -> position)
    """
    
    def __init__(
        self,
        enabled: bool = True,
        font_size: Optional[int] = 10,
        padding: int = 10,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        font_path: Optional[str] = None,
        min_width: int = 28,
        max_width: int = 1024,
        min_height: int = 28,
        max_height: int = 1024,
        max_workers: Optional[int] = None,
        use_parallel: bool = True,
        use_precise: bool = True,
        fast_mode: bool = True,
        enable_cache: bool = True,
        compact_mode: bool = False,
        compact_symbol: str = COMPACT_NEWLINE_SYMBOL,
        compact_symbol_color: Tuple[int, int, int] = COMPACT_SYMBOL_COLOR,
        **kwargs
    ):
        """
        Initialize the OCRTool with ultra-optimized settings for maximum text coverage
        and minimum resolution while maintaining clarity.
        
        Args:
            enabled: Whether the tool is enabled (can be toggled at runtime)
            font_size: Font size for text rendering
            padding: Padding around text in pixels
            bg_color: Background color as RGB tuple
            text_color: Text color as RGB tuple
            font_path: Path to custom font file
            min_width: Minimum image width in pixels
            max_width: Maximum image width in pixels
            min_height: Minimum image height in pixels
            max_height: Maximum image height in pixels
            max_workers: Maximum number of parallel workers (None for auto)
            use_parallel: Whether to use parallel processing for batch conversion
            use_precise: Use precise font measurements for optimal packing (recommended)
            fast_mode: Use fast mode (fixed width) for real-time performance (default True)
            enable_cache: Enable LRU caching of rendered images for speedup (default True)
            compact_mode: Enable compact mode (replace newlines with colored symbols)
            compact_symbol: Symbol to use for newline replacement in compact mode
            compact_symbol_color: RGB color for the compact symbol
            **kwargs: Additional parameters passed to trajectory_to_image
        """
        self.enabled = enabled
        self.font_size = font_size
        self.padding = padding
        self.bg_color = tuple(bg_color)
        self.text_color = tuple(text_color)
        self.font_path = font_path
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.max_workers = max_workers if max_workers is not None else min(32, (os.cpu_count() or 1) + 4)
        self.use_parallel = use_parallel
        self.use_precise = use_precise
        self.fast_mode = fast_mode
        self.enable_cache = enable_cache
        self.compact_mode = compact_mode
        self.compact_symbol = compact_symbol
        self.compact_symbol_color = tuple(compact_symbol_color)
        self.kwargs = kwargs
        # Initialize folder for saving trajectory images
        self.trajectory_images_dir = os.path.join(os.getcwd(), "logs/trajectory_images")
        os.makedirs(self.trajectory_images_dir, exist_ok=True)
        self.image_save_counter = 0
        # Incremental rendering: use master image + height indices to save memory
        # Format: {env_idx: {'master_img': np.ndarray, 'indices': {step_range_hash: (start, end)}}}
        self._master_images = {} if enable_cache else None
        # Cache statistics
        self._cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
        # Compact mode cache: stores incomplete line text for each environment
        # Format: {env_idx: {'incomplete_text': str, 'complete_lines_img': np.ndarray, 'complete_lines_count': int}}
        self._compact_cache = {} if enable_cache and compact_mode else None
        # Compact mode cache statistics
        self._compact_cache_stats = {
            'total': 0,              # Total render requests
            'full_hits': 0,          # Exact context hash matches (no re-render needed)
            'partial_hits': 0,       # Reused complete lines from cache
            'misses': 0,             # No cache to reuse (first render or context changed)
            'no_complete_lines': 0,  # Content too short to have complete lines to cache
            'cached_lines_reused': 0,  # Total number of cached lines reused
            'lines_rendered': 0,     # Total number of lines actually rendered
        }
    
    def convert(
        self,
        trajectory_text: Union[str, List[str]],
        **override_kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Convert trajectory text to image(s).
        
        Args:
            trajectory_text: Single trajectory text string or list of trajectory texts
            **override_kwargs: Parameters to override default configuration
        
        Returns:
            PIL Image object or list of PIL Image objects
        """
        if not self.is_enabled():
            return None if isinstance(trajectory_text, str) else [None] * len(trajectory_text)
        
        # Merge default config with override parameters
        config = self._get_config(**override_kwargs)
        
        # Handle both single string and list of strings
        if isinstance(trajectory_text, str):
            return self._convert_single(trajectory_text, config)
        else:
            return [self._convert_single(text, config) for text in trajectory_text]
    
    def convert_batch(
        self,
        trajectory_texts: List[str],
        **override_kwargs
    ) -> List[Image.Image]:
        """
        Convert a batch of trajectory texts to images with optional parallel processing.
        
        Args:
            trajectory_texts: List of trajectory text strings
            **override_kwargs: Parameters to override default configuration
        
        Returns:
            List of PIL Image objects
        """
        if not self.is_enabled():
            return [None] * len(trajectory_texts)
        
        if not trajectory_texts:
            return []
        
        # Merge default config with override parameters
        config = self._get_config(**override_kwargs)
        
        # Use parallel processing for batches larger than 1 if enabled
        if self.use_parallel and len(trajectory_texts) > 1:
            return self._convert_batch_parallel(trajectory_texts, config)
        else:
            return [self._convert_single(text, config) for text in trajectory_texts]
    
    def _convert_batch_parallel(
        self,
        trajectory_texts: List[str],
        config: Dict[str, Any]
    ) -> List[Image.Image]:
        """
        Convert a batch of trajectory texts to images using parallel processing.
        
        Args:
            trajectory_texts: List of trajectory text strings
            config: Configuration dictionary
        
        Returns:
            List of PIL Image objects (in the same order as input)
        """
        results = [None] * len(trajectory_texts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._convert_single, text, config): idx
                for idx, text in enumerate(trajectory_texts)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Fallback to blank image on error
                    results[idx] = Image.new(
                        'RGB',
                        (self.min_width, self.min_height),
                        self.bg_color
                    )
        
        return results
    
    def _convert_single(
        self,
        trajectory_text: str,
        config: Dict[str, Any]
    ) -> Image.Image:
        """
        Convert a single trajectory text to an image with optimized packing.
        
        Args:
            trajectory_text: Trajectory text string
            config: Configuration dictionary
        
        Returns:
            PIL Image object with optimally packed text
        """
        trajectory_text = trajectory_text.strip()
        if not trajectory_text:
            # Return a blank image if trajectory is empty
            return Image.new(
                'RGB',
                (self.min_width, self.min_height),
                self.bg_color
            )
        
        
        # Render image
        img = trajectory_to_image(
            trajectory_text,
            font_size=config['font_size'],
            padding=config['padding'],
            bg_color=config['bg_color'],
            text_color=config['text_color'],
            font_path=config['font_path'],
            min_width=config['min_width'],
            max_width=config['max_width'],
            min_height=config['min_height'],
            max_height=config['max_height'],
            use_precise=config['use_precise'],
            fast_mode=config['fast_mode'],
            compact_mode=config['compact_mode'],
            compact_symbol=config['compact_symbol'],
            compact_symbol_color=config['compact_symbol_color'],
            **config['extra_kwargs']
        )
        
        
        return img

    def _render_lines(
        self,
        lines: List[str],
        **override_kwargs
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Render a list of lines into a stacked image and return per-line height ranges.
        """
        if not lines:
            blank = self._get_blank_array(**override_kwargs)
            return blank, [(0, blank.shape[0])]
        
        # Render without any padding; padding will be added later in the pipeline.
        render_kwargs = {**override_kwargs, 'padding': 0, 'min_height': 0}
        images = self.convert_batch(lines, **render_kwargs)
        arrays = []
        ranges: List[Tuple[int, int]] = []
        current_h = 0
        
        for img in images:
            arr = np.array(img) if img is not None else self._get_blank_array(**override_kwargs)
            start_h = current_h
            current_h += arr.shape[0]
            ranges.append((start_h, current_h))
            arrays.append(arr)
        
        stacked = arrays[0] if len(arrays) == 1 else np.vstack(arrays)
        return stacked, ranges
    
    def _get_cache_key(self, text: str, config: Dict[str, Any]) -> str:
        """Generate a cache key for a text and config combination."""
        # Use hash for efficient key generation
        config_str = f"{config['font_size']}_{config['padding']}"
        config_str += f"_{config['min_width']}_{config['max_width']}_{config['use_precise']}_{config['fast_mode']}"
        key = f"{hash(text)}_{config_str}"
        return key
    
    def _get_config(self, **override_kwargs) -> Dict[str, Any]:
        """
        Get configuration dictionary, merging defaults with overrides.
        
        Args:
            **override_kwargs: Parameters to override
        
        Returns:
            Configuration dictionary
        """
        # Extract extra kwargs that are not direct parameters
        extra_kwargs = {}
        direct_params = {
            'font_size', 'padding',
            'bg_color', 'text_color', 'font_path', 'min_width', 'max_width',
            'min_height', 'max_height', 'use_precise', 'fast_mode',
            'compact_mode', 'compact_symbol', 'compact_symbol_color'
        }
        
        for key, value in override_kwargs.items():
            if key not in direct_params:
                extra_kwargs[key] = value
        
        # Merge with instance kwargs
        extra_kwargs = {**self.kwargs, **extra_kwargs}
        
        return {
            'font_size': override_kwargs.get('font_size', self.font_size),
            'padding': override_kwargs.get('padding', self.padding),
            'bg_color': override_kwargs.get('bg_color', self.bg_color),
            'text_color': override_kwargs.get('text_color', self.text_color),
            'font_path': override_kwargs.get('font_path', self.font_path),
            'min_width': override_kwargs.get('min_width', self.min_width),
            'max_width': override_kwargs.get('max_width', self.max_width),
            'min_height': override_kwargs.get('min_height', self.min_height),
            'max_height': override_kwargs.get('max_height', self.max_height),
            'use_precise': override_kwargs.get('use_precise', self.use_precise),
            'fast_mode': override_kwargs.get('fast_mode', self.fast_mode),
            'compact_mode': override_kwargs.get('compact_mode', self.compact_mode),
            'compact_symbol': override_kwargs.get('compact_symbol', self.compact_symbol),
            'compact_symbol_color': override_kwargs.get('compact_symbol_color', self.compact_symbol_color),
            'extra_kwargs': extra_kwargs
        }
    
    def is_enabled(self) -> bool:
        """
        Check if the OCR tool is enabled and ready to use.
        
        Returns:
            True if the tool is enabled, False otherwise
        """
        return self.enabled
    
    def enable(self):
        """Enable the OCR tool."""
        self.enabled = True
    
    def disable(self):
        """Disable the OCR tool."""
        self.enabled = False
    
    def enable_compact_mode(self):
        """
        Enable compact mode (replace newlines with colored symbols).
        Initializes compact cache if not already present.
        """
        self.compact_mode = True
        if self._compact_cache is None and self.enable_cache:
            self._compact_cache = {}
    
    def disable_compact_mode(self):
        """
        Disable compact mode (use normal newline rendering).
        Clears compact cache to free memory.
        """
        self.compact_mode = False
        if self._compact_cache is not None:
            self._compact_cache.clear()
    
    def is_compact_mode(self) -> bool:
        """Check if compact mode is enabled."""
        return self.compact_mode
    
    def set_compact_symbol(self, symbol: str, color: Optional[Tuple[int, int, int]] = None):
        """
        Set the symbol used for newline replacement in compact mode.
        
        Args:
            symbol: The symbol to use (e.g., '⏎', '↵', '¶')
            color: Optional RGB color tuple for the symbol
        """
        self.compact_symbol = symbol
        if color is not None:
            self.compact_symbol_color = tuple(color)
    
    def update_config(self, **kwargs):
        """
        Update configuration parameters at runtime.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
    
    def reset(self):
        """
        Reset the OCR tool state, clearing all caches and statistics.
        This is useful when starting a new episode or batch of episodes.
        """
        # Clear master images cache
        if self._master_images is not None:
            self._master_images.clear()
        
        # Clear compact mode cache
        if self._compact_cache is not None:
            self._compact_cache.clear()
        
        # Reset cache statistics
        self._cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
        
        # Reset compact mode cache statistics
        self._compact_cache_stats = {
            'total': 0,
            'full_hits': 0,
            'partial_hits': 0,
            'misses': 0,
            'no_complete_lines': 0,
            'cached_lines_reused': 0,
            'lines_rendered': 0,
        }
    
    def _find_matching_segments(self, context: str, env_idx: int) -> Optional[Tuple[List[str], List[Tuple[int, int]], List[Dict], int]]:
        """
        Find matching segments in cache for incremental rendering.
        Supports sliding window by matching individual segments rather than full prefixes.
        Segments are split by newlines (\n) to match memory structure.
        
        Args:
            context: Current trajectory context
            env_idx: Environment index
            
        Returns:
            (matched_segments, matched_ranges, matched_seg_infos, total_height) if found, None otherwise
            - matched_segments: List of matched segment texts (lines)
            - matched_ranges: List of (start_h, end_h) tuples for each matched segment
            - matched_seg_infos: List of segment info dicts (includes padding info)
            - total_height: Total height after all matched segments
        """
        if self._master_images is None or env_idx not in self._master_images:
            return None
        
        master_data = self._master_images[env_idx]
        segments = master_data.get('segments', [])
        
        if not segments:
            return None
        
        # Split context into segments by newlines (to match memory structure)
        context_segments = [line.strip() for line in context.split('\n') if line.strip()]
        
        if not context_segments:
            return None
        
        # Try to match segments from the beginning
        matched_segments = []
        matched_ranges = []
        matched_seg_infos = []
        
        for ctx_seg in context_segments:
            ctx_seg_hash = hash(ctx_seg)
            
            # Find matching segment in cache
            found = False
            for seg_info in segments:
                if seg_info['content_hash'] == ctx_seg_hash:
                    matched_segments.append(ctx_seg)
                    matched_ranges.append((seg_info['start_h'], seg_info['end_h']))
                    matched_seg_infos.append(seg_info)
                    found = True
                    break
            
            if not found:
                # No more consecutive matches, stop here
                break
        
        if matched_segments:
            # Calculate total height
            total_height = matched_ranges[-1][1] if matched_ranges else 0
            return (matched_segments, matched_ranges, matched_seg_infos, total_height)
        
        return None
    
    def _convert_incremental(
        self,
        trajectory_contexts: List[str],
        current_steps: List[int],
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Convert trajectory texts to images using incremental rendering with master image.
        
        Key optimizations:
        1. Uses master image + height indices to save memory (no redundant storage)
        2. Finds longest matching prefix (handles \n in actions)
        3. Supports sliding window (history_length < total_steps)
        4. Tracks cache hit rate
        
        Args:
            trajectory_contexts: List of trajectory text strings
            current_steps: List of current step numbers for each environment
            **override_kwargs: Override configuration parameters
        
        Returns:
            List of numpy arrays representing the images
        """
        if self._master_images is None:
            self._master_images = {}
        
        max_height = override_kwargs.get('max_height', self.max_height)
        image_arrays = []
        
        for env_idx, (context, current_step) in enumerate(zip(trajectory_contexts, current_steps)):
            self._cache_stats['total'] += 1
            
            context = context.strip() if context else ""
            if not context:
                # Empty context, return blank
                self._cache_stats['misses'] += 1
                image_arrays.append(self._get_blank_array(**override_kwargs))
                continue
            
            # Initialize master image for this environment if needed
            if env_idx not in self._master_images:
                self._master_images[env_idx] = {'master_img': None, 'indices': {}, 'segments': []}
            
            master_data = self._master_images[env_idx]
            context_hash = hash(context)
            
            # Check if we have this exact context cached (for backward compatibility)
            if context_hash in master_data.get('indices', {}):
                # Cache hit on exact context!
                self._cache_stats['hits'] += 1
                start_h, end_h, _, _ = master_data['indices'][context_hash]
                result = master_data['master_img'][start_h:end_h, :, :].copy()
                image_arrays.append(result)
                continue
            
            # Try to find matching segments for incremental rendering
            segment_match = self._find_matching_segments(context, env_idx)
            
            if segment_match is not None:
                # Cache hit on segments!
                matched_segments, matched_ranges, matched_seg_infos, total_height = segment_match
                
                # Extract the new content by using segment count instead of string matching
                # This avoids issues with whitespace/formatting differences
                context_segments = [line.strip() for line in context.split('\n') if line.strip()]
                num_matched = len(matched_segments)
                
                # The new segments are everything after the matched ones
                new_segments = context_segments[num_matched:]
                new_content = '\n'.join(new_segments) if new_segments else ''
                
                # Render only the new content
                if new_content:
                    self._cache_stats['hits'] += 1
                    # Render new content per line to get precise ranges
                    new_lines = [line.strip() for line in new_content.split('\n') if line.strip()]
                    new_array, new_ranges = self._render_lines(new_lines, **override_kwargs)
                    
                    # Use current_step as the step range (no pattern extraction needed)
                    step_start = current_step
                    step_end = current_step
                    
                    # Use _update_master_image to append new content with per-line ranges
                    self._update_master_image(env_idx, new_content, hash(new_content), new_array, new_ranges,
                                              step_start, step_end, **override_kwargs)
                    
                    # Combine matched segments with new content
                    combined_parts = []
                    
                    # Extract and combine matched segments
                    for start_h, end_h in matched_ranges:
                        segment_img = master_data['master_img'][start_h:end_h, :, :]
                        combined_parts.append(segment_img)
                    
                    combined_parts.append(new_array)
                    
                    combined = np.vstack(combined_parts) if combined_parts else new_array
                else:
                    # No new content, just use matched segments (sliding window case)
                    self._cache_stats['hits'] += 1
                    combined_parts = []
                    
                    # Extract and combine matched segments
                    for start_h, end_h in matched_ranges:
                        segment_img = master_data['master_img'][start_h:end_h, :, :]
                        combined_parts.append(segment_img)
                    
                    combined = np.vstack(combined_parts) if combined_parts else self._get_blank_array(**override_kwargs)
                
                # Note: We don't cache the combined context here because:
                # 1. matched_ranges may not start from 0, making (0, combined.shape[0]) incorrect
                # 2. segment matching already provides efficient lookup for repeated contexts
                
                image_arrays.append(combined)
            else:
                # Cache miss - render from scratch (per-line)
                self._cache_stats['misses'] += 1
                context_lines = [line.strip() for line in context.split('\n') if line.strip()]
                img_array, line_ranges = self._render_lines(context_lines, **override_kwargs)
                
                # Use current_step as the step range (no pattern extraction needed)
                step_start = current_step
                step_end = current_step
                
                # Update master image and indices
                self._update_master_image(env_idx, context, context_hash, img_array, line_ranges,
                                         step_start, step_end, **override_kwargs)
                
                image_arrays.append(img_array.copy())
            
            # # Periodically print cache stats
            if self._cache_stats['total'] % 256 == 0:
                self._print_cache_stats()
        
        return image_arrays
    
    def _convert_incremental_compact(
        self,
        trajectory_contexts: List[str],
        current_steps: List[int],
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Convert trajectory texts to images using compact mode with incremental caching.
        
        In compact mode:
        - Newlines are replaced with colored symbols (e.g., ⏎)
        - All content is treated as a single paragraph
        - Line wrapping happens due to fixed width
        - Complete lines (filled to width) are cached as images
        - Incomplete lines are kept as text and prepended to next render
        
        Caching Strategy:
        - Track the text that corresponds to cached complete lines
        - If new context starts with cached text, reuse cached image
        - Only render new content (incomplete text + new additions)
        
        Args:
            trajectory_contexts: List of trajectory text strings
            current_steps: List of current step numbers for each environment
            **override_kwargs: Override configuration parameters
        
        Returns:
            List of numpy arrays representing the images
        """
        if self._compact_cache is None:
            self._compact_cache = {}
        
        config = self._get_config(**override_kwargs)
        image_arrays = []
        
        for env_idx, (context, current_step) in enumerate(zip(trajectory_contexts, current_steps)):
            self._cache_stats['total'] += 1
            self._compact_cache_stats['total'] += 1
            
            context = context.strip() if context else ""
            if not context:
                self._cache_stats['misses'] += 1
                self._compact_cache_stats['misses'] += 1
                image_arrays.append(self._get_blank_array(**override_kwargs))
                continue
            
            # Initialize compact cache for this environment if needed
            if env_idx not in self._compact_cache:
                self._compact_cache[env_idx] = {
                    'complete_lines_img': None,
                    'complete_lines_count': 0,
                    'last_full_compact_text': '',  # Full compact text from last render
                    'incomplete_text': '',          # Remaining text (didn't fill a line)
                    'last_context_hash': None
                }
            
            cache_data = self._compact_cache[env_idx]
            context_hash = hash(context)
            
            # Apply compact mode transformation to get the full compact text
            compact_text = apply_compact_mode(context, config['compact_symbol'])
            
            # Check if this is the exact same context (full cache hit)
            if cache_data['last_context_hash'] == context_hash and cache_data['complete_lines_img'] is not None:
                self._cache_stats['hits'] += 1
                self._compact_cache_stats['full_hits'] += 1
                self._compact_cache_stats['cached_lines_reused'] += cache_data['complete_lines_count']
                # Reconstruct from cached complete lines + incomplete portion
                result = self._render_compact_with_cache(env_idx, context, config)
                image_arrays.append(result)
                continue
            
            # Check if new context EXTENDS the cached content (incremental hit)
            # We check if the new compact_text starts with the last full compact_text
            last_compact_text = cache_data.get('last_full_compact_text', '')
            can_reuse_cache = (
                cache_data['complete_lines_img'] is not None and
                last_compact_text and
                compact_text.startswith(last_compact_text)
            )
            
            if can_reuse_cache:
                # Incremental update: reuse cached complete lines, only render new content
                self._cache_stats['hits'] += 1
                self._compact_cache_stats['partial_hits'] += 1
                self._compact_cache_stats['cached_lines_reused'] += cache_data['complete_lines_count']
                
                # Get font metrics
                font = _get_cached_font(config['font_path'], config['font_size'])
                _, line_height = get_font_metrics(font, config['font_size'])
                
                # The new content is everything after the last full compact text
                new_addition = compact_text[len(last_compact_text):].strip()
                
                # Text to render = incomplete_text from before + new_addition
                if cache_data['incomplete_text']:
                    text_to_render = cache_data['incomplete_text'] + ' ' + new_addition
                else:
                    text_to_render = new_addition
                text_to_render = text_to_render.strip()
                
                if text_to_render:
                    # Render only the new content
                    new_img, new_complete_lines, new_incomplete_text, new_lines = text_to_adaptive_image_compact(
                        text_to_render,
                        font_size=config['font_size'],
                        padding=0,
                        bg_color=config['bg_color'],
                        text_color=config['text_color'],
                        font_path=config['font_path'],
                        min_width=config['min_width'],
                        max_width=config['max_width'],
                        min_height=0,
                        max_height=config['max_height'],
                        use_precise=config['use_precise'],
                        compact_symbol=config['compact_symbol'],
                        compact_symbol_color=config['compact_symbol_color']
                    )
                    new_img_array = np.array(new_img)
                    
                    # Track newly rendered lines
                    self._compact_cache_stats['lines_rendered'] += len(new_lines)
                    
                    # Combine cached complete lines with newly rendered content
                    combined = np.vstack([cache_data['complete_lines_img'], new_img_array])
                    
                    # Update cache
                    if new_complete_lines > 0:
                        new_complete_height = new_complete_lines * line_height
                        
                        # New cached image = old cached + new complete lines portion
                        total_cached_height = cache_data['complete_lines_img'].shape[0] + new_complete_height
                        cache_data['complete_lines_img'] = combined[:total_cached_height, :, :].copy()
                        cache_data['complete_lines_count'] += new_complete_lines
                    
                    cache_data['incomplete_text'] = new_incomplete_text
                    cache_data['last_full_compact_text'] = compact_text  # Store the full compact text
                    cache_data['last_context_hash'] = context_hash
                    
                    image_arrays.append(combined)
                else:
                    # No new content, just use cached (shouldn't happen often)
                    cache_data['last_full_compact_text'] = compact_text
                    cache_data['last_context_hash'] = context_hash
                    result = self._render_compact_with_cache(env_idx, context, config)
                    image_arrays.append(result)
            else:
                # Cache miss or context doesn't extend cached content - full re-render
                self._cache_stats['misses'] += 1
                
                # Render the full compact text
                img, num_complete_lines, incomplete_text, lines = text_to_adaptive_image_compact(
                    compact_text,
                    font_size=config['font_size'],
                    padding=0,
                    bg_color=config['bg_color'],
                    text_color=config['text_color'],
                    font_path=config['font_path'],
                    min_width=config['min_width'],
                    max_width=config['max_width'],
                    min_height=0,
                    max_height=config['max_height'],
                    use_precise=config['use_precise'],
                    compact_symbol=config['compact_symbol'],
                    compact_symbol_color=config['compact_symbol_color']
                )
                
                img_array = np.array(img)
                
                # Get font metrics for height calculations
                font = _get_cached_font(config['font_path'], config['font_size'])
                _, line_height = get_font_metrics(font, config['font_size'])
                
                # Track lines rendered
                self._compact_cache_stats['lines_rendered'] += len(lines)
                
                # Update cache with complete lines
                if num_complete_lines > 0:
                    complete_height = num_complete_lines * line_height
                    cache_data['complete_lines_img'] = img_array[:complete_height, :, :].copy()
                    cache_data['complete_lines_count'] = num_complete_lines
                    # This is a real miss (had to re-render with cacheable content)
                    self._compact_cache_stats['misses'] += 1
                else:
                    cache_data['complete_lines_img'] = None
                    cache_data['complete_lines_count'] = 0
                    # Content too short to fill a complete line - not a cache failure
                    self._compact_cache_stats['no_complete_lines'] += 1
                
                cache_data['incomplete_text'] = incomplete_text
                cache_data['last_full_compact_text'] = compact_text  # Store full compact text for next comparison
                cache_data['last_context_hash'] = context_hash
                
                image_arrays.append(img_array)
            
            # Periodically print compact cache stats
            if self._compact_cache_stats['total'] % 256 == 0:
                self._print_compact_cache_stats()
        
        return image_arrays
    
    def _render_compact_with_cache(
        self,
        env_idx: int,
        context: str,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Render compact mode image using cached complete lines.
        
        Args:
            env_idx: Environment index
            context: Current context text
            config: Configuration dictionary
        
        Returns:
            Rendered image as numpy array
        """
        cache_data = self._compact_cache[env_idx]
        
        # If no cached complete lines, render from scratch
        if cache_data['complete_lines_img'] is None:
            img, _, incomplete_text, _ = text_to_adaptive_image_compact(
                context,
                font_size=config['font_size'],
                padding=0,
                bg_color=config['bg_color'],
                text_color=config['text_color'],
                font_path=config['font_path'],
                min_width=config['min_width'],
                max_width=config['max_width'],
                min_height=0,
                max_height=config['max_height'],
                use_precise=config['use_precise'],
                compact_symbol=config['compact_symbol'],
                compact_symbol_color=config['compact_symbol_color']
            )
            cache_data['incomplete_text'] = incomplete_text
            return np.array(img)
        
        # Render only the incomplete portion and combine with cached complete lines
        incomplete_text = cache_data['incomplete_text']
        
        if incomplete_text:
            # Render the incomplete text
            img, _, new_incomplete, _ = text_to_adaptive_image_compact(
                incomplete_text,
                font_size=config['font_size'],
                padding=0,
                bg_color=config['bg_color'],
                text_color=config['text_color'],
                font_path=config['font_path'],
                min_width=config['min_width'],
                max_width=config['max_width'],
                min_height=0,
                max_height=config['max_height'],
                use_precise=config['use_precise'],
                compact_symbol=config['compact_symbol'],
                compact_symbol_color=config['compact_symbol_color']
            )
            incomplete_img = np.array(img)
            
            # Combine cached complete lines with incomplete line render
            combined = np.vstack([cache_data['complete_lines_img'], incomplete_img])
            return combined
        else:
            # No incomplete text, just return cached complete lines
            return cache_data['complete_lines_img'].copy()
    
    def _update_master_image(self, env_idx: int, context: str, context_hash: int,
                            new_img: np.ndarray, line_ranges: Optional[List[Tuple[int, int]]],
                            step_start: int, step_end: int,
                            **override_kwargs):
        """
        Update master image for an environment by appending new content.
        Stores individual segments (lines split by \n) to support sliding window matching.
        
        Optimized strategy: Directly use the pre-rendered image without re-rendering.
        Each new_content (already rendered) is treated as one or more segments.
        
        Args:
            env_idx: Environment index
            context: Full context string (used to extract line segments)
            context_hash: Hash of context (for backward compatibility)
            new_img: Pre-rendered image to append (already rendered, no re-rendering needed)
            step_start: Starting step number of this context
            step_end: Ending step number of this context
        """
        master_data = self._master_images[env_idx]
        
        # Initialize segments list if needed
        if 'segments' not in master_data:
            master_data['segments'] = []
        if 'indices' not in master_data:
            master_data['indices'] = {}  # Keep for backward compatibility
        
        # Split context into segments by newlines (to match memory structure)
        context_lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        # Append the pre-rendered image to master image
        if master_data['master_img'] is None:
            master_data['master_img'] = new_img
            start_h = 0
            end_h = new_img.shape[0]
        else:
            start_h = master_data['master_img'].shape[0]
            master_data['master_img'] = np.vstack([master_data['master_img'], new_img])
            end_h = master_data['master_img'].shape[0]
        
        # Store each line as a separate segment for cache matching.
        # If line_ranges is provided, use precise heights per line; otherwise fall back to the whole block.
        if line_ranges:
            ranges_iter = [(start_h + s, start_h + e) for (s, e) in line_ranges]
        else:
            ranges_iter = [(start_h, end_h)] * max(len(context_lines), 1)
        for line, (seg_start, seg_end) in zip(context_lines, ranges_iter):
            line_hash = hash(line)
            exists = any(seg['content_hash'] == line_hash for seg in master_data['segments'])
            if not exists:
                master_data['segments'].append({
                    'content_hash': line_hash,
                    'step': step_end,
                    'start_h': seg_start,
                    'end_h': seg_end,
                    'text': line
                })
        
        # Store index for backward compatibility (for exact context matching)
        master_data['indices'][context_hash] = (start_h, end_h, step_start, step_end)
        
    
    
    def _print_cache_stats(self):
        """Print cache hit rate statistics."""
        stats = self._cache_stats
        total = stats['total']
        hits = stats['hits']
        misses = stats['misses']
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        print(f"[OCR Cache] Total: {total}, Hits: {hits}, Misses: {misses}, Hit Rate: {hit_rate:.1f}%")
    
    def get_cache_stats(self):
        """Get cache statistics."""
        stats = self._cache_stats
        total = stats['total']
        hits = stats['hits']
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'hits': hits,
            'misses': stats['misses'],
            'hit_rate': f'{hit_rate:.1f}%'
        }
    
    def _print_compact_cache_stats(self):
        """Print compact mode cache statistics with detailed breakdown."""
        stats = self._compact_cache_stats
        total = stats['total']
        full_hits = stats['full_hits']
        partial_hits = stats['partial_hits']
        misses = stats['misses']
        no_complete = stats['no_complete_lines']
        cached_lines = stats['cached_lines_reused']
        rendered_lines = stats['lines_rendered']
        
        # Calculate rates (exclude no_complete_lines from miss rate since it's not a cache failure)
        cacheable_total = total - no_complete
        full_hit_rate = (full_hits / cacheable_total * 100) if cacheable_total > 0 else 0
        partial_hit_rate = (partial_hits / cacheable_total * 100) if cacheable_total > 0 else 0
        total_hit_rate = ((full_hits + partial_hits) / cacheable_total * 100) if cacheable_total > 0 else 0
        
        # Calculate line-level savings
        total_lines = cached_lines + rendered_lines
        line_savings = (cached_lines / total_lines * 100) if total_lines > 0 else 0
        
        print(f"[OCR Compact Cache] Total: {total} | "
              f"Full Hits: {full_hits} ({full_hit_rate:.1f}%) | "
              f"Partial Hits: {partial_hits} ({partial_hit_rate:.1f}%) | "
              f"Misses: {misses} | "
              f"NoCache: {no_complete} | "
              f"Hit Rate: {total_hit_rate:.1f}%")
        print(f"[OCR Compact Cache] Lines Reused: {cached_lines} | "
              f"Lines Rendered: {rendered_lines} | "
              f"Line Savings: {line_savings:.1f}%")
    
    def get_compact_cache_stats(self):
        """Get compact mode cache statistics."""
        stats = self._compact_cache_stats
        total = stats['total']
        full_hits = stats['full_hits']
        partial_hits = stats['partial_hits']
        no_complete = stats['no_complete_lines']
        cached_lines = stats['cached_lines_reused']
        rendered_lines = stats['lines_rendered']
        
        # Calculate hit rate excluding non-cacheable requests
        cacheable_total = total - no_complete
        total_hit_rate = ((full_hits + partial_hits) / cacheable_total * 100) if cacheable_total > 0 else 0
        total_lines = cached_lines + rendered_lines
        line_savings = (cached_lines / total_lines * 100) if total_lines > 0 else 0
        
        return {
            'total': total,
            'full_hits': full_hits,
            'partial_hits': partial_hits,
            'misses': stats['misses'],
            'no_complete_lines': no_complete,
            'hit_rate': f'{total_hit_rate:.1f}%',
            'cached_lines_reused': cached_lines,
            'lines_rendered': rendered_lines,
            'line_savings': f'{line_savings:.1f}%'
        }
    
    def convert_texts_to_images(
        self,
        trajectory_contexts: Optional[List[str]],
        batch_size: Optional[int] = None,
        active_masks: Optional[List[bool]] = None,
        save_img: bool = False,
        compression_factor: Optional[List[float]] = None,
        resample_method: int = Image.LANCZOS,
        current_steps: Optional[List[int]] = None,
        enable_cache: bool = True,
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Unified method to convert trajectory texts to images or create blank images if no history.
        
        Args:
            trajectory_contexts: List of trajectory text strings (from memory.fetch()), or None/empty for blank images
            batch_size: Number of images to create (required if trajectory_contexts is None/empty)
            active_masks: List of boolean masks indicating which trajectories are active. If False, renders blank image.
            save_img: Whether to save the generated images to disk
            compression_factor: List of compression factors (one per image, should be >= 1.0). If None, no compression applied.
            resample_method: PIL resampling method for compression (default: Image.LANCZOS for best quality)
            current_steps: List of current step numbers for each environment (for incremental rendering)
            enable_cache: Enable cache-based rendering mode (requires current_steps)
            **override_kwargs: Parameters to override default configuration (can include 'step_info', 'env_idx' for custom filenames)
        
        Returns:
            List of numpy arrays representing the images
        """
        if not self.is_enabled():
            if batch_size is not None:
                return np.array([]).reshape(0, *self._get_blank_image_shape(**override_kwargs))
            return np.array([])
        
        # Rendering happens without padding and without enforced min height;
        # padding is applied only after optional compression.
        render_kwargs = {**override_kwargs, 'padding': 0, 'min_height': 0}
        
        # If no trajectory contexts provided, create blank images
        if trajectory_contexts is None or len(trajectory_contexts) == 0:
            if batch_size is None:
                raise ValueError("batch_size must be provided when trajectory_contexts is None or empty")
            image_arrays = self.create_blank_images(batch_size, **override_kwargs)
        else:
            # If active_masks is None, set all to True
            if active_masks is None:
                active_masks = [True] * len(trajectory_contexts)
            
            if len(active_masks) != len(trajectory_contexts):
                raise ValueError(f"Length of active_masks ({len(active_masks)}) must match length of trajectory_contexts ({len(trajectory_contexts)})")
            
            # Pre-create blank array for inactive entries
            width = override_kwargs.get('min_width', self.min_width)
            height = override_kwargs.get('min_height', self.min_height)
            bg_color = override_kwargs.get('bg_color', self.bg_color)
            blank_img = Image.new('RGB', (width, height), bg_color)
            blank_array = np.array(blank_img)
            
            # Separate active and inactive indices
            active_indices = [i for i, mask in enumerate(active_masks) if mask]
            inactive_indices = [i for i, mask in enumerate(active_masks) if not mask]
            
            # Only process active trajectories
            if active_indices:
                active_contexts = [trajectory_contexts[i] for i in active_indices]
                active_current_steps = [current_steps[i] for i in active_indices] if current_steps is not None else None
                
                # Incremental rendering mode for active trajectories
                if enable_cache and active_current_steps is not None and self.enable_cache:
                    compact_mode = override_kwargs.get('compact_mode', self.compact_mode)
                    if compact_mode:
                        active_image_arrays = self._convert_incremental_compact(
                            active_contexts, active_current_steps, **render_kwargs
                        )
                    else:
                        active_image_arrays = self._convert_incremental(
                            active_contexts, active_current_steps, **render_kwargs
                        )
                else:
                    # Normal rendering mode for active trajectories
                    active_images = self.convert_batch(active_contexts, **render_kwargs)
                    active_image_arrays = []
                    for img in active_images:
                        if img is not None:
                            active_image_arrays.append(np.array(img))
                        else:
                            active_image_arrays.append(blank_array.copy())
            else:
                active_image_arrays = []
            
            # Reconstruct full array with blanks for inactive entries
            image_arrays = [None] * len(trajectory_contexts)
            for idx, img_array in zip(active_indices, active_image_arrays):
                image_arrays[idx] = img_array
            for idx in inactive_indices:
                image_arrays[idx] = blank_array.copy()
        
        # Apply compression if specified
        if compression_factor is not None:
            if len(compression_factor) != len(image_arrays):
                raise ValueError(f"Length of compression_factor ({len(compression_factor)}) must match length of image_arrays ({len(image_arrays)})")
            invalid_factors = [cf for cf in compression_factor if cf < 1.0]
            if invalid_factors:
                raise ValueError(f"All compression_factors must be >= 1.0, got {invalid_factors}")
            # Only compress if at least one factor > 1.0 (compress_image_arrays handles cf == 1.0 by skipping)
            if any(cf > 1.0 for cf in compression_factor):
                image_arrays = self.compress_image_arrays(
                    image_arrays,
                    compression_factor=compression_factor,
                    resample_method=resample_method
                )
        
        # Apply padding after compression so that borders are not compressed.
        padding_to_add = override_kwargs.get('padding', self.padding)
        if padding_to_add and padding_to_add > 0:
            bg_color = override_kwargs.get('bg_color', self.bg_color)
            image_arrays = [
                self._add_padding_to_array(arr, padding_to_add, bg_color)
                for arr in image_arrays
            ]
        
        # Save images if requested (save after compression to save disk space)
        if save_img and image_arrays:
            self._save_images(image_arrays, **override_kwargs)
        
        return image_arrays
    
    def _get_blank_image_shape(self, **override_kwargs) -> Tuple[int, int, int]:
        """Get the shape of a blank image (H, W, 3)."""
        width = override_kwargs.get('min_width', self.min_width)
        height = override_kwargs.get('min_height', self.min_height)
        return (height, width, 3)
    
    def _get_blank_array(self, **override_kwargs) -> np.ndarray:
        """Get a blank image as numpy array."""
        width = override_kwargs.get('min_width', self.min_width)
        height = override_kwargs.get('min_height', self.min_height)
        bg_color = override_kwargs.get('bg_color', self.bg_color)
        blank_img = Image.new('RGB', (width, height), bg_color)
        return np.array(blank_img)
    
    def create_blank_images(
        self,
        batch_size: int,
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Create a batch of blank images (useful for first step when there's no history).
        
        Args:
            batch_size: Number of blank images to create
            **override_kwargs: Parameters to override default configuration (e.g., min_width, min_height, bg_color)
        
        Returns:
            List of numpy arrays representing the blank images
        """
        if not self.is_enabled():
            return np.array([])
        
        width = override_kwargs.get('min_width', self.min_width)
        height = override_kwargs.get('min_height', self.min_height)
        bg_color = override_kwargs.get('bg_color', self.bg_color)
        
        blank_image = Image.new('RGB', (width, height), bg_color)
        blank_array = np.array(blank_image)
        # Stack the same blank image batch_size times
        return [blank_array] * batch_size
    
    def compress_image_arrays(
        self,
        image_arrays: List[np.ndarray],
        compression_factor: List[float],
        keep_aspect_ratio: bool = True,
        resample_method: int = Image.LANCZOS
    ) -> List[np.ndarray]:
        """
        Compress image arrays by a given factor while maintaining image clarity.
        
        Uses high-quality resampling (Lanczos by default) to preserve sharpness and details
        during downscaling. This is particularly useful for reducing memory usage and 
        computational costs while keeping OCR-readable images.
        
        Args:
            image_arrays: List of numpy arrays to compress
            compression_factor: List of factors by which to compress each image (e.g., 2.0 means halving the dimensions)
                              Must be >= 1.0 (1.0 = no compression, > 1.0 = compress). One factor per image.
            keep_aspect_ratio: Whether to maintain the original aspect ratio (default: True)
            resample_method: PIL resampling method. Options:
                           - Image.LANCZOS (default): Highest quality for downsampling
                           - Image.BICUBIC: Good quality, faster than Lanczos
                           - Image.BILINEAR: Faster but lower quality
                           - Image.NEAREST: Fastest but lowest quality
        
        Returns:
            List of compressed image arrays
        
        Examples:
            >>> # Compress batch of images with different factors per image
            >>> compressed_batch = ocr_tool.compress_image_arrays(images, [1.5, 2.0, 1.0])
            
            >>> # Use faster but lower quality resampling
            >>> compressed = ocr_tool.compress_image_arrays(images, [2.0, 2.0], resample_method=Image.BICUBIC)
        """
        if len(compression_factor) != len(image_arrays):
            raise ValueError(f"Length of compression_factor ({len(compression_factor)}) must match length of image_arrays ({len(image_arrays)})")
        
        for cf in compression_factor:
            if cf < 1.0:
                raise ValueError(f"All compression_factors must be >= 1.0, got {cf}")
        
        compressed_arrays = []
        
        for img_array, cf in zip(image_arrays, compression_factor):
            if img_array is None or not isinstance(img_array, np.ndarray):
                compressed_arrays.append(img_array)
                continue
            
            # Skip compression if factor is 1.0 (no compression)
            if cf == 1.0:
                compressed_arrays.append(img_array)
                continue
            
            # Get original dimensions
            height, width = img_array.shape[:2]
            
            # Calculate new dimensions
            new_width = max(28, int(width / cf))
            new_height = max(28, int(height / cf))
            
            # Ensure minimum dimensions for readability
            new_width = max(new_width, self.min_width)
            new_height = max(new_height, self.min_height)
            
            # Convert numpy array to PIL Image
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            
            img = Image.fromarray(img_array)
            
            # Resize using high-quality resampling
            compressed_img = img.resize((new_width, new_height), resample=resample_method)
            
            # Convert back to numpy array
            compressed_array = np.array(compressed_img)
            compressed_arrays.append(compressed_array)
        
        return compressed_arrays
    
    def _add_padding_to_array(
        self,
        img_array: Optional[np.ndarray],
        padding: int,
        bg_color: Tuple[int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Add uniform padding around an image array using the given background color.
        """
        if img_array is None or not isinstance(img_array, np.ndarray) or padding <= 0:
            return img_array
        
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        
        img = Image.fromarray(img_array)
        padded_img = ImageOps.expand(img, border=padding, fill=bg_color)
        return np.array(padded_img)
    
    def _save_images(
        self,
        image_arrays: List[np.ndarray],
        **kwargs
    ) -> None:
        """
        Save trajectory images to disk.
        
        Args:
            image_arrays: List of numpy arrays representing images
            **kwargs: Additional parameters for customizing filenames (e.g., 'step_info', 'env_idx')
        """
        from datetime import datetime
        
        step_info = kwargs.get('step_info', 'unknown')
        
        for i, img_array in enumerate(image_arrays):
            if img_array is not None:
                # Convert numpy array to PIL Image
                if isinstance(img_array, np.ndarray):
                    img = Image.fromarray(img_array.astype(np.uint8))
                else:
                    img = img_array
                
                # Create filename with optional custom info
                env_idx = kwargs.get('env_idx', i)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trajectory_env{env_idx}_{step_info}_{self.image_save_counter:06d}_{timestamp}.png"
                filepath = os.path.join(self.trajectory_images_dir, filename)
                img.save(filepath)
                # print(f"Saved trajectory image to: {filepath}")
        
        self.image_save_counter += 1

