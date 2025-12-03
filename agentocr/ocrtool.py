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
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from functools import lru_cache
import hashlib

from .base import BaseOCRTool
from .utils import trajectory_to_image


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
    
    Usage:
        # Basic usage
        ocr_tool = OCRTool()
        image = ocr_tool.convert(trajectory_text)
        
        # Batch processing
        images = ocr_tool.convert_batch([traj1, traj2, traj3])
        
        # With custom configuration
        ocr_tool = OCRTool(
            font_size=8,
            padding=30,
            compact_format=True
        )
        image = ocr_tool.convert(trajectory_text)
        
        # Integration with environment manager (minimal modification):
        # 
        # 1. Initialize in __init__:
        #    from agentocr import OCRTool
        #    self.ocr_tool = OCRTool(enabled=config.get('use_ocr', False))
        #
        # 2. Use in step() method:
        #    if self.ocr_tool and self.ocr_tool.is_enabled():
        #        trajectory_texts, _ = self.memory.fetch(
        #            history_length=self.config.env.history_length,
        #            obs_key="text_obs",
        #            action_key="action"
        #        )
        #        trajectory_images = self.ocr_tool.convert_texts_to_images(trajectory_texts)
        #        # Add to observations or use as needed
        #        # next_observations['trajectory_images'] = trajectory_images
    """
    
    def __init__(
        self,
        enabled: bool = True,
        font_size: Optional[int] = 10,
        padding: int = 10,
        compact_format: bool = True,
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
        **kwargs
    ):
        """
        Initialize the OCRTool with ultra-optimized settings for maximum text coverage
        and minimum resolution while maintaining clarity.
        
        Args:
            enabled: Whether the tool is enabled (can be toggled at runtime)
            font_size: Font size for text rendering
            padding: Padding around text in pixels
            compact_format: Whether to use compact format for trajectory display
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
            **kwargs: Additional parameters passed to trajectory_to_image
        """
        self.enabled = enabled
        self.font_size = font_size
        self.padding = padding
        self.compact_format = compact_format
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
        self.kwargs = kwargs
        # Initialize folder for saving trajectory images
        self.trajectory_images_dir = os.path.join(os.getcwd(), "logs/trajectory_images")
        os.makedirs(self.trajectory_images_dir, exist_ok=True)
        self.image_save_counter = 0
        # Image cache for fast repeated lookups
        self._image_cache = {}
        # Incremental rendering: use master image + height indices to save memory
        # Format: {env_idx: {'master_img': np.ndarray, 'indices': {step_range_hash: (start, end)}}}
        self._master_images = {} if enable_cache else None
        # Cache statistics
        self._cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
    
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
        if not trajectory_text:
            # Return a blank image if trajectory is empty
            return Image.new(
                'RGB',
                (self.min_width, self.min_height),
                self.bg_color
            )
        
        # Check cache if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(trajectory_text, config)
            if cache_key in self._image_cache:
                return self._image_cache[cache_key].copy()
        
        # Render image
        img = trajectory_to_image(
            trajectory_text,
            font_size=config['font_size'],
            padding=config['padding'],
            compact_format=config['compact_format'],
            bg_color=config['bg_color'],
            text_color=config['text_color'],
            font_path=config['font_path'],
            min_width=config['min_width'],
            max_width=config['max_width'],
            min_height=config['min_height'],
            max_height=config['max_height'],
            use_precise=config['use_precise'],
            fast_mode=config['fast_mode'],
            **config['extra_kwargs']
        )
        
        # Store in cache if enabled
        if self.enable_cache:
            # Limit cache size to prevent memory issues
            if len(self._image_cache) > 100:
                # Remove oldest entry (simple FIFO)
                self._image_cache.pop(next(iter(self._image_cache)))
            self._image_cache[cache_key] = img
        
        return img
    
    def _get_cache_key(self, text: str, config: Dict[str, Any]) -> str:
        """Generate a cache key for a text and config combination."""
        # Use hash for efficient key generation
        config_str = f"{config['font_size']}_{config['padding']}_{config['compact_format']}"
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
            'font_size', 'padding', 'compact_format',
            'bg_color', 'text_color', 'font_path', 'min_width', 'max_width',
            'min_height', 'max_height', 'use_precise', 'fast_mode'
        }
        
        for key, value in override_kwargs.items():
            if key not in direct_params:
                extra_kwargs[key] = value
        
        # Merge with instance kwargs
        extra_kwargs = {**self.kwargs, **extra_kwargs}
        
        return {
            'font_size': override_kwargs.get('font_size', self.font_size),
            'padding': override_kwargs.get('padding', self.padding),
            'compact_format': override_kwargs.get('compact_format', self.compact_format),
            'bg_color': override_kwargs.get('bg_color', self.bg_color),
            'text_color': override_kwargs.get('text_color', self.text_color),
            'font_path': override_kwargs.get('font_path', self.font_path),
            'min_width': override_kwargs.get('min_width', self.min_width),
            'max_width': override_kwargs.get('max_width', self.max_width),
            'min_height': override_kwargs.get('min_height', self.min_height),
            'max_height': override_kwargs.get('max_height', self.max_height),
            'use_precise': override_kwargs.get('use_precise', self.use_precise),
            'fast_mode': override_kwargs.get('fast_mode', self.fast_mode),
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
    
    def _combine_images_vertical(self, img1: np.ndarray, img2: np.ndarray, max_height: Optional[int] = None) -> np.ndarray:
        """
        Vertically concatenate two images.
        
        Args:
            img1: First image (top)
            img2: Second image (bottom)
            max_height: Maximum height for the combined image (will truncate from top if exceeded)
        
        Returns:
            Combined image as numpy array
        """
        combined = np.vstack([img1, img2])
        
        # Truncate from top if max_height exceeded
        if max_height is not None and combined.shape[0] > max_height:
            # Keep bottom portion (most recent history)
            combined = combined[-max_height:, :, :]
        
        return combined
    
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
        context_segments = [line for line in context.split('\n') if line]
        
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
                
                # Extract the new content (everything after matched segments)
                matched_text = '\n'.join(matched_segments)
                
                if context.startswith(matched_text):
                    # Perfect prefix match
                    new_content = context[len(matched_text):].lstrip('\n')
                else:
                    # Partial match - need to find where matched content ends
                    # This can happen with sliding windows
                    new_content = context
                    for seg in matched_segments:
                        if new_content.startswith(seg):
                            new_content = new_content[len(seg):].lstrip('\n')
                
                # Render only the new content
                if new_content:
                    self._cache_stats['hits'] += 1
                    new_img = self.convert_batch([new_content], **override_kwargs)[0]
                    new_array = np.array(new_img) if new_img is not None else self._get_blank_array(**override_kwargs)
                    
                    # Use current_step as the step range (no pattern extraction needed)
                    step_start = current_step
                    step_end = current_step
                    
                    # Use _update_master_image to append new content
                    self._update_master_image(env_idx, new_content, hash(new_content), new_array,
                                             step_start, step_end, **override_kwargs)
                    
                    # Get the combined image from master
                    # Combine matched segments with new content, removing padding between them
                    padding = override_kwargs.get('padding', self.padding)
                    if matched_ranges:
                        combined_parts = []
                        # Extract all matched segments from master image and remove padding
                        for idx, (start_h, end_h) in enumerate(matched_ranges):
                            segment_img = master_data['master_img'][start_h:end_h, :, :]
                            
                            # First segment: keep as is (with all padding)
                            if idx == 0:
                                # Remove bottom padding before next segment
                                if segment_img.shape[0] > padding:
                                    segment_img = segment_img[:-padding, :, :]
                            # Middle segments: remove top and bottom padding
                            elif idx < len(matched_ranges) - 1:
                                if segment_img.shape[0] > 2 * padding:
                                    segment_img = segment_img[padding:-padding, :, :]
                                elif segment_img.shape[0] > padding:
                                    segment_img = segment_img[padding:, :, :]
                            # Last segment: remove top and bottom padding (bottom for new content)
                            else:
                                if segment_img.shape[0] > 2 * padding:
                                    segment_img = segment_img[padding:-padding, :, :]
                                elif segment_img.shape[0] > padding:
                                    segment_img = segment_img[padding:, :, :]
                            
                            combined_parts.append(segment_img)
                        
                        # Remove top padding from new content before appending
                        if new_array.shape[0] > padding:
                            new_array_no_top = new_array[padding:, :, :]
                        else:
                            new_array_no_top = new_array
                        
                        combined_parts.append(new_array_no_top)
                        combined = np.vstack(combined_parts)
                    else:
                        combined = new_array
                else:
                    # No new content, just use matched segments (sliding window case)
                    self._cache_stats['hits'] += 1
                    padding = override_kwargs.get('padding', self.padding)
                    if matched_ranges:
                        combined_parts = []
                        # Extract all matched segments from master image and remove padding between them
                        for idx, (start_h, end_h) in enumerate(matched_ranges):
                            segment_img = master_data['master_img'][start_h:end_h, :, :]
                            
                            # First segment: keep as is (with all padding)
                            if idx == 0:
                                # Remove bottom padding if not the last segment
                                if idx < len(matched_ranges) - 1 and segment_img.shape[0] > padding:
                                    segment_img = segment_img[:-padding, :, :]
                            # Middle segments: remove top and bottom padding
                            elif idx < len(matched_ranges) - 1:
                                if segment_img.shape[0] > 2 * padding:
                                    segment_img = segment_img[padding:-padding, :, :]
                                elif segment_img.shape[0] > padding:
                                    segment_img = segment_img[padding:, :, :]
                            # Last segment: remove top padding only, keep bottom padding
                            else:
                                if segment_img.shape[0] > padding:
                                    segment_img = segment_img[padding:, :, :]
                            
                            combined_parts.append(segment_img)
                        
                        combined = np.vstack(combined_parts)
                    else:
                        combined = self._get_blank_array(**override_kwargs)
                
                # Cache this exact context for future use
                master_data['indices'][context_hash] = (0, combined.shape[0], current_step, current_step)
                
                # 检查是否需要清理
                max_master_height = override_kwargs.get('max_height', self.max_height) * 50
                if master_data.get('master_img') is not None and master_data['master_img'].shape[0] > max_master_height:
                    self._cleanup_master_image(env_idx, current_step)
                
                image_arrays.append(combined)
            else:
                # Cache miss - render from scratch
                self._cache_stats['misses'] += 1
                images = self.convert_batch([context], **override_kwargs)
                img_array = np.array(images[0]) if images[0] is not None else self._get_blank_array(**override_kwargs)
                
                # Use current_step as the step range (no pattern extraction needed)
                step_start = current_step
                step_end = current_step
                
                # Update master image and indices
                self._update_master_image(env_idx, context, context_hash, img_array,
                                         step_start, step_end, **override_kwargs)
                
                image_arrays.append(img_array.copy())
            
            # # Periodically print cache stats
            # if self._cache_stats['total'] % 100 == 0:
            #     self._print_cache_stats()
        
        return image_arrays
    
    def _update_master_image(self, env_idx: int, context: str, context_hash: int,
                            new_img: np.ndarray, step_start: int, step_end: int,
                            **override_kwargs):
        """
        Update master image for an environment by appending new content.
        Stores individual segments (lines split by \n) to support sliding window matching.
        
        Strategy: For single lines, use provided image. For multiple lines,
        render each separately for precise height tracking.
        
        When appending segments, removes top and bottom padding to avoid gaps.
        
        Args:
            env_idx: Environment index
            context: Full context string
            context_hash: Hash of context (for backward compatibility)
            new_img: Pre-rendered image to append (used for single lines or incremental additions)
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
        context_lines = [line for line in context.split('\n') if line]
        
        # Check if this is a first-time render with multiple new lines
        # In this case, render each line separately for better granularity
        should_render_separately = (
            len(context_lines) > 1 and
            # Check if any of these lines are already cached
            not any(
                any(seg['content_hash'] == hash(line) for seg in master_data['segments'])
                for line in context_lines
            )
        )
        
        if should_render_separately:
            # Multiple new lines - render each separately for precise height tracking
            for line in context_lines:
                line_hash = hash(line)
                
                # Render this line separately
                line_img = self.convert_batch([line], **override_kwargs)[0]
                line_array = np.array(line_img) if line_img is not None else self._get_blank_array(**override_kwargs)
                
                # Append to master image - keep segments complete with all padding
                if master_data['master_img'] is None:
                    master_data['master_img'] = line_array
                    start_h = 0
                    end_h = line_array.shape[0]
                else:
                    start_h = master_data['master_img'].shape[0]
                    master_data['master_img'] = np.vstack([master_data['master_img'], line_array])
                    end_h = master_data['master_img'].shape[0]
                
                # Store segment with its precise height range
                master_data['segments'].append({
                    'content_hash': line_hash,
                    'step': step_end,
                    'start_h': start_h,
                    'end_h': end_h,
                    'text': line
                })
            
            # Store index for the full context
            start_h = 0
            end_h = master_data['master_img'].shape[0] if master_data['master_img'] is not None else 0
        else:
            # Single line or incremental addition - use provided image
            if master_data['master_img'] is None:
                master_data['master_img'] = new_img
                start_h = 0
                end_h = new_img.shape[0]
            else:
                # Simply append without removing any padding
                # Padding will be removed during extraction/combination
                start_h = master_data['master_img'].shape[0]
                master_data['master_img'] = np.vstack([master_data['master_img'], new_img])
                end_h = master_data['master_img'].shape[0]
            
            # Store segment(s) - each line as a separate segment
            for line in context_lines:
                line_hash = hash(line)
                exists = any(seg['content_hash'] == line_hash for seg in master_data['segments'])
                
                if not exists:
                    master_data['segments'].append({
                        'content_hash': line_hash,
                        'step': step_end,
                        'start_h': start_h,
                        'end_h': end_h,
                        'text': line
                    })
        
        # Store index for backward compatibility (for exact context matching)
        master_data['indices'][context_hash] = (start_h, end_h, step_start, step_end)
        
        # Cleanup if master image gets too large
        max_master_height = override_kwargs.get('max_height', self.max_height) * 50
        if master_data['master_img'].shape[0] > max_master_height:
            self._cleanup_master_image(env_idx, step_end)
    
    def _cleanup_master_image(self, env_idx: int, current_step: int, keep_recent_steps: int = 10):
        """
        Clean up master image to prevent unbounded growth.
        Keeps only recent contexts/segments and rebuilds master image.
        
        Args:
            env_idx: Environment index
            current_step: Current step number
            keep_recent_steps: Number of recent step ranges to keep
        """
        if env_idx not in self._master_images:
            return
        
        master_data = self._master_images[env_idx]
        segments = master_data.get('segments', [])
        indices = master_data.get('indices', {})
        
        # Sort segments by step (most recent last)
        sorted_segments = sorted(segments, key=lambda x: x['step'])
        
        # Keep only recent segments
        keep_segments = sorted_segments[-keep_recent_steps:] if len(sorted_segments) > keep_recent_steps else sorted_segments
        
        if len(keep_segments) < len(sorted_segments):
            # Rebuild master image with only kept segments
            new_master = None
            new_segments = []
            current_h = 0
            
            for seg_info in keep_segments:
                old_start_h = seg_info['start_h']
                old_end_h = seg_info['end_h']
                
                # Extract this slice from old master
                slice_img = master_data['master_img'][old_start_h:old_end_h, :, :]
                
                if new_master is None:
                    new_master = slice_img
                else:
                    new_master = np.vstack([new_master, slice_img])
                
                # Update segment with new heights
                new_end_h = current_h + slice_img.shape[0]
                new_segments.append({
                    'content_hash': seg_info['content_hash'],
                    'step': seg_info['step'],
                    'start_h': current_h,
                    'end_h': new_end_h,
                    'text': seg_info.get('text', '')
                })
                current_h = new_end_h
            
            # Update master data
            master_data['master_img'] = new_master
            master_data['segments'] = new_segments
            
            # Clean up indices as well (remove stale entries)
            # Keep indices that reference steps in recent range
            min_keep_step = keep_segments[0]['step'] if keep_segments else current_step
            new_indices = {}
            for ctx_hash, (start_h, end_h, step_start, step_end) in indices.items():
                if step_end >= min_keep_step:
                    # Keep this index, but update heights if needed
                    # Note: Heights might be stale after cleanup, better to invalidate
                    pass  # Skip for now, let it be recomputed on next access
            master_data['indices'] = new_indices
    
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
    
    def convert_texts_to_images(
        self,
        trajectory_contexts: Optional[List[str]],
        batch_size: Optional[int] = None,
        save_img: bool = False,
        compression_factor: Optional[float] = 1.0,
        resample_method: int = Image.LANCZOS,
        current_steps: Optional[List[int]] = None,
        enable_cache: bool = False,
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Unified method to convert trajectory texts to images or create blank images if no history.
        
        Args:
            trajectory_contexts: List of trajectory text strings (from memory.fetch()), or None/empty for blank images
            batch_size: Number of images to create (required if trajectory_contexts is None/empty)
            save_img: Whether to save the generated images to disk
            compression_factor: compression factor (should be > 1.0). If None, no compression applied.
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
        
        # If no trajectory contexts provided, create blank images
        if trajectory_contexts is None or len(trajectory_contexts) == 0:
            if batch_size is None:
                raise ValueError("batch_size must be provided when trajectory_contexts is None or empty")
            image_arrays = self.create_blank_images(batch_size, **override_kwargs)
        # Incremental rendering mode
        elif enable_cache and current_steps is not None and self.enable_cache:
            image_arrays = self._convert_incremental(
                trajectory_contexts, current_steps, **override_kwargs
            )
        else:
            # Convert trajectory texts to images (normal mode)
            images = self.convert_batch(trajectory_contexts, **override_kwargs)
            
            # Optimize: Pre-create blank array for reuse
            width = override_kwargs.get('min_width', self.min_width)
            height = override_kwargs.get('min_height', self.min_height)
            bg_color = override_kwargs.get('bg_color', self.bg_color)
            blank_array = None
            
            image_arrays = []
            for img in images:
                if img is not None:
                    image_arrays.append(np.array(img))
                else:
                    # Reuse blank array to avoid repeated Image.new calls
                    if blank_array is None:
                        blank_img = Image.new('RGB', (width, height), bg_color)
                        blank_array = np.array(blank_img)
                    image_arrays.append(blank_array.copy())
        
        # Apply compression if specified
        if compression_factor > 1.0:
            image_arrays = self.compress_image_arrays(
                image_arrays,
                compression_factor=compression_factor,
                resample_method=resample_method
            )
        
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
        image_arrays: Union[np.ndarray, List[np.ndarray]],
        compression_factor: float,
        keep_aspect_ratio: bool = True,
        resample_method: int = Image.LANCZOS
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Compress image arrays by a given factor while maintaining image clarity.
        
        Uses high-quality resampling (Lanczos by default) to preserve sharpness and details
        during downscaling. This is particularly useful for reducing memory usage and 
        computational costs while keeping OCR-readable images.
        
        Args:
            image_arrays: Single numpy array or list of numpy arrays to compress
            compression_factor: Factor by which to compress the images (e.g., 2.0 means halving the dimensions)
                              Must be > 1 to reduce size.
            keep_aspect_ratio: Whether to maintain the original aspect ratio (default: True)
            resample_method: PIL resampling method. Options:
                           - Image.LANCZOS (default): Highest quality for downsampling
                           - Image.BICUBIC: Good quality, faster than Lanczos
                           - Image.BILINEAR: Faster but lower quality
                           - Image.NEAREST: Fastest but lowest quality
        
        Returns:
            Compressed image array(s) in the same format as input (single array or list)
        
        Examples:
            >>> # Compress single image by 2x (halve dimensions)
            >>> compressed = ocr_tool.compress_image_arrays(image, 2.0)
            
            >>> # Compress batch of images by 3x
            >>> compressed_batch = ocr_tool.compress_image_arrays(images, 3.0)
            
            >>> # Use faster but lower quality resampling
            >>> compressed = ocr_tool.compress_image_arrays(image, 2.0, resample_method=Image.BICUBIC)
        """
        if compression_factor <= 1.0:
            raise ValueError(f"compression_factor must be > 1.0, got {compression_factor}")
        
        # Handle single array vs list of arrays
        single_input = not isinstance(image_arrays, list)
        if single_input:
            image_arrays = [image_arrays]
        
        compressed_arrays = []
        
        for img_array in image_arrays:
            if img_array is None or not isinstance(img_array, np.ndarray):
                compressed_arrays.append(img_array)
                continue
            
            # Get original dimensions
            height, width = img_array.shape[:2]
            
            # Calculate new dimensions
            new_width = max(1, int(width / compression_factor))
            new_height = max(1, int(height / compression_factor))
            
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
        
        # Return in the same format as input
        return compressed_arrays[0] if single_input else compressed_arrays
    
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
                print(f"Saved trajectory image to: {filepath}")
        
        self.image_save_counter += 1

