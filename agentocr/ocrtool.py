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

from .base import BaseOCRTool
from .utils import trajectory_to_image


class OCRTool(BaseOCRTool):
    """
    OCR Tool for converting trajectory history records (text) into images.
    
    This tool is designed to be:
    - Highly flexible: Supports various trajectory formats and configurations
    - Decoupled: Works independently of the main pipeline
    - Easy to integrate: Minimal modifications needed to environment managers
    
    Usage:
        # Basic usage
        ocr_tool = OCRTool()
        image = ocr_tool.convert(trajectory_text)
        
        # Batch processing
        images = ocr_tool.convert_batch([traj1, traj2, traj3])
        
        # With custom configuration
        ocr_tool = OCRTool(
            font_size=12,
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
        font_size: Optional[int] = None,
        padding: int = 20,
        compact_format: bool = True,
        min_font_size: int = 10,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        font_path: Optional[str] = None,
        min_width: int = 256,
        max_width: int = 2048,
        min_height: int = 256,
        max_height: int = 2048,
        max_workers: Optional[int] = None,
        use_parallel: bool = True,
        **kwargs
    ):
        """
        Initialize the OCRTool.
        
        Args:
            enabled: Whether the tool is enabled (can be toggled at runtime)
            font_size: Font size for text rendering (None for auto-calculation)
            padding: Padding around text in pixels
            compact_format: Whether to use compact format for trajectory display
            min_font_size: Minimum font size for readability
            bg_color: Background color as RGB tuple
            text_color: Text color as RGB tuple
            font_path: Path to custom font file (None for default)
            min_width: Minimum image width in pixels
            max_width: Maximum image width in pixels
            min_height: Minimum image height in pixels
            max_height: Maximum image height in pixels
            max_workers: Maximum number of parallel workers (None for auto)
            use_parallel: Whether to use parallel processing for batch conversion
            **kwargs: Additional parameters passed to trajectory_to_image
        """
        self.enabled = enabled
        self.font_size = font_size
        self.padding = padding
        self.compact_format = compact_format
        self.min_font_size = min_font_size
        self.bg_color = tuple(bg_color)
        self.text_color = text_color
        self.font_path = font_path
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.max_workers = max_workers if max_workers is not None else min(32, (os.cpu_count() or 1) + 4)
        self.use_parallel = use_parallel
        self.kwargs = kwargs
    
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
        Convert a single trajectory text to an image.
        
        Args:
            trajectory_text: Trajectory text string
            config: Configuration dictionary
        
        Returns:
            PIL Image object
        """
        if not trajectory_text or not trajectory_text.strip():
            # Return a blank image if trajectory is empty
            return Image.new(
                'RGB',
                (self.min_width, self.min_height),
                self.bg_color
            )
        
        return trajectory_to_image(
            trajectory_text,
            font_size=config['font_size'],
            padding=config['padding'],
            compact_format=config['compact_format'],
            min_font_size=config['min_font_size'],
            bg_color=config['bg_color'],
            text_color=config['text_color'],
            font_path=config['font_path'],
            min_width=config['min_width'],
            max_width=config['max_width'],
            min_height=config['min_height'],
            max_height=config['max_height'],
            **config['extra_kwargs']
        )
    
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
            'font_size', 'padding', 'compact_format', 'min_font_size',
            'bg_color', 'text_color', 'font_path', 'min_width', 'max_width',
            'min_height', 'max_height'
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
            'min_font_size': override_kwargs.get('min_font_size', self.min_font_size),
            'bg_color': override_kwargs.get('bg_color', self.bg_color),
            'text_color': override_kwargs.get('text_color', self.text_color),
            'font_path': override_kwargs.get('font_path', self.font_path),
            'min_width': override_kwargs.get('min_width', self.min_width),
            'max_width': override_kwargs.get('max_width', self.max_width),
            'min_height': override_kwargs.get('min_height', self.min_height),
            'max_height': override_kwargs.get('max_height', self.max_height),
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
    
    def convert_texts_to_images(
        self,
        trajectory_contexts: Optional[List[str]],
        batch_size: Optional[int] = None,
        **override_kwargs
    ) -> List[np.ndarray]:
        """
        Unified method to convert trajectory texts to images or create blank images if no history.
        
        Args:
            trajectory_contexts: List of trajectory text strings (from memory.fetch()), or None/empty for blank images
            batch_size: Number of images to create (required if trajectory_contexts is None/empty)
            **override_kwargs: Parameters to override default configuration
        
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
            return self.create_blank_images(batch_size, **override_kwargs)
        
        # Convert trajectory texts to images
        images = self.convert_batch(trajectory_contexts, **override_kwargs)
        image_arrays = []
        for img in images:
            if img is not None:
                image_arrays.append(np.array(img))
            else:
                # Create blank image for None values to maintain batch size
                width = override_kwargs.get('min_width', self.min_width)
                height = override_kwargs.get('min_height', self.min_height)
                bg_color = override_kwargs.get('bg_color', self.bg_color)
                blank_img = Image.new('RGB', (width, height), bg_color)
                image_arrays.append(np.array(blank_img))
        
        return image_arrays
    
    def _get_blank_image_shape(self, **override_kwargs) -> Tuple[int, int, int]:
        """Get the shape of a blank image (H, W, 3)."""
        width = override_kwargs.get('min_width', self.min_width)
        height = override_kwargs.get('min_height', self.min_height)
        return (height, width, 3)
    
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

