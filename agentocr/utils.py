from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List
import re

def parse_trajectory_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract Observation and Action pairs from trajectory text.

    Args:
        text: The original trajectory text, can contain multiple rounds

    Returns:
        A list of tuples in the format [(obs_num, obs_text, action_text), ...]
    """
    pairs = []
    
    # Match [Observation N: '...', Action N: '...'] format
    pattern = r"\[Observation\s+(\d+):\s*'(.*?)',\s*Action\s+\d+:\s*'(.*?)'\]"
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        obs_num, obs_text, action_text = match
        # Unescape the text (handle \n, \t, etc.)
        obs_text = obs_text.replace('\\n', '\n').replace('\\t', '\t')
        action_text = action_text.replace('\\n', '\n').replace('\\t', '\t')
        
        # Replace all newlines with spaces to keep text on single line
        obs_text = obs_text.replace('\n', ' ').replace('\r', ' ')
        action_text = action_text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove multiple consecutive spaces
        obs_text = ' '.join(obs_text.split())
        action_text = ' '.join(action_text.split())
        
        pairs.append((obs_num, obs_text.strip(), action_text.strip()))
    
    return pairs


def format_trajectory_compact(pairs: List[Tuple[str, str, str]]) -> str:
    """
    Format Observation-Action pairs into a compact format without empty lines
    """
    lines = []
    for obs_num, obs_text, action_text in pairs:
        lines.append(f"[Observation {obs_num}]: {obs_text}")
        lines.append(f"[Action {obs_num}]: {action_text}")
    
    result = "\n".join(lines)
    return result


def wrap_text_fast(text: str, max_chars_per_line: int) -> List[str]:
    """
    Fast text wrapping based on character count.
    """
    lines = []
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            lines.append("")
            continue
        
        words = paragraph.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                
                if len(word) > max_chars_per_line:
                    for i in range(0, len(word), max_chars_per_line):
                        lines.append(word[i:i + max_chars_per_line])
                    current_line = ""
                else:
                    current_line = word
        
        if current_line:
            lines.append(current_line)
    
    return lines


def find_optimal_dimensions(
    text: str,
    font_size: int,
    padding: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int
) -> Tuple[int, int, List[str]]:
    """
    找到最优的图像尺寸 - 平衡宽度和高度，优先考虑可读性
    
    Returns:
        (width, height, wrapped_lines)
    """
    line_height = int(font_size * 1.5)
    avg_char_width = font_size * 0.6
    
    # 定义合理的宽度范围（字符数）
    # 字体大小固定为8
    min_chars = 100
    ideal_chars = 160
    
    # 计算对应的像素宽度
    ideal_width = int(ideal_chars * avg_char_width) + 2 * padding
    ideal_width = ((ideal_width + 27) // 28) * 28  # 调整到28的倍数
    ideal_width = max(min_width, min(max_width, ideal_width))
    
    # 先尝试理想宽度
    available_width = ideal_width - 2 * padding
    max_chars_per_line = int(available_width / avg_char_width)
    lines = wrap_text_fast(text, max_chars_per_line)
    
    # 计算需要的高度
    total_text_height = len(lines) * line_height
    required_height = total_text_height + 2 * padding
    
    # 如果高度合适，直接使用
    if required_height <= max_height:
        height = ((required_height + 27) // 28) * 28
        height = max(min_height, min(max_height, height))
        return (ideal_width, height, lines)
    
    # 如果高度超限，尝试增加宽度来减少行数
    best_solution = None
    best_waste = float('inf')  # 用浪费的空间作为评分标准
    
    # 尝试不同的宽度
    possible_widths = list(range(ideal_width, max_width + 1, 28 * 4))  # 更大的步长以提高速度
    
    for width in possible_widths:
        available_width = width - 2 * padding
        max_chars_per_line = int(available_width / avg_char_width)
        
        if max_chars_per_line < min_chars:
            continue
        
        lines = wrap_text_fast(text, max_chars_per_line)
        total_text_height = len(lines) * line_height
        required_height = total_text_height + 2 * padding
        
        if required_height <= max_height:
            height = ((required_height + 27) // 28) * 28
            height = max(min_height, min(max_height, height))
            
            # 计算浪费的空间（越少越好）
            used_area = len(lines) * max_chars_per_line * avg_char_width * line_height
            total_area = width * height
            waste = total_area - used_area
            
            if waste < best_waste:
                best_waste = waste
                best_solution = (width, height, lines)
            
            # 如果找到了合适的解决方案，可以提前退出
            if required_height < max_height * 0.9:  # 高度利用率不错
                break
    
    # 如果还是没找到，使用最大宽度并截断
    if best_solution is None:
        width = max_width
        height = max_height
        available_width = width - 2 * padding
        max_chars_per_line = int(available_width / avg_char_width)
        lines = wrap_text_fast(text, max_chars_per_line)
        
        available_height = height - 2 * padding
        max_lines = int(available_height / line_height)
        lines = lines[:max_lines]
        
        best_solution = (width, height, lines)
    
    return best_solution


def text_to_adaptive_image(
    text: str,
    font_size: int = 8,
    padding: int = 20,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_path: Optional[str] = None,
    min_width: int = 256,
    max_width: int = 1024,
    min_height: int = 256,
    max_height: int = 1024,
    **kwargs,
) -> Image.Image:
    """
    Convert text to image with optimized layout.
    Font size is fixed at 8.
    """
    
    # 确保尺寸是28的倍数
    def round_to_28(value: int, min_val: int, max_val: int) -> int:
        rounded = round(value / 28) * 28
        return max(min_val, min(max_val, rounded))
    
    min_width = max(round_to_28(min_width, 28, max_width), 252)
    max_width = round_to_28(max_width, min_width, 1024)
    min_height = max(round_to_28(min_height, 28, max_height), 252)
    max_height = round_to_28(max_height, min_height, 1024)
    
    # 加载字体
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("Arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # 找到最优尺寸
    img_width, img_height, lines = find_optimal_dimensions(
        text, font_size, padding, min_width, max_width, min_height, max_height
    )
    
    line_height = int(font_size * 1.5)

    # 创建图像
    img = Image.new('RGB', (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    # 绘制文本
    y_position = padding
    for line in lines:
        draw.text((padding, y_position), line, fill=text_color, font=font)
        y_position += line_height
    
    return img


def trajectory_to_image(
    trajectory_text: str,
    font_size: int = 8,
    padding: int = 20,
    compact_format: bool = True,
    **kwargs
) -> Image.Image:
    """
    Transform trajectory text to image with optimized layout.
    Font size is fixed at 8.
    
    Args:
        trajectory_text: trajectory text
        font_size: font size (fixed at 8)
        padding: padding
        compact_format: whether to use compact format
        **kwargs: other parameters passed to text_to_adaptive_image

    Returns:
        PIL Image object
    """
    pairs = []
    if "[Observation" in trajectory_text and "Action" in trajectory_text:
        pairs = parse_trajectory_text(trajectory_text)
        if pairs and compact_format:
            formatted_text = format_trajectory_compact(pairs)
        else:
            formatted_text = trajectory_text
    else:
        formatted_text = trajectory_text

    return text_to_adaptive_image(
        formatted_text,
        font_size,
        padding=padding,
        **kwargs
    )