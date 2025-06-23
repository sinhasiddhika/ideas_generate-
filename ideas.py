import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps
import io
import base64
from sklearn.cluster import KMeans
import cv2
from scipy import ndimage
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

def analyze_input_pattern(image):
    """Analyze the input pattern for creating variations"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    return {
        'original_size': (h, w),
        'tile_size': (min(h, w), min(h, w)),  # Square tile for variations
        'half_size': (h//2, w//2),
        'quarter_size': (h//4, w//4) if h > 100 and w > 100 else (h//2, w//2)
    }

def create_pattern_variation_1_original(image, output_width, output_height):
    """Variation 1: Original Pattern - Direct Tiling"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Pure tiling of original pattern
    tiles_x = (output_width + w - 1) // w
    tiles_y = (output_height + h - 1) // h
    
    if len(img_array.shape) == 3:
        tiled_array = np.tile(img_array, (tiles_y, tiles_x, 1))
    else:
        tiled_array = np.tile(img_array, (tiles_y, tiles_x))
    
    final_array = tiled_array[:output_height, :output_width]
    return Image.fromarray(final_array), "Original Pattern"

def create_pattern_variation_2_mirrored(image, output_width, output_height):
    """Variation 2: Mirrored Pattern - Creates symmetrical designs"""
    img_array = np.array(image)
    
    # Create mirrored version
    mirrored_h = np.hstack([img_array, np.fliplr(img_array)])
    mirrored_full = np.vstack([mirrored_h, np.flipud(mirrored_h)])
    
    # Tile this mirrored pattern
    mh, mw = mirrored_full.shape[:2]
    tiles_x = (output_width + mw - 1) // mw
    tiles_y = (output_height + mh - 1) // mh
    
    if len(mirrored_full.shape) == 3:
        tiled_array = np.tile(mirrored_full, (tiles_y, tiles_x, 1))
    else:
        tiled_array = np.tile(mirrored_full, (tiles_y, tiles_x))
    
    final_array = tiled_array[:output_height, :output_width]
    return Image.fromarray(final_array), "Mirrored Symmetry"

def create_pattern_variation_3_rotated(image, output_width, output_height):
    """Variation 3: Rotated Pattern - Creates dynamic layouts"""
    img_array = np.array(image)
    
    # Create 4-way rotated pattern
    img_pil = Image.fromarray(img_array)
    rot90 = np.array(img_pil.rotate(90, expand=True))
    rot180 = np.array(img_pil.rotate(180, expand=True))
    rot270 = np.array(img_pil.rotate(270, expand=True))
    
    # Resize all to same dimensions
    h, w = img_array.shape[:2]
    rot90_resized = np.array(Image.fromarray(rot90).resize((w, h)))
    rot180_resized = np.array(Image.fromarray(rot180).resize((w, h)))
    rot270_resized = np.array(Image.fromarray(rot270).resize((w, h)))
    
    # Combine in 2x2 grid
    top_row = np.hstack([img_array, rot90_resized])
    bottom_row = np.hstack([rot270_resized, rot180_resized])
    rotated_pattern = np.vstack([top_row, bottom_row])
    
    # Tile this rotated pattern
    rh, rw = rotated_pattern.shape[:2]
    tiles_x = (output_width + rw - 1) // rw
    tiles_y = (output_height + rh - 1) // rh
    
    if len(rotated_pattern.shape) == 3:
        tiled_array = np.tile(rotated_pattern, (tiles_y, tiles_x, 1))
    else:
        tiled_array = np.tile(rotated_pattern, (tiles_y, tiles_x))
    
    final_array = tiled_array[:output_height, :output_width]
    return Image.fromarray(final_array), "4-Way Rotated"

def create_pattern_variation_4_scaled(image, output_width, output_height):
    """Variation 4: IMPROVED Multi-Scale Pattern - Better proportions and placement"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create different scales with better proportions
    img_pil = Image.fromarray(img_array)
    
    # Create a base pattern with the original size
    base_pattern = img_array
    
    # Create smaller versions for filling
    small_size = max(w//4, h//4, 20)  # Ensure minimum size
    medium_size = max(w//2, h//2, 40)  # Ensure minimum size
    
    small_pattern = np.array(img_pil.resize((small_size, small_size)))
    medium_pattern = np.array(img_pil.resize((medium_size, medium_size)))
    
    # Create a repeating base with small patterns
    small_tiles_x = (output_width + small_size - 1) // small_size
    small_tiles_y = (output_height + small_size - 1) // small_size
    
    # Create background with small tiles
    background = np.tile(small_pattern, (small_tiles_y, small_tiles_x, 1))[:output_height, :output_width]
    
    # Overlay medium patterns at regular intervals
    med_step_x = w
    med_step_y = h
    
    for y in range(0, output_height - medium_size, med_step_y):
        for x in range(0, output_width - medium_size, med_step_x):
            if y + medium_size <= output_height and x + medium_size <= output_width:
                background[y:y+medium_size, x:x+medium_size] = medium_pattern
    
    # Add some original size patterns as focal points
    focal_step_x = w * 2
    focal_step_y = h * 2
    
    for y in range(h//2, output_height - h, focal_step_y):
        for x in range(w//2, output_width - w, focal_step_x):
            if y + h <= output_height and x + w <= output_width:
                background[y:y+h, x:x+w] = base_pattern
    
    return Image.fromarray(background), "Multi-Scale Mix"

def create_pattern_variation_5_brick(image, output_width, output_height):
    """Variation 5: IMPROVED Brick/Offset Pattern - Better offset calculation"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create brick pattern with proper offset
    offset = w // 2
    if offset == 0:
        offset = max(1, w // 4)  # Ensure some offset
    
    # Calculate tiles needed
    tiles_x = (output_width + w - 1) // w + 2  # Extra tiles for offset
    tiles_y = (output_height + h - 1) // h
    
    # Create canvas
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Fill with brick pattern
    for row in range(tiles_y):
        y_pos = row * h
        if y_pos >= output_height:
            break
            
        for col in range(tiles_x):
            x_pos = col * w
            
            # Apply offset to odd rows
            if row % 2 == 1:
                x_pos -= offset
            
            # Check boundaries and place tile
            if x_pos < output_width and y_pos < output_height:
                # Calculate actual placement area
                end_y = min(y_pos + h, output_height)
                end_x = min(x_pos + w, output_width)
                
                if x_pos >= 0:  # Only place if not negative
                    tile_h = end_y - y_pos
                    tile_w = end_x - x_pos
                    canvas[y_pos:end_y, x_pos:end_x] = img_array[:tile_h, :tile_w]
                elif x_pos + w > 0:  # Partial tile on left edge
                    start_x_tile = -x_pos
                    tile_w = end_x
                    tile_h = end_y - y_pos
                    canvas[y_pos:end_y, 0:tile_w] = img_array[:tile_h, start_x_tile:start_x_tile + tile_w]
    
    return Image.fromarray(canvas), "Brick/Offset Layout"

def create_pattern_variation_6_diagonal(image, output_width, output_height):
    """Variation 6: IMPROVED Diagonal Pattern - Better diamond arrangement"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create diamond/diagonal pattern
    img_pil = Image.fromarray(img_array)
    
    # Create a diamond-shaped arrangement
    # Calculate diagonal spacing
    diag_spacing_x = int(w * 0.8)
    diag_spacing_y = int(h * 0.8)
    
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Create diagonal grid
    for row in range(-2, (output_height // diag_spacing_y) + 3):
        for col in range(-2, (output_width // diag_spacing_x) + 3):
            # Calculate position with diagonal offset
            if row % 2 == 0:
                x_pos = col * diag_spacing_x
            else:
                x_pos = col * diag_spacing_x + diag_spacing_x // 2
            
            y_pos = row * diag_spacing_y
            
            # Place pattern if within bounds
            if (x_pos + w > 0 and x_pos < output_width and 
                y_pos + h > 0 and y_pos < output_height):
                
                # Calculate clipping
                start_x = max(0, x_pos)
                start_y = max(0, y_pos)
                end_x = min(output_width, x_pos + w)
                end_y = min(output_height, y_pos + h)
                
                # Calculate source clipping
                src_start_x = max(0, -x_pos)
                src_start_y = max(0, -y_pos)
                src_end_x = src_start_x + (end_x - start_x)
                src_end_y = src_start_y + (end_y - start_y)
                
                canvas[start_y:end_y, start_x:end_x] = img_array[src_start_y:src_end_y, src_start_x:src_end_x]
    
    return Image.fromarray(canvas), "Diagonal Diamond"

def create_pattern_variation_7_hexagonal(image, output_width, output_height):
    """Variation 7: Hexagonal/Honeycomb Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Hexagonal spacing
    hex_width = w * 0.75
    hex_height = h * 0.87  # sqrt(3)/2 for proper hex spacing
    
    row = 0
    y_pos = 0
    while y_pos < output_height:
        col = 0
        x_pos = 0
        
        # Offset every other row for hex pattern
        if row % 2 == 1:
            x_pos = hex_width / 2
        
        while x_pos < output_width:
            # Place pattern
            start_x = max(0, int(x_pos))
            start_y = max(0, int(y_pos))
            end_x = min(output_width, int(x_pos) + w)
            end_y = min(output_height, int(y_pos) + h)
            
            if start_x < end_x and start_y < end_y:
                src_w = end_x - start_x
                src_h = end_y - start_y
                canvas[start_y:end_y, start_x:end_x] = img_array[:src_h, :src_w]
            
            x_pos += hex_width
            col += 1
        
        y_pos += hex_height
        row += 1
    
    return Image.fromarray(canvas), "Hexagonal Grid"

def create_pattern_variation_8_circular(image, output_width, output_height):
    """Variation 8: Circular/Radial Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Create circular arrangement
    center_x, center_y = output_width // 2, output_height // 2
    radius_step = max(w, h)
    
    for radius in range(radius_step, max(output_width, output_height), radius_step):
        # Calculate number of patterns for this radius
        circumference = 2 * np.pi * radius
        num_patterns = max(1, int(circumference // max(w, h)))
        
        for i in range(num_patterns):
            angle = (2 * np.pi * i) / num_patterns
            x_pos = int(center_x + radius * np.cos(angle) - w // 2)
            y_pos = int(center_y + radius * np.sin(angle) - h // 2)
            
            # Place pattern if within bounds
            if (x_pos + w > 0 and x_pos < output_width and 
                y_pos + h > 0 and y_pos < output_height):
                
                start_x = max(0, x_pos)
                start_y = max(0, y_pos)
                end_x = min(output_width, x_pos + w)
                end_y = min(output_height, y_pos + h)
                
                src_start_x = max(0, -x_pos)
                src_start_y = max(0, -y_pos)
                src_end_x = src_start_x + (end_x - start_x)
                src_end_y = src_start_y + (end_y - start_y)
                
                canvas[start_y:end_y, start_x:end_x] = img_array[src_start_y:src_end_y, src_start_x:src_end_x]
    
    return Image.fromarray(canvas), "Circular Radial"

def create_pattern_variation_9_wave(image, output_width, output_height):
    """Variation 9: Wave/Sinusoidal Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Wave parameters
    wave_amplitude = h // 2
    wave_frequency = 2 * np.pi / (w * 3)
    
    # Create wave pattern
    for col in range(0, output_width, w // 2):
        for row in range(0, output_height, h):
            # Calculate wave offset
            wave_offset = int(wave_amplitude * np.sin(wave_frequency * col))
            
            x_pos = col
            y_pos = row + wave_offset
            
            # Place pattern
            if (x_pos + w > 0 and x_pos < output_width and 
                y_pos + h > 0 and y_pos < output_height):
                
                start_x = max(0, x_pos)
                start_y = max(0, y_pos)
                end_x = min(output_width, x_pos + w)
                end_y = min(output_height, y_pos + h)
                
                if start_x < end_x and start_y < end_y:
                    src_w = end_x - start_x
                    src_h = end_y - start_y
                    canvas[start_y:end_y, start_x:end_x] = img_array[:src_h, :src_w]
    
    return Image.fromarray(canvas), "Wave Pattern"

def create_pattern_variation_10_scattered(image, output_width, output_height):
    """Variation 10: Random Scattered Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Create base tiling first
    base_tiles_x = (output_width + w*2 - 1) // (w*2)
    base_tiles_y = (output_height + h*2 - 1) // (h*2)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Place base patterns with some randomness
    for row in range(base_tiles_y):
        for col in range(base_tiles_x):
            # Add random offset
            random_offset_x = np.random.randint(-w//4, w//4)
            random_offset_y = np.random.randint(-h//4, h//4)
            
            x_pos = col * w*2 + random_offset_x
            y_pos = row * h*2 + random_offset_y
            
            # Place pattern
            if (x_pos + w > 0 and x_pos < output_width and 
                y_pos + h > 0 and y_pos < output_height):
                
                start_x = max(0, x_pos)
                start_y = max(0, y_pos)
                end_x = min(output_width, x_pos + w)
                end_y = min(output_height, y_pos + h)
                
                if start_x < end_x and start_y < end_y:
                    src_start_x = max(0, -x_pos)
                    src_start_y = max(0, -y_pos)
                    src_end_x = src_start_x + (end_x - start_x)
                    src_end_y = src_start_y + (end_y - start_y)
                    
                    canvas[start_y:end_y, start_x:end_x] = img_array[src_start_y:src_end_y, src_start_x:src_end_x]
    
    return Image.fromarray(canvas), "Random Scattered"

def create_pattern_variation_11_chevron(image, output_width, output_height):
    """Variation 11: Chevron/Zigzag Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Create chevron pattern
    row_height = h
    chevron_width = w
    
    for row in range(0, output_height, row_height):
        for col in range(0, output_width + chevron_width, chevron_width):
            # Determine if this is a "peak" or "valley" row
            row_index = row // row_height
            
            if row_index % 2 == 0:
                # Normal placement
                x_pos = col
            else:
                # Offset for chevron effect
                x_pos = col - chevron_width // 2
            
            y_pos = row
            
            # Place pattern
            if (x_pos + w > 0 and x_pos < output_width and 
                y_pos + h > 0 and y_pos < output_height):
                
                start_x = max(0, x_pos)
                start_y = max(0, y_pos)
                end_x = min(output_width, x_pos + w)
                end_y = min(output_height, y_pos + h)
                
                if start_x < end_x and start_y < end_y:
                    src_start_x = max(0, -x_pos)
                    src_start_y = max(0, -y_pos)
                    src_end_x = src_start_x + (end_x - start_x)
                    src_end_y = src_start_y + (end_y - start_y)
                    
                    canvas[start_y:end_y, start_x:end_x] = img_array[src_start_y:src_end_y, src_start_x:src_end_x]
    
    return Image.fromarray(canvas), "Chevron Zigzag"

def create_pattern_variation_12_kaleidoscope(image, output_width, output_height):
    """Variation 12: Kaleidoscope/Mandala Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create a kaleidoscope effect
    img_pil = Image.fromarray(img_array)
    
    # Create multiple rotations
    rotations = [0, 60, 120, 180, 240, 300]
    patterns = []
    
    for angle in rotations:
        rotated = img_pil.rotate(angle, expand=True)
        patterns.append(np.array(rotated))
    
    # Create kaleidoscope arrangement
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Place patterns in a kaleidoscope arrangement
    center_x, center_y = output_width // 2, output_height // 2
    
    for i, pattern in enumerate(patterns):
        ph, pw = pattern.shape[:2]
        
        # Calculate position for each rotated pattern
        angle_rad = np.radians(rotations[i])
        
        # Place multiple copies at different radii
        for radius in range(0, min(output_width, output_height) // 2, max(w, h)):
            if radius == 0:
                x_pos = center_x - pw // 2
                y_pos = center_y - ph // 2
            else:
                x_pos = int(center_x + radius * np.cos(angle_rad) - pw // 2)
                y_pos = int(center_y + radius * np.sin(angle_rad) - ph // 2)
            
            # Place pattern
            if (x_pos + pw > 0 and x_pos < output_width and 
                y_pos + ph > 0 and y_pos < output_height):
                
                start_x = max(0, x_pos)
                start_y = max(0, y_pos)
                end_x = min(output_width, x_pos + pw)
                end_y = min(output_height, y_pos + ph)
                
                if start_x < end_x and start_y < end_y:
                    src_start_x = max(0, -x_pos)
                    src_start_y = max(0, -y_pos)
                    src_end_x = src_start_x + (end_x - start_x)
                    src_end_y = src_start_y + (end_y - start_y)
                    
                    canvas[start_y:end_y, start_x:end_x] = pattern[src_start_y:src_end_y, src_start_x:src_end_x]
    
    return Image.fromarray(canvas), "Kaleidoscope Mandala"

def generate_all_pattern_variations(original_image, output_width, output_height):
    """Generate all 12 pattern variations"""
    variations = []
    
    try:
        # Generate each variation
        var1, name1 = create_pattern_variation_1_original(original_image, output_width, output_height)
        variations.append((var1, name1))
        
        var2, name2 = create_pattern_variation_2_mirrored(original_image, output_width, output_height)
        variations.append((var2, name2))
        
        var3, name3 = create_pattern_variation_3_rotated(original_image, output_width, output_height)
        variations.append((var3, name3))
        
        var4, name4 = create_pattern_variation_4_scaled(original_image, output_width, output_height)
        variations.append((var4, name4))
        
        var5, name5 = create_pattern_variation_5_brick(original_image, output_width, output_height)
        variations.append((var5, name5))
        
        var6, name6 = create_pattern_variation_6_diagonal(original_image, output_width, output_height)
        variations.append((var6, name6))
        
        var7, name7 = create_pattern_variation_7_hexagonal(original_image, output_width, output_height)
        variations.append((var7, name7))
        
        var8, name8 = create_pattern_variation_8_circular(original_image, output_width, output_height)
        variations.append((var8, name8))
        
        var9, name9 = create_pattern_variation_9_wave(original_image, output_width, output_height)
        variations.append((var9, name9))
        
        var10, name10 = create_pattern_variation_10_scattered(original_image, output_width, output_height)
        variations.append((var10, name10))
        
        var11, name11 = create_pattern_variation_11_chevron(original_image, output_width, output_height)
        variations.append((var11, name11))
        
        var12, name12 = create_pattern_variation_12_kaleidoscope(original_image, output_width, output_height)
        variations.append((var12, name12))
        
    except Exception as e:
        st.error(f"Error generating variations: {e}")
        return []
    
    return variations

def get_download_link(img, filename):
    """Generate download link for image"""
    try:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", quality=95, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="background-color: #ff4b4b; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 5px; display: inline-block;">📥 Download</a>'
        return href
    except Exception as e:
        return f"<p>Error generating download link: {str(e)}</p>"

def main():
    st.set_page_config(
        page_title="Ultimate Pattern Carpet Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .pattern-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 10px 0;
    }
    .pattern-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Ultimate Pattern Carpet Generator</h1>
        <p>Create 12 different carpet patterns from your single input image!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Design Controls")
        
        uploaded_file = st.file_uploader(
            "Upload Pattern Sample",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload your pattern - get 12 different variations!"
        )
        
        st.subheader("📐 Output Dimensions")
        
        # Preset sizes
        size_preset = st.selectbox(
            "Size Preset",
            ["Custom", "Small Rug (900×600)", "Medium Rug (1200×800)", "Large Rug (1500×1000)", "Room Size (2000×1400)"]
        )
        
        if size_preset == "Small Rug (900×600)":
            default_w, default_h = 900, 600
        elif size_preset == "Medium Rug (1200×800)":
            default_w, default_h = 1200, 800
        elif size_preset == "Large Rug (1500×1000)":
            default_w, default_h = 1500, 1000
        elif size_preset == "Room Size (2000×1400)":
            default_w, default_h = 2000, 1400
        else:
            default_w, default_h = 1200, 800
        
        col1, col2 = st.columns(2)
        with col1:
            output_width = st.number_input("Width", min_value=100, max_value=4000, value=default_w, step=50)
        with col2:
            output_height = st.number_input("Height", min_value=100, max_value=4000, value=default_h, step=50)
        
        preview_mode = st.checkbox("Preview Mode (600x400 max)", value=True)
        
        st.subheader("🎯 Pattern Types")
        st.markdown("""
        **12 Variations Generated:**
        1. **Original** - Direct tiling
        2. **Mirrored** - Symmetrical design
        3. **4-Way Rotated** - Dynamic layout
        4. **Multi-Scale** - Improved size mixing
        5. **Brick/Offset** - Enhanced staggered layout
        6. **Diagonal Diamond** - Improved diagonal
        7. **Hexagonal Grid** - Honeycomb pattern
        8. **Circular Radial** - Radial arrangement
        9. **Wave Pattern** - Sinusoidal flow
        10. **Random Scattered** - Organic placement
        11. **Chevron Zigzag** - Angular design
        12. **Kaleidoscope** - Mandala-like pattern
        """)
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load image
            original_image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📷 Input Pattern")
                st.image(original_image, caption=f"Size: {original_image.size[0]}×{original_image.size[1]}px")
                
                st.info("✨ This will generate 12 different pattern variations!")
            
            with col2:
                st.subheader("🎨 Generate Pattern Variations")
                
                # Adjust for preview
                if preview_mode:
                    gen_width = min(output_width, 600)
                    gen_height = min(output_height, 400)
                    st.warning(f"Preview Mode: {gen_width}×{gen_height}px")
                else:
                    gen_width, gen_height = output_width, output_height
                
                if st.button("🚀 Generate 12 Pattern Variations", type="primary", use_container_width=True):
                    
                    with st.spinner("Creating 12 different pattern variations..."):
                        # Generate all variations
                        variations = generate_all_pattern_variations(
                            original_image, gen_width, gen_height
                        )
                        
                        if variations:
                            st.success("✅ Generated 12 unique pattern variations!")
                            
                            # Display all variations
                            st.subheader("✨ All 12 Pattern Variations")
                            
                            # Display in 3 columns, 4 rows
                            for i in range(0, len(variations), 3):
                                col_a, col_b, col_c = st.columns(3)
                                
                                # First pattern
                                if i < len(variations):
                                    pattern_img, pattern_name = variations[i]
                                    with col_a:
                                        st.markdown(f'<div class="pattern-card">', unsafe_allow_html=True)
                                        st.write(f"**{i+1}. {pattern_name}**")
                                        st.image(pattern_img, use_column_width=True)
                                        st.markdown(get_download_link(pattern_img, f"pattern_{i+1:02d}_{pattern_name.lower().replace(' ', '_').replace('/', '_')}.png"), unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Second pattern
                                if i+1 < len(variations):
                                    pattern_img, pattern_name = variations[i+1]
                                    with col_b:
                                        st.markdown(f'<div class="pattern-card">', unsafe_allow_html=True)
                                        st.write(f"**{i+2}. {pattern_name}**")
                                        st.image(pattern_img, use_column_width=True)
                                        st.markdown(get_download_link(pattern_img, f"pattern_{i+2:02d}_{pattern_name.lower().replace(' ', '_').replace('/', '_')}.png"), unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Third pattern
                                if i+2 < len(variations):
                                    pattern_img, pattern_name = variations[i+2]
                                    with col_c:
                                        st.markdown(f'<div class="pattern-card">', unsafe_allow_html=True)
                                        st.write(f"**{i+3}. {pattern_name}**")
                                        st.image(pattern_img, use_column_width=True)
                                        st.markdown(get_download_link(pattern_img, f"pattern_{i+3:02d}_{pattern_name.lower().replace(' ', '_').replace('/', '_')}.png"), unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Enhanced download section
                            st.subheader("📥 Download All Patterns")
                            st.markdown("**Quick Download Links:**")
                            
                            # Create download grid
                            download_cols = st.columns(4)
                            for i, (pattern_img, pattern_name) in enumerate(variations):
                                with download_cols[i % 4]:
                                    st.markdown(get_download_link(pattern_img, f"variation_{i+1:02d}_{pattern_name.lower().replace(' ', '_').replace('/', '_')}.png"), unsafe_allow_html=True)
                            
                            # Pattern analysis
                            st.subheader("🔍 Pattern Analysis")
                            analysis_col1, analysis_col2 = st.columns(2)
                            
                            with analysis_col1:
                                st.markdown("**Best for Different Spaces:**")
                                st.markdown("""
                                - **Living Room**: Original, Mirrored, Multi-Scale
                                - **Bedroom**: Wave, Scattered, Circular
                                - **Modern Spaces**: Hexagonal, Chevron, Kaleidoscope
                                - **Traditional**: Brick/Offset, 4-Way Rotated
                                """)
                            
                            with analysis_col2:
                                st.markdown("**Pattern Characteristics:**")
                                st.markdown("""
                                - **High Symmetry**: Mirrored, Kaleidoscope, Hexagonal
                                - **Dynamic Flow**: Wave, Circular, Chevron
                                - **Organic Feel**: Scattered, Multi-Scale
                                - **Geometric**: Brick, Diagonal, 4-Way Rotated
                                """)
                            
                            if preview_mode:
                                st.info("👆 Preview mode active. Uncheck for full resolution.")
                        else:
                            st.error("Failed to generate variations. Please try again.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("👆 Upload a pattern to generate 12 different variations!")
        
        st.subheader("🎨 Pattern Variations Preview")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📋 Pattern List", "🎯 Use Cases", "💡 Tips"])
        
        with tab1:
            st.markdown("""
            **Complete Pattern Collection (12 Variations):**
            
            **Basic Patterns:**
            1. **Original Pattern** - Clean direct tiling
            2. **Mirrored Symmetry** - Four-way mirror reflection
            3. **4-Way Rotated** - 90° rotational arrangement
            
            **Enhanced Patterns:**
            4. **Multi-Scale Mix** - Improved size combinations
            5. **Brick/Offset Layout** - Enhanced staggered pattern
            6. **Diagonal Diamond** - Improved diagonal arrangement
            
            **Advanced Patterns:**
            7. **Hexagonal Grid** - Honeycomb-style layout
            8. **Circular Radial** - Radial/concentric arrangement
            9. **Wave Pattern** - Sinusoidal flow design
            10. **Random Scattered** - Organic, natural placement
            11. **Chevron Zigzag** - Angular, dynamic design
            12. **Kaleidoscope Mandala** - Complex symmetrical pattern
            """)
        
        with tab2:
            st.markdown("""
            **Perfect Applications:**
            
            **Living Spaces:**
            - **Living Room**: Original, Mirrored, Multi-Scale, Circular
            - **Bedroom**: Wave, Scattered, Kaleidoscope (calming effects)
            - **Dining Room**: Hexagonal, Brick/Offset (structured feel)
            - **Study/Office**: Chevron, 4-Way Rotated (energizing)
            
            **Style Matching:**
            - **Modern/Contemporary**: Hexagonal, Circular, Wave
            - **Traditional**: Brick/Offset, Original, Mirrored
            - **Bohemian**: Scattered, Kaleidoscope, Wave
            - **Minimalist**: Original, Diagonal, 4-Way Rotated
            - **Eclectic**: Multi-Scale, Chevron, Circular
            """)
        
        with tab3:
            st.markdown("""
            **Pro Tips for Best Results:**
            
            **Input Pattern Tips:**
            - Use high-contrast patterns for better visibility
            - Square patterns work best for most variations
            - Avoid very small details that might get lost
            - Consider how your pattern will look when repeated
            
            **Size Recommendations:**
            - **Preview Mode**: Great for testing different patterns
            - **Small Rugs**: Perfect for accent pieces
            - **Large Rugs**: Better for main room carpets
            - **Room Size**: For wall-to-wall coverage
            
            **Pattern Selection:**
            - **High Traffic Areas**: Use simpler patterns (Original, Mirrored)
            - **Accent Areas**: Try complex patterns (Kaleidoscope, Circular)
            - **Large Spaces**: Multi-Scale and Scattered work well
            - **Small Spaces**: Avoid overly busy patterns
            """)

if __name__ == "__main__":
    main()
