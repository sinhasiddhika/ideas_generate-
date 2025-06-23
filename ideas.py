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
        'tile_size': (min(h, w), min(h, w)),
        'half_size': (h//2, w//2),
        'quarter_size': (h//4, w//4) if h > 100 and w > 100 else (h//2, w//2)
    }

def create_pattern_variation_1_original(image, output_width, output_height):
    """Variation 1: Original Pattern - Direct Tiling"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
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
    
    mirrored_h = np.hstack([img_array, np.fliplr(img_array)])
    mirrored_full = np.vstack([mirrored_h, np.flipud(mirrored_h)])
    
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
    
    img_pil = Image.fromarray(img_array)
    rot90 = np.array(img_pil.rotate(90, expand=True))
    rot180 = np.array(img_pil.rotate(180, expand=True))
    rot270 = np.array(img_pil.rotate(270, expand=True))
    
    h, w = img_array.shape[:2]
    rot90_resized = np.array(Image.fromarray(rot90).resize((w, h)))
    rot180_resized = np.array(Image.fromarray(rot180).resize((w, h)))
    rot270_resized = np.array(Image.fromarray(rot270).resize((w, h)))
    
    top_row = np.hstack([img_array, rot90_resized])
    bottom_row = np.hstack([rot270_resized, rot180_resized])
    rotated_pattern = np.vstack([top_row, bottom_row])
    
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
    """Variation 4: Multi-Scale Pattern - FIXED VERSION"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create different scales
    img_pil = Image.fromarray(img_array)
    large = np.array(img_pil.resize((w, h)))
    medium = np.array(img_pil.resize((w//2, h//2)))
    small = np.array(img_pil.resize((w//4, h//4)))
    
    # Create base pattern with medium tiles
    med_h, med_w = medium.shape[:2]
    base_tiles_x = (output_width + med_w - 1) // med_w
    base_tiles_y = (output_height + med_h - 1) // med_h
    
    # Create base canvas with medium pattern
    if len(medium.shape) == 3:
        base_pattern = np.tile(medium, (base_tiles_y, base_tiles_x, 1))
    else:
        base_pattern = np.tile(medium, (base_tiles_y, base_tiles_x))
    
    base_pattern = base_pattern[:output_height, :output_width]
    
    # Add large patterns strategically
    spacing_h = h + h//2
    spacing_w = w + w//2
    
    for y_pos in range(0, output_height - h, spacing_h):
        for x_pos in range(0, output_width - w, spacing_w):
            end_y = min(y_pos + h, output_height)
            end_x = min(x_pos + w, output_width)
            base_pattern[y_pos:end_y, x_pos:end_x] = large[:end_y-y_pos, :end_x-x_pos]
    
    # Add small pattern details
    small_h, small_w = small.shape[:2]
    small_spacing = max(small_h * 3, small_w * 3)
    
    for y_pos in range(small_spacing//2, output_height - small_h, small_spacing):
        for x_pos in range(small_spacing//2, output_width - small_w, small_spacing):
            end_y = min(y_pos + small_h, output_height)
            end_x = min(x_pos + small_w, output_width)
            base_pattern[y_pos:end_y, x_pos:end_x] = small[:end_y-y_pos, :end_x-x_pos]
    
    return Image.fromarray(base_pattern), "Multi-Scale Mix"

def create_pattern_variation_5_brick(image, output_width, output_height):
    """Variation 5: Brick/Offset Pattern - Staggered layout"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    offset = w // 2
    tiles_x = (output_width + w - 1) // w + 1
    tiles_y = (output_height + h - 1) // h
    
    canvas_h = tiles_y * h
    canvas_w = tiles_x * w + offset
    
    if len(img_array.shape) == 3:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    for row in range(tiles_y):
        for col in range(tiles_x):
            y_pos = row * h
            x_pos = col * w
            
            if row % 2 == 1:
                x_pos += offset
            
            if y_pos + h <= canvas_h and x_pos + w <= canvas_w:
                canvas[y_pos:y_pos+h, x_pos:x_pos+w] = img_array
    
    final_array = canvas[:output_height, :output_width]
    return Image.fromarray(final_array), "Brick/Offset Layout"

def create_pattern_variation_6_diagonal(image, output_width, output_height):
    """Variation 6: Diagonal Pattern - 45-degree arrangement"""
    img_array = np.array(image)
    img_pil = Image.fromarray(img_array)
    
    diagonal_img = img_pil.rotate(45, expand=True, fillcolor=(128, 128, 128))
    diag_array = np.array(diagonal_img)
    dh, dw = diag_array.shape[:2]
    
    tiles_x = (output_width + dw - 1) // dw
    tiles_y = (output_height + dh - 1) // dh
    
    if len(diag_array.shape) == 3:
        tiled_array = np.tile(diag_array, (tiles_y, tiles_x, 1))
    else:
        tiled_array = np.tile(diag_array, (tiles_y, tiles_x))
    
    final_array = tiled_array[:output_height, :output_width]
    return Image.fromarray(final_array), "Diagonal Layout"

def create_pattern_variation_7_hexagonal(image, output_width, output_height):
    """Variation 7: Hexagonal/Honeycomb Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create hexagonal offset pattern
    hex_offset_x = w * 3 // 4
    hex_offset_y = h * 2 // 3
    
    if len(img_array.shape) == 3:
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((output_height, output_width), dtype=np.uint8)
    
    row = 0
    y_pos = 0
    while y_pos < output_height:
        col = 0
        x_pos = 0 if row % 2 == 0 else hex_offset_x // 2
        
        while x_pos < output_width:
            end_y = min(y_pos + h, output_height)
            end_x = min(x_pos + w, output_width)
            
            canvas[y_pos:end_y, x_pos:end_x] = img_array[:end_y-y_pos, :end_x-x_pos]
            
            x_pos += hex_offset_x
            col += 1
        
        y_pos += hex_offset_y
        row += 1
    
    return Image.fromarray(canvas), "Hexagonal Layout"

def create_pattern_variation_8_radial(image, output_width, output_height):
    """Variation 8: Radial/Circular Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create multiple rotated versions
    img_pil = Image.fromarray(img_array)
    angles = [0, 60, 120, 180, 240, 300]
    rotated_versions = []
    
    for angle in angles:
        rotated = img_pil.rotate(angle, expand=False)
        rotated_versions.append(np.array(rotated))
    
    # Create radial pattern
    center_x, center_y = output_width // 2, output_height // 2
    if len(img_array.shape) == 3:
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # Place patterns in radial arrangement
    radius_step = max(w, h)
    for radius in range(0, max(output_width, output_height), radius_step):
        angle_count = max(6, (2 * np.pi * radius) // max(w, h))
        for i in range(int(angle_count)):
            angle = 2 * np.pi * i / angle_count
            x = int(center_x + radius * np.cos(angle) - w//2)
            y = int(center_y + radius * np.sin(angle) - h//2)
            
            if 0 <= x < output_width - w and 0 <= y < output_height - h:
                pattern_idx = i % len(rotated_versions)
                canvas[y:y+h, x:x+w] = rotated_versions[pattern_idx]
    
    return Image.fromarray(canvas), "Radial Layout"

def create_pattern_variation_9_wave(image, output_width, output_height):
    """Variation 9: Wave/Sine Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    if len(img_array.shape) == 3:
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # Create wave pattern
    wave_amplitude = h // 2
    wave_frequency = 2 * np.pi / (w * 3)
    
    for row in range(0, output_height, h):
        for col in range(0, output_width, w):
            # Calculate wave offset
            wave_offset = int(wave_amplitude * np.sin(wave_frequency * col))
            
            y_pos = row + wave_offset
            x_pos = col
            
            # Ensure within bounds
            if 0 <= y_pos < output_height - h and 0 <= x_pos < output_width - w:
                canvas[y_pos:y_pos+h, x_pos:x_pos+w] = img_array
    
    return Image.fromarray(canvas), "Wave Pattern"

def create_pattern_variation_10_spiral(image, output_width, output_height):
    """Variation 10: Spiral Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    if len(img_array.shape) == 3:
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # Create spiral
    center_x, center_y = output_width // 2, output_height // 2
    max_radius = min(output_width, output_height) // 2
    
    for angle in np.arange(0, 8 * np.pi, 0.3):
        radius = (angle / (8 * np.pi)) * max_radius
        x = int(center_x + radius * np.cos(angle) - w//2)
        y = int(center_y + radius * np.sin(angle) - h//2)
        
        if 0 <= x < output_width - w and 0 <= y < output_height - h:
            # Rotate pattern based on angle
            rotation_angle = angle * 180 / np.pi
            rotated_img = Image.fromarray(img_array).rotate(rotation_angle, expand=False)
            rotated_array = np.array(rotated_img)
            canvas[y:y+h, x:x+w] = rotated_array
    
    return Image.fromarray(canvas), "Spiral Layout"

def create_pattern_variation_11_checkerboard(image, output_width, output_height):
    """Variation 11: Checkerboard with Alternating Patterns"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create inverted/modified version
    img_pil = Image.fromarray(img_array)
    inverted = ImageOps.invert(img_pil) if img_pil.mode == 'RGB' else img_pil.rotate(180)
    inverted_array = np.array(inverted)
    
    if len(img_array.shape) == 3:
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # Create checkerboard pattern
    for row in range(0, output_height, h):
        for col in range(0, output_width, w):
            # Determine which pattern to use
            tile_row = row // h
            tile_col = col // w
            use_original = (tile_row + tile_col) % 2 == 0
            
            pattern = img_array if use_original else inverted_array
            
            end_row = min(row + h, output_height)
            end_col = min(col + w, output_width)
            
            canvas[row:end_row, col:end_col] = pattern[:end_row-row, :end_col-col]
    
    return Image.fromarray(canvas), "Checkerboard Pattern"

def create_pattern_variation_12_kaleidoscope(image, output_width, output_height):
    """Variation 12: Kaleidoscope Pattern"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create kaleidoscope effect
    img_pil = Image.fromarray(img_array)
    
    # Create 8 rotated versions
    rotations = []
    for angle in range(0, 360, 45):
        rotated = img_pil.rotate(angle, expand=False)
        rotations.append(np.array(rotated))
    
    # Create pattern with different rotations
    if len(img_array.shape) == 3:
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((output_height, output_width), dtype=np.uint8)
    
    rotation_idx = 0
    for row in range(0, output_height, h):
        for col in range(0, output_width, w):
            end_row = min(row + h, output_height)
            end_col = min(col + w, output_width)
            
            pattern = rotations[rotation_idx % len(rotations)]
            canvas[row:end_row, col:end_col] = pattern[:end_row-row, :end_col-col]
            
            rotation_idx += 1
    
    return Image.fromarray(canvas), "Kaleidoscope Pattern"

def generate_all_pattern_variations(original_image, output_width, output_height):
    """Generate all 12 pattern variations"""
    variations = []
    
    try:
        # Generate each variation
        variations.append(create_pattern_variation_1_original(original_image, output_width, output_height))
        variations.append(create_pattern_variation_2_mirrored(original_image, output_width, output_height))
        variations.append(create_pattern_variation_3_rotated(original_image, output_width, output_height))
        variations.append(create_pattern_variation_4_scaled(original_image, output_width, output_height))
        variations.append(create_pattern_variation_5_brick(original_image, output_width, output_height))
        variations.append(create_pattern_variation_6_diagonal(original_image, output_width, output_height))
        variations.append(create_pattern_variation_7_hexagonal(original_image, output_width, output_height))
        variations.append(create_pattern_variation_8_radial(original_image, output_width, output_height))
        variations.append(create_pattern_variation_9_wave(original_image, output_width, output_height))
        variations.append(create_pattern_variation_10_spiral(original_image, output_width, output_height))
        variations.append(create_pattern_variation_11_checkerboard(original_image, output_width, output_height))
        variations.append(create_pattern_variation_12_kaleidoscope(original_image, output_width, output_height))
        
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
        page_title="Advanced Multi-Pattern Carpet Generator",
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Advanced Multi-Pattern Carpet Generator</h1>
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
        4. **Multi-Scale** - Different sizes mixed (FIXED!)
        5. **Brick/Offset** - Staggered arrangement
        6. **Diagonal** - 45-degree layout
        7. **Hexagonal** - Honeycomb pattern
        8. **Radial** - Circular arrangement
        9. **Wave** - Sine wave pattern
        10. **Spiral** - Spiral layout
        11. **Checkerboard** - Alternating patterns
        12. **Kaleidoscope** - Multi-rotation effect
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
                            st.subheader("✨ All Pattern Variations")
                            
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
                                        st.markdown(get_download_link(pattern_img, f"pattern_{i+1}_{pattern_name.lower().replace(' ', '_').replace('/', '_')}.png"), unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Second pattern
                                if i+1 < len(variations):
                                    pattern_img, pattern_name = variations[i+1]
                                    with col_b:
                                        st.markdown(f'<div class="pattern-card">', unsafe_allow_html=True)
                                        st.write(f"**{i+2}. {pattern_name}**")
                                        st.image(pattern_img, use_column_width=True)
                                        st.markdown(get_download_link(pattern_img, f"pattern_{i+2}_{pattern_name.lower().replace(' ', '_').replace('/', '_')}.png"), unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Third pattern
                                if i+2 < len(variations):
                                    pattern_img, pattern_name = variations[i+2]
                                    with col_c:
                                        st.markdown(f'<div class="pattern-card">', unsafe_allow_html=True)
                                        st.write(f"**{i+3}. {pattern_name}**")
                                        st.image(pattern_img, use_column_width=True)
                                        st.markdown(get_download_link(pattern_img, f"pattern_{i+3}_{pattern_name.lower().replace(' ', '_').replace('/', '_')}.png"), unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Bulk download section
                            st.subheader("📥 Download All Patterns")
                            download_cols = st.columns(4)
                            for i, (pattern_img, pattern_name) in enumerate(variations):
                                with download_cols[i % 4]:
                                    st.markdown(get_download_link(pattern_img, f"variation_{i+1}_{pattern_name.lower().replace(' ', '_').replace('/', '_')}.png"), unsafe_allow_html=True)
                            
                            if preview_mode:
                                st.info("👆 Preview mode active. Uncheck for full resolution.")
                        else:
                            st.error("Failed to generate variations. Please try again.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("👆 Upload a pattern to generate 12 different variations!")
        
        st.subheader("🎨 Pattern Variations Preview")
        st.markdown("""
        **What you'll get from your input:**
        
        1. **Original Pattern** - Clean direct tiling of your input
        2. **Mirrored Symmetry** - Creates beautiful symmetrical designs
        3. **4-Way Rotated** - Dynamic layout with 90° rotations
        4. **Multi-Scale Mix** - Different sizes combined artistically (FIXED - no more blank spaces!)
        5. **Brick/Offset Layout** - Staggered arrangement like brickwork
        6. **Diagonal Layout** - 45-degree rotated arrangement
        7. **Hexagonal Layout** - Honeycomb-style pattern arrangement
        8. **Radial Layout** - Circular/radial pattern distribution
        9. **Wave Pattern** - Sine wave-based flowing arrangement
        10. **Spiral Layout** - Spiral-based pattern placement
        11. **Checkerboard Pattern** - Alternating original/inverted tiles
        12. **Kaleidoscope Pattern** - Multi-rotation kaleidoscope effect
        
        **Perfect for:**
        - Choosing the best layout for your space
        - Seeing different aesthetic options
        - Creating unique carpet designs
        - Exploring creative possibilities
        - Professional carpet design presentations
        - Interior design projects
        """)

if __name__ == "__main__":
    main()
