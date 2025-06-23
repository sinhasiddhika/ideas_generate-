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
    """Variation 4: Multi-Scale Pattern - Different sizes combined"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create different scales
    img_pil = Image.fromarray(img_array)
    large = np.array(img_pil.resize((w, h)))  # Original size
    medium = np.array(img_pil.resize((w//2, h//2)))  # Half size
    small = np.array(img_pil.resize((w//3, h//3)))  # Third size
    
    # Create a composite pattern
    # Place large in center, medium around it, small in corners
    composite_h = h * 2
    composite_w = w * 2
    composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
    
    # Place large pattern in center
    start_h = (composite_h - h) // 2
    start_w = (composite_w - w) // 2
    composite[start_h:start_h+h, start_w:start_w+w] = large
    
    # Fill remaining space with medium pattern
    med_h, med_w = medium.shape[:2]
    for i in range(0, composite_h, med_h):
        for j in range(0, composite_w, med_w):
            end_i = min(i + med_h, composite_h)
            end_j = min(j + med_w, composite_w)
            if i < start_h or i >= start_h + h or j < start_w or j >= start_w + w:
                composite[i:end_i, j:end_j] = medium[:end_i-i, :end_j-j]
    
    # Tile this composite pattern
    ch, cw = composite.shape[:2]
    tiles_x = (output_width + cw - 1) // cw
    tiles_y = (output_height + ch - 1) // ch
    
    tiled_array = np.tile(composite, (tiles_y, tiles_x, 1))
    final_array = tiled_array[:output_height, :output_width]
    return Image.fromarray(final_array), "Multi-Scale Mix"

def create_pattern_variation_5_brick(image, output_width, output_height):
    """Variation 5: Brick/Offset Pattern - Staggered layout"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create brick pattern by offsetting every other row
    offset = w // 2
    
    # Calculate how many tiles we need
    tiles_x = (output_width + w - 1) // w + 1  # Extra for offset
    tiles_y = (output_height + h - 1) // h
    
    # Create larger canvas for brick pattern
    canvas_h = tiles_y * h
    canvas_w = tiles_x * w + offset
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Fill with brick pattern
    for row in range(tiles_y):
        for col in range(tiles_x):
            y_pos = row * h
            x_pos = col * w
            
            # Offset every other row
            if row % 2 == 1:
                x_pos += offset
            
            # Make sure we don't go out of bounds
            if y_pos + h <= canvas_h and x_pos + w <= canvas_w:
                canvas[y_pos:y_pos+h, x_pos:x_pos+w] = img_array
    
    # Crop to final size
    final_array = canvas[:output_height, :output_width]
    return Image.fromarray(final_array), "Brick/Offset Layout"

def create_pattern_variation_6_diagonal(image, output_width, output_height):
    """Variation 6: Diagonal Pattern - 45-degree arrangement"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create diagonal pattern by rotating and arranging
    img_pil = Image.fromarray(img_array)
    
    # Rotate 45 degrees
    diagonal_img = img_pil.rotate(45, expand=True, fillcolor=(128, 128, 128))
    diag_array = np.array(diagonal_img)
    dh, dw = diag_array.shape[:2]
    
    # Create alternating diagonal pattern
    # Tile the diagonal version
    tiles_x = (output_width + dw - 1) // dw
    tiles_y = (output_height + dh - 1) // dh
    
    if len(diag_array.shape) == 3:
        tiled_array = np.tile(diag_array, (tiles_y, tiles_x, 1))
    else:
        tiled_array = np.tile(diag_array, (tiles_y, tiles_x))
    
    final_array = tiled_array[:output_height, :output_width]
    return Image.fromarray(final_array), "Diagonal Layout"

def generate_all_pattern_variations(original_image, output_width, output_height):
    """Generate all 6 pattern variations"""
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
        page_title="Multi-Pattern Carpet Generator",
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
        <h1>🎨 Multi-Pattern Carpet Generator</h1>
        <p>Create 6 different carpet patterns from your single input image!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Design Controls")
        
        uploaded_file = st.file_uploader(
            "Upload Pattern Sample",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload your pattern - get 6 different variations!"
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
        **6 Variations Generated:**
        1. **Original** - Direct tiling
        2. **Mirrored** - Symmetrical design
        3. **4-Way Rotated** - Dynamic layout
        4. **Multi-Scale** - Different sizes mixed
        5. **Brick/Offset** - Staggered arrangement
        6. **Diagonal** - 45-degree layout
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
                
                st.info("✨ This will generate 6 different pattern variations!")
            
            with col2:
                st.subheader("🎨 Generate Pattern Variations")
                
                # Adjust for preview
                if preview_mode:
                    gen_width = min(output_width, 600)
                    gen_height = min(output_height, 400)
                    st.warning(f"Preview Mode: {gen_width}×{gen_height}px")
                else:
                    gen_width, gen_height = output_width, output_height
                
                if st.button("🚀 Generate 6 Pattern Variations", type="primary", use_container_width=True):
                    
                    with st.spinner("Creating 6 different pattern variations..."):
                        # Generate all variations
                        variations = generate_all_pattern_variations(
                            original_image, gen_width, gen_height
                        )
                        
                        if variations:
                            st.success("✅ Generated 6 unique pattern variations!")
                            
                            # Display all variations
                            st.subheader("✨ All Pattern Variations")
                            
                            # Display in 2 columns, 3 rows
                            for i in range(0, len(variations), 2):
                                col_a, col_b = st.columns(2)
                                
                                # First pattern
                                if i < len(variations):
                                    pattern_img, pattern_name = variations[i]
                                    with col_a:
                                        st.markdown(f'<div class="pattern-card">', unsafe_allow_html=True)
                                        st.write(f"**{i+1}. {pattern_name}**")
                                        st.image(pattern_img, use_column_width=True)
                                        st.markdown(get_download_link(pattern_img, f"pattern_{i+1}_{pattern_name.lower().replace(' ', '_')}.png"), unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Second pattern
                                if i+1 < len(variations):
                                    pattern_img, pattern_name = variations[i+1]
                                    with col_b:
                                        st.markdown(f'<div class="pattern-card">', unsafe_allow_html=True)
                                        st.write(f"**{i+2}. {pattern_name}**")
                                        st.image(pattern_img, use_column_width=True)
                                        st.markdown(get_download_link(pattern_img, f"pattern_{i+2}_{pattern_name.lower().replace(' ', '_')}.png"), unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Bulk download section
                            st.subheader("📥 Download All Patterns")
                            download_cols = st.columns(3)
                            for i, (pattern_img, pattern_name) in enumerate(variations):
                                with download_cols[i % 3]:
                                    st.markdown(get_download_link(pattern_img, f"variation_{i+1}_{pattern_name.lower().replace(' ', '_')}.png"), unsafe_allow_html=True)
                            
                            if preview_mode:
                                st.info("👆 Preview mode active. Uncheck for full resolution.")
                        else:
                            st.error("Failed to generate variations. Please try again.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("👆 Upload a pattern to generate 6 different variations!")
        
        st.subheader("🎨 Pattern Variations Preview")
        st.markdown("""
        **What you'll get from your input:**
        
        1. **Original Pattern** - Clean direct tiling of your input
        2. **Mirrored Symmetry** - Creates beautiful symmetrical designs
        3. **4-Way Rotated** - Dynamic layout with 90° rotations
        4. **Multi-Scale Mix** - Different sizes combined artistically
        5. **Brick/Offset Layout** - Staggered arrangement like brickwork
        6. **Diagonal Layout** - 45-degree rotated arrangement
        
        **Perfect for:**
        - Choosing the best layout for your space
        - Seeing different aesthetic options
        - Creating unique carpet designs
        - Exploring creative possibilities
        """)

if __name__ == "__main__":
    main()