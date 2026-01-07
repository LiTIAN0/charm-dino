import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import os
from dino import segment_color_patch, calculate_metrics

st.set_page_config(page_title="CHARM Ink Analyst", page_icon="✒️", layout="wide")

st.title("✒️ CHARM AI: Medieval Manuscript Analyst")
st.markdown("Use computer vision to identify pigments and inks based on their spectral properties.")

# ==========================================
# 1. Sidebar: Configuration
# ==========================================
st.sidebar.header("1. Settings")
target_color = st.sidebar.selectbox("Target Color:", ["Black", "Blue", "Red"])

st.sidebar.header("2. Image Input")
input_method = st.sidebar.radio("Source:", ["📂 Use Demo Gallery", "📤 Upload Images"])

vis_image = None # RGB format
aux_type = "IR"  # Default

# Logic branch: Different colors require different auxiliary images
if target_color == "Red":
    aux_label = "UV Image"
    aux_type = "UV"
else:
    aux_label = "IR Image"
    aux_type = "IR"

# --- Image loading logic ---
if input_method == "📂 Use Demo Gallery":
    # Built-in image library  
    if target_color == "Black":
        samples = ["I32_black (Iron Gall)", "II115_black (Carbon-like)"]
    elif target_color == "Blue":
        samples = ["I32_blue (Plant-based)", "III125_blue (Mineral)"]
    else:
        samples = ["I32_red (Mineral/Cinnabar)", "VII78_red (Mineral/Cinnabar)"]
        
    choice = st.sidebar.selectbox("Select Sample:", samples)
    base_name = choice.split(" ")[0]
    
    try:
        # Read and convert to RGB / Grayscale  
        vis_path = os.path.join("demo_images", f"{base_name}_VIS.bmp")
        vis_image = cv2.cvtColor(cv2.imread(vis_path), cv2.COLOR_BGR2RGB)
        
        aux_path = os.path.join("demo_images", f"{base_name}_{aux_type}.bmp")
        # Read in color for display, convert to grayscale for computation
        # 1. Read in color (BGR → RGB) for display
        aux_image_display = cv2.cvtColor(cv2.imread(aux_path), cv2.COLOR_BGR2RGB)
        
        # 2. Convert to grayscale for score calculation
        aux_image_calc = cv2.cvtColor(aux_image_display, cv2.COLOR_RGB2GRAY)
    except:
        st.sidebar.error(f"Missing demo files for {base_name}")

else:
    u_vis = st.sidebar.file_uploader("Upload VIS (Required)", type=['jpg','png', 'bmp'])
    u_aux = st.sidebar.file_uploader(f"Upload {aux_type} (Required)", type=['jpg','png', 'bmp'])
    
    if u_vis and u_aux:
        # Decode VIS as RGB
        file_bytes = np.asarray(bytearray(u_vis.read()), dtype=np.uint8)
        vis_image = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
        
        # Decode Aux as grayscale
        file_bytes_aux = np.asarray(bytearray(u_aux.read()), dtype=np.uint8)
        # Decode as color for display
        aux_image_display = cv2.cvtColor(cv2.imdecode(file_bytes_aux, 1), cv2.COLOR_BGR2RGB)
        # Convert to grayscale for computation
        aux_image_calc = cv2.cvtColor(aux_image_display, cv2.COLOR_RGB2GRAY)

# ==========================================
# 2. Main interface  
# ==========================================

if vis_image is not None and aux_image_display is not None:
    
    # --- Step 1: Human Judgment (Dynamic indicators) ---
    st.subheader("Step 1: Human Inspection")
    
    # Display different Field Guides based on color
    guide_text = ""
    if target_color == "Black":
        guide_text = """
        **Look at the IR Image:**
        *   Does the ink **disappear**? -> Likely **Iron Gall Ink**
        *   Does it **stay dark**? -> Likely **Carbon Ink**
        """
        options = ["Iron Gall Ink", "Carbon Ink", "Unsure"]
    elif target_color == "Blue":
        guide_text = """
        **Look at the IR Image:**
        *   Does it **stay dark**? -> Likely **Mineral (Azurite)**
        *   Does it **become transparent**? -> Likely **Plant-based (Indigo)**
        """
        options = ["Mineral Blue", "Plant-based Blue", "Unsure"]
    elif target_color == "Red":
        guide_text = """
        **Look at the UV Image:**
        *   Does it **glow bright orange**? -> Likely **Madder (Organic)**
        *   Is it **dark/purple**? -> Likely **Vermilion/Minium (Mineral)**
        """
        options = ["Madder (Fluorescent)", "Mineral Red (Non-fluo)", "Unsure"]

    with st.expander(f"📖 Field Guide for {target_color}", expanded=True):
        st.markdown(guide_text)

    c1, c2 = st.columns(2)
    c1.image(vis_image, caption="Visible Light", use_column_width=True)
    c2.image(aux_image_display, caption=f"{aux_type} Light", use_column_width=True)
    
    user_guess = st.radio("Your Hypothesis:", options, horizontal=True)

    # --- Step 2: AI Analysis ---
    if st.button("🚀 Run AI Analysis"):
        st.divider()
        
        # A. Run segmentation
        # Note: RGB image arrays here
        vis_rgb_clean, mask = segment_color_patch(vis_image, target_color)
        
        # B. Show intermediate steps
        st.write("### AI Vision Process")
        c1, c2, c3 = st.columns(3)
        c1.image(vis_image, caption="1. Original", use_column_width=True)
        c2.image(mask, caption="2. AI Mask (ROI)", use_column_width=True)
        
        segmented = cv2.bitwise_and(vis_image, vis_image, mask=mask)
        segmented[mask==0] = 255 # White background
        c3.image(segmented, caption="3. Extracted Pigment", use_column_width=True)
        
        # C. Calculate score
        ir_in = aux_image_calc if aux_type == "IR" else None
        uv_in = aux_image_calc if aux_type == "UV" else None
        
        metrics = calculate_metrics(vis_image, ir_in, uv_in, mask)
        
        # D. Dashboard and conclusion
        final_pred = "Unknown"
        final_score = 0
        
        if target_color == "Black":
            final_score = metrics['ir_score']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = final_score,
                title = {'text': "IR Transparency Score"},
                gauge = {'axis': {'range': [0, 1.2]},
                         'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                   {'range': [0.9, 1.2], 'color': "lightgreen"}]}))
            
            if final_score > 0.95: final_pred = "Iron Gall Ink"
            elif final_score < 0.85: final_pred = "Carbon Ink"
            else: final_pred = "Mixed / Thick Ink"
            
        elif target_color == "Blue":
            final_score = metrics['ir_score']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = final_score,
                title = {'text': "IR Transparency Score"},
                gauge = {'axis': {'range': [0, 1.2]},
                         'steps': [{'range': [0, 0.5], 'color': "lightblue"}, # Mineral
                                   {'range': [0.8, 1.2], 'color': "lightgreen"}]})) # Plant
            
            if final_score > 0.8: final_pred = "Plant-based Blue"
            else: final_pred = "Mineral Blue"

        elif target_color == "Red":
            final_score = metrics['uv_score']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = final_score,
                title = {'text': "UV Fluorescence Score"},
                gauge = {'axis': {'range': [-1, 2]},
                         'steps': [{'range': [-1, 0.1], 'color': "lightgray"}, # Mineral
                                   {'range': [0.2, 2], 'color': "orange"}]})) # Fluorescent
            
            if final_score > 0.2: final_pred = "Madder (Fluorescent)"
            else: final_pred = "Mineral Red (Non-fluo)"

        st.plotly_chart(fig, use_column_width=True)
        
        st.success(f"🤖 **AI Conclusion:** {final_pred}")
        
        if user_guess in final_pred:
            st.balloons()
            st.write("🎉 You and the AI agree!")
        else:
            st.write("🤔 Differing opinions. Check the score visualization above.")

else:
    st.info("👈 Please select a demo sample or upload images to begin.")