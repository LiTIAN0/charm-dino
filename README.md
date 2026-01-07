# CHARM Project: Automated Pigment & Ink Identification Pipeline

This repository contains the computational workflow developed for the **CHARM Project** (Combining Humanities and natural science research to study medieval texts, scribes, and craftsmanship). It is based on Computer Vision technique to automate the identification of medieval writing materials based on their optical properties in multispectral images (Visible, UV, and Infrared).

## 📂 Repository Structure

*   **`dino.ipynb`**: The complete research pipeline in a Jupyter Notebook. Best for **batch processing** large datasets and performing detailed analysis (clustering, visualization).
*   **`app.py`** & **`dino.py`**: The source code for the interactive Web Application. Best for **demonstrations** or analyzing single images.
*   **`requirements.txt`**: List of Python dependencies.
*   **`demo_images/`**: Sample images for testing.

---

## 🚀 Quick Start

### Option 1: Interactive Web App (Recommended for Demo)
If you have a few images or want to visualize the extraction process, use the web interface.
*   **Live Link:** https://charm-dino.streamlit.app
*   **Usage:** Upload images or select from the gallery to get an instant material prediction.

### Option 2: Jupyter Notebook (Recommended for Research)
If you have a large dataset (hundreds of images) and need to run batch analysis or clustering, use `dino.ipynb`.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Prepare Data:**
    Ensure your images follow the **strict naming convention** below.
3.  **Run the Notebook:**
    Open `dino.ipynb` and execute the cells sequentially. You will need to update the `INPUT_FOLDER` path to point to your dataset.

---

## 📋 Data Preparation (Crucial)

The pipeline relies on file naming to identify the target color and light source. Please ensure all image files are named strictly as follows:

**Format:** `ID_Color_Light.jpg`

*   **ID:** The manuscript or fragment identifier (e.g., `I32`, `Coll.454`).
*   **Color:** The target color to analyze. Currently supports: **`Black`**, **`Red`**, **`Blue`**.
*   **Light:** The light source. Must be **`VIS`** (Visible), **`UV`** (Ultraviolet), or **`IR`** (Infrared).

**Examples:**
*   ✅ `I32_Red_VIS.jpg` (Correct)
*   ✅ `I53_Black_IR.bmp` (Correct)
*   ❌ `Image1.jpg` (Incorrect - will fail)

---

## 🧠 Analysis Pipeline (`dino.ipynb`)

The notebook guides you through the following four steps:

### 1. Targeted Color Segmentation
*   Extracts the specific ink/pigment from the parchment background using HSV color space thresholds.
*   *Note:* The notebook includes a testing block to visualize and verify the segmentation before processing the full dataset.

### 2. Quantitative Feature Extraction
*   Calculates physics-informed metrics to quantify how the material changes under different lights.
*   **IR Transparency Score:** Measures how much contrast is lost in IR (High = Iron Gall Ink).
*   **UV Fluorescence Score:** Measures brightness relative to the background (High = Organic/Madder).

### 3. Batch Processing & Clustering
*   Iterates through the entire image folder.
*   Performs unsupervised clustering (K-Means) to discover natural groupings in the data (e.g., separating "Visible" vs. "Invisible" black inks).

### 4. Automated Prediction
*   Predicts the material composition based on established spectral thresholds (derived from *Cosentino, 2014* and project data).
*   **Black:** Classification into Iron Gall Ink vs. Carbon-based Ink.
*   **Red:** Classification into Mineral Red (Cinnabar/Red Lead) vs. Organic Red (Madder).
*   **Blue:** Classification into Mineral Blue (Azurite) vs. Plant-based Blue (Indigo).

---

## 🔮 Future Improvements

*   **Expand Color Support:** Currently optimized for Black, Red, and Blue. Logic for other colors like Green can be added. 
*   **Improve Quantitative Analysis:** The score calculation methods can be adjusted based on requirements and observed performance.
*   **Advanced Models:** As the labeled dataset grows (verified by XRF), the rule-based classifier can be replaced with supervised Machine Learning models (Random Forest) or Deep Learning (CNNs) for end-to-end feature extraction.
*   **XRF Integration:** Future versions could integrate XRF spectral data for multi-modal analysis.

---

*Developed by Li Tian for the CHARM Project, University of Helsinki.*