import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import joblib
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import frangi
import pytesseract
import re
from difflib import get_close_matches
import os

# Set page config with improved layout
st.set_page_config(
    page_title="EyeAI",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "### Diabetic Retinopathy Detection with Personalized Nutrition Advisor\n\nThis app helps detect diabetic retinopathy stages and recommends safe foods based on your condition."
    }
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Text */
    p, div, span {
        color: #FAFAFA !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #1E1E1E;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.5rem;
        border-radius: 0.5rem;
        background-color: #1E1E1E;
        color: #AAAAAA;
        border: 1px solid #333333;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4B286D;
        color: #FFFFFF !important;
        font-weight: bold;
        border: 1px solid #6E3CA1;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #4B286D;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #6E3CA1;
        transform: scale(1.05);
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px dashed #4B286D;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: 1px solid #333333;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #333333;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1E1E1E !important;
        color: #FAFAFA !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #6E3CA1;
    }
    
    /* Diagnosis boxes */
    .diagnosis-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    .no-dr {background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); border-left: 5px solid #4caf50;}
    .mild {background: linear-gradient(135deg, #f57f17 0%, #f9a825 100%); border-left: 5px solid #ffc107;}
    .moderate {background: linear-gradient(135deg, #ff6f00 0%, #ff8f00 100%); border-left: 5px solid #ff9800;}
    .severe {background: linear-gradient(135deg, #c62828 0%, #d32f2f 100%); border-left: 5px solid #f44336;}
    .proliferate-dr {background: linear-gradient(135deg, #6a1b9a 0%, #8e24aa 100%); border-left: 5px solid #9c27b0;}
    
    /* Food indicators */
    .food-safe {color: #4caf50; font-weight: bold;}
    .food-unsafe {color: #f44336;}
    
    /* Section titles */
    .section-title {
        border-bottom: 2px solid #4B286D; 
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Feature boxes */
    .feature-box {
        padding: 1rem; 
        background-color: #1E1E1E; 
        border-radius: 0.5rem;
        border: 1px solid #333333;
        margin-bottom: 0.5rem;
    }
    
    /* Divider styling */
    hr {
        border-color: #4B286D;
        margin: 2rem 0;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #1E1E1E;
        border: 1px solid #4B286D;
        border-radius: 0.5rem;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #1E1E1E;
        border: 1px solid #2E7D32;
        border-radius: 0.5rem;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #1E1E1E;
        border: 1px solid #F57F17;
        border-radius: 0.5rem;
    }
    
    /* Error boxes */
    .stError {
        background-color: #1E1E1E;
        border: 1px solid #C62828;
        border-radius: 0.5rem;
    }

</style>
""", unsafe_allow_html=True)

# --- Constants ---
FEATURE_NAMES = [
    "entropy", "contrast", "homogeneity", "energy", "correlation",
    "blur", "vessel_area"
] + [f"lbp_{i}" for i in range(10)]

# Corrected DR stages based on the dataset
DR_STAGES = {
    "No_DR": "No DR",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "Proliferate_DR": "Proliferative DR"
}

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("j48_xgb_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        print(scaler)
        return model, label_encoder, scaler
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

model, label_encoder, scaler = load_models()

# --- Nutrition Database Functions ---
@st.cache_data
def load_database():
    try:
        db = pd.read_csv('data/Indian_Food_Nutrition_Processed.csv')
        db.columns = [col.strip().lower().replace("_", " ") for col in db.columns]
        if "dish name" not in db.columns:
            possible_cols = [c for c in db.columns if "dish" in c]
            if possible_cols:
                db.rename(columns={possible_cols[0]: "dish name"}, inplace=True)
        return db
    except Exception as e:
        st.error(f"Failed to load nutrition database: {e}")
        return pd.DataFrame()

def clean_text(text):
    return re.sub(r"[^a-zA-Z\s]", "", text).lower().strip()

def extract_menu_from_image(image):
    gray_image = image.convert("L")
    text = pytesseract.image_to_string(gray_image, lang="eng")
    raw_items = re.split(r'[\n,@#&/()]', text)
    menu_items = [clean_text(item) for item in raw_items if len(clean_text(item)) > 3]
    return menu_items

def match_menu_items(menu_items, db):
    matched_rows = []
    db_dishes = db["dish name"].str.lower().tolist()
    for item in menu_items:
        if not item or item in ["soup", "noodles", "curd", "papad", "raita", "roll", "fries", "paratha", "naan", "roti", "biryani"]:
            continue
        close = get_close_matches(item, db_dishes, n=1, cutoff=0.6)
        if close:
            match = db[db["dish name"].str.lower() == close[0]]
        else:
            match = db[db["dish name"].str.lower().str.contains(item, na=False)]
        if not match.empty:
            matched_rows.append(match)
    return pd.concat(matched_rows).drop_duplicates() if matched_rows else pd.DataFrame()

def is_food_safe(row, dr_stage):
    nutrition_values = {
        "carbs": row.get("carbs (g)", 0),
        "sugars": row.get("sugars (g)", 0),
        "sodium": row.get("sodium (mg)", 0),
        "calories": row.get("calories", 0),
        "fat": row.get("fat (g)", 0),
        "fiber": row.get("fiber (g)", 0),
        "protein": row.get("protein (g)", 0)
    }
    
    thresholds = {
        "No_DR": {
            "carbs": float('inf'),
            "sugars": float('inf'),
            "sodium": float('inf'),
            "calories": float('inf'),
            "fat": float('inf'),
            "fiber": 0,
            "protein": 0
        },
        "Mild": {
            "carbs": 180,
            "sugars": 90,
            "sodium": 1900,
            "calories": 800,
            "fat": 50,
            "fiber": 0,
            "protein": 0
        },
        "Moderate": {
            "carbs": 65,
            "sugars": 22,
            "sodium": 1000,
            "calories": 500,
            "fat": 20,
            "fiber": 1,
            "protein": 2
        },
        "Severe": {
            "carbs": 50,
            "sugars": 15,
            "sodium": 500,
            "calories": 400,
            "fat": 15,
            "fiber": 1,
            "protein": 2
        },
        "Proliferate_DR": {
            "carbs": 40,
            "sugars": 10,
            "sodium": 300,
            "calories": 300,
            "fat": 10,
            "fiber": 2,
            "protein": 3
        }
    }
    
    if dr_stage not in thresholds:
        return False
    
    return all(
        nutrition_values[nutrient] <= thresholds[dr_stage][nutrient] 
        if nutrient not in ["fiber", "protein"] 
        else nutrition_values[nutrient] >= thresholds[dr_stage][nutrient]
        for nutrient in nutrition_values
    )

# --- Image Processing Functions ---
def preprocess_image(image):
    img = image.resize((224, 224)).convert("RGB")
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def extract_image_features(image):
    try:
        gray = preprocess_image(image)
        entropy = shannon_entropy(gray)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                          symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        vessels = frangi(gray)
        vessel_area = np.sum(vessels > 0.5) / vessels.size
        return [entropy, contrast, homogeneity, energy, correlation, blur, vessel_area] + lbp_hist.tolist()
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        return None

# --- Main App ---
st.title("👁️ NutriVision")
st.markdown("### Diabetic Retinopathy Detection with Personalized Nutrition Advisor")

# Initialize session state
if 'dr_stage' not in st.session_state:
    st.session_state.dr_stage = None
if 'diagnosis_made' not in st.session_state:
    st.session_state.diagnosis_made = False

# Tab interface
tab1, tab2 = st.tabs(["🧬 Diabetic Retinopathy Detection", "🍽️ Personalized Food Advisor"])

with tab1:
    st.header("Retinal Image Analysis", divider=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload retinal scan", 
            type=["jpg", "jpeg", "png"], 
            help="Upload a clear image of the retina for analysis"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Retinal Scan", width=350)

            if st.button("Analyze Retinal Image", type="primary", use_container_width=True):
                with st.spinner("Extracting retinal features..."):
                    features = extract_image_features(image)
                    if features:
                        df = pd.DataFrame([features], columns=FEATURE_NAMES)
                        df_scaled = scaler.transform(df)
                        prediction = model.predict(df_scaled)[0]
                        dr_class = label_encoder.inverse_transform([prediction])[0]
                        probabilities = model.predict_proba(df_scaled)[0]

                        st.session_state.dr_stage = dr_class
                        st.session_state.diagnosis_made = True
                        st.session_state.dr_probabilities = probabilities

                        # --- Diagnosis Result Card ---
                        diagnosis_class = dr_class.lower().replace("_", "-")
                        st.markdown(f"""
                        <div class="diagnosis-box {diagnosis_class}">
                            <h3>🧾 Diagnosis Result</h3>
                            <p><strong>Stage:</strong> {DR_STAGES.get(dr_class, dr_class)}</p>
                            <p><strong>Confidence:</strong> {probabilities[prediction]*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.divider()

                        # --- Probabilities ---
                        with st.expander("Stage Probabilities", expanded=True):
                            prob_df = pd.DataFrame({
                                "Stage": [DR_STAGES.get(stage, stage) for stage in label_encoder.classes_],
                                "Probability": probabilities
                            })
                            st.bar_chart(prob_df.set_index("Stage"))

                        # --- Feature Importance ---
                        with st.expander("Key Analysis Factors", expanded=False):
                            important_features = {
                                "Vessel Abnormalities": features[6],
                                "Texture Contrast": features[1],
                                "Image Sharpness": features[5]
                            }
                            for feature, value in important_features.items():
                                st.progress(min(int(value * 100), 100), 
                                           text=f"{feature}: {value:.2f}")

    with col2:
        if st.session_state.diagnosis_made:
            st.subheader("About Your Diagnosis")
            dr_info = {
                "No_DR": "✅ No signs of diabetic retinopathy detected. Maintain regular checkups.",
                "Mild": "🟡 Early signs of retinopathy. Manage blood sugar carefully.",
                "Moderate": "🟠 Moderate non-proliferative retinopathy. Requires medical attention.",
                "Severe": "🔴 Severe non-proliferative retinopathy. Urgent medical care needed.",
                "Proliferate_DR": "🟣 Proliferative diabetic retinopathy. Immediate specialist care required."
            }
            st.info(dr_info.get(st.session_state.dr_stage, "Consult your doctor for detailed interpretation."))

            st.subheader("Next Steps")
            next_steps = {
                "No_DR": "✅ Continue regular eye exams (annually)",
                "Mild": "⚠️ Schedule follow-up in 6 months",
                "Moderate": "❗ Consult an ophthalmologist within 1 month",
                "Severe": "🚨 Consult an ophthalmologist immediately",
                "Proliferate_DR": "🚨 Emergency consultation required immediately"
            }
            
            status_type = {
                "No_DR": st.success,
                "Mild": st.warning,
                "Moderate": st.warning,
                "Severe": st.error,
                "Proliferate_DR": st.error
            }
            
            status_type[st.session_state.dr_stage](next_steps.get(st.session_state.dr_stage, "Consult your doctor"))

            st.markdown("""
            <div style="margin-top: 2rem; padding: 1rem; border-radius: 0.5rem; background-color: #1E1E1E; border: 1px solid #4B286D;">
                <h4>🍽️ Food Advisor Ready</h4>
                <p>Switch to the <strong>Personalized Food Advisor</strong> tab for dietary recommendations based on your diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("Personalized Food Advisor", divider=True)

    # --- DR Stage Section ---
    if st.session_state.diagnosis_made:
        st.success(
            f"Using DR stage from your retinal analysis: "
            f"**{DR_STAGES.get(st.session_state.dr_stage, st.session_state.dr_stage)}**"
        )
        dr_stage = st.session_state.dr_stage
        st.info("You can manually change this below if needed.")
    else:
        st.warning("No retinal analysis detected. Please analyze an image or select your DR stage manually.")
    
    manual_dr_stage = st.selectbox(
        "Select or confirm DR Stage",
        options=list(DR_STAGES.keys()),
        format_func=lambda x: DR_STAGES.get(x, x),
        index=list(DR_STAGES.keys()).index(st.session_state.dr_stage) if st.session_state.dr_stage else 0
    )
    dr_stage = manual_dr_stage

    st.divider()

    # --- Menu Input Section ---
    st.subheader("Menu Input Options", divider=True)
    menu_source = st.radio(
        "Choose how to input your menu:", 
        ["Upload Image", "Enter Text"], 
        horizontal=True
    )

    menu_items = []
    if menu_source == "Upload Image":
        uploaded_image = st.file_uploader(
            "Upload a menu image", 
            type=["png", "jpg", "jpeg"],
            help="Take a photo of a menu or nutrition label"
        )
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Menu", width=300)
            if st.button("Extract Text from Menu", use_container_width=True):
                with st.spinner("Extracting text from menu..."):
                    menu_items = extract_menu_from_image(image)
                    st.session_state.menu_items = menu_items
    else:
        menu_text = st.text_area(
            "Paste menu items (one per line):",
            height=150,
            placeholder="Enter food items...\nExample:\nChicken Biryani\nDal Tadka\nRoti"
        )
        if menu_text:
            menu_items = [line.strip() for line in menu_text.split("\n") if line.strip()]
            st.session_state.menu_items = menu_items

    if 'menu_items' in st.session_state and st.session_state.menu_items:
        with st.expander("Show extracted menu items", expanded=False):
            st.write(st.session_state.menu_items)

    st.divider()

    # --- Analysis Section ---
    db = load_database()

    if st.button("Analyze Food Options", type="primary", use_container_width=True) and 'menu_items' in st.session_state and st.session_state.menu_items and not db.empty:
        with st.spinner("Analyzing nutrition and matching dishes..."):
            matched_df = match_menu_items(st.session_state.menu_items, db)
            
            if not matched_df.empty:
                matched_df["Safe to Eat?"] = matched_df.apply(
                    lambda row: "✅ Yes" if is_food_safe(row, dr_stage) else "❌ No", 
                    axis=1
                )

                st.subheader("Food Analysis Results", divider=True)

                # --- Summary metrics ---
                safe_count = (matched_df["Safe to Eat?"] == "✅ Yes").sum()
                unsafe_count = (matched_df["Safe to Eat?"] == "❌ No").sum()
                col1, col2 = st.columns(2)
                col1.metric("Safe Options", safe_count, delta_color="normal")
                col2.metric("Unsafe Options", unsafe_count, delta_color="inverse")

                # --- Detailed table ---
                with st.expander("Detailed Results Table", expanded=True):
                    display_cols = ["dish name", "Safe to Eat?"] + [c for c in matched_df.columns if c not in ["dish name", "Safe to Eat?"]]
                    st.dataframe(
                        matched_df[display_cols].style.applymap(
                            lambda x: 'background-color: #1B5E20' if x == "✅ Yes" else 'background-color: #C62828',
                            subset=["Safe to Eat?"]
                        ),
                        use_container_width=True,
                        height=min(400, 35 * len(matched_df) + 35)
                    )

                st.divider()

                # --- Nutrition Insights ---
                st.subheader("Nutrition Insights", divider=True)
                nutrition_cols = [c for c in matched_df.columns if c.strip() in [
                    "calories", "carbs (g)", "sugars (g)", "sodium (mg)", 
                    "fat (g)", "fiber (g)", "protein (g)"
                ]]

                if nutrition_cols:
                    safe_df = matched_df[matched_df["Safe to Eat?"] == "✅ Yes"]

                    if not safe_df.empty:
                        st.success(f"Found **{len(safe_df)} safe options** for your DR stage!")

                        avg_nutrition = safe_df[nutrition_cols].mean()
                        st.markdown("**Average nutrition per safe dish:**")
                        st.dataframe(avg_nutrition.to_frame().T.style.format("{:.1f}"))

                        # Highlight best & worst calorie dishes
                        if "calories" in matched_df.columns:
                            best_dish = safe_df.loc[safe_df["calories"].idxmin()]
                            worst_dish = safe_df.loc[safe_df["calories"].idxmax()]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🥗 Lowest Calorie Option", 
                                          f"{best_dish['calories']:.0f} kcal", 
                                          best_dish["dish name"])
                            with col2:
                                st.metric("🍛 Highest Calorie Option", 
                                          f"{worst_dish['calories']:.0f} kcal", 
                                          worst_dish["dish name"])
                    else:
                        st.warning("No safe dishes found for your current DR stage.")
            else:
                st.error("No matching dishes found in the database.")

    # --- Footer ---
    st.divider()
    st.caption("""
    ⚕️ This application provides health suggestions based on image analysis and nutritional data.  
    It is not a substitute for professional medical advice. Always consult your healthcare provider.
    """)