# 🌟 EyeAI

**EyeAI** is a web-based health assistant that combines **machine learning**, **image processing**, and **NLP-powered food analysis** to support people with **Diabetic Retinopathy (DR)**.  

It detects the **stage of DR** from retinal fundus images and recommends **diabetes-friendly meals** from restaurant menus — empowering users to make informed, healthy dining choices.  

---

## 🚀 Features  

- 🧠 **Diabetic Retinopathy Detection**  
  - Preprocessing: CLAHE, grayscale conversion  
  - Feature extraction: GLCM, LBP, entropy, vessel density  
  - Classification: Trained **XGBoost model**  
  - Predicts **5 DR stages** with confidence scores  

- 🍲 **Personalized Food Advisor**  
  - Analyzes restaurant menus via **text input** or **web-scraping**  
  - Extracts nutritional data and evaluates dishes against DR-specific dietary rules  
  - Recommends the most **diabetes-friendly options**  

- 🔗 **Integrated Health Management**  
  - Combines medical diagnosis with nutrition guidance  
  - Provides personalized, stage-aware food recommendations  

---

## 🛠️ Tech Stack  

- **Machine Learning**: XGBoost  
- **Image Processing**: OpenCV, scikit-image  
- **NLP & Data Handling**: NLTK / difflib, pandas  
- **Frontend**: Streamlit  
- **Backend**: Python  

---

## 📊 DR Detection Workflow  

1. **Input**: Retinal fundus image  
2. **Preprocessing**: CLAHE enhancement + grayscale  
3. **Feature Extraction**:  
   - Gray-Level Co-occurrence Matrix (GLCM)  
   - Local Binary Patterns (LBP)  
   - Image entropy  
   - Vessel density  
4. **Classification**: XGBoost model → predicts **No DR, Mild, Moderate, Severe, or Proliferative**  
5. **Output**: Predicted stage + confidence score  

---

## 🍽️ Nutrition Recommendation Workflow  

1. **Input**: Restaurant menu (image/text/web-scraped)  
2. **Processing**: NLP-based dish name matching with nutrition database  
3. **Analysis**: Evaluates calories, carbs, sugars, sodium, etc.  
4. **Output**: Safe vs unsafe dishes with personalized recommendations based on DR stage  

