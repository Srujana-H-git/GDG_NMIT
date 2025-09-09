# 🎶 Song BPM Prediction using Machine Learning  

> ## 🚀 Predicting the **tempo (Beats Per Minute - BPM)** of a song using **audio features** and advanced machine learning models.  

---

## 📌 Problem Statement  
Given a dataset of audio features like **RhythmScore, Energy, VocalContent, Loudness, TrackDuration**, the goal is to build a regression model that predicts the **Beats Per Minute (BPM)** of songs.  

This problem simulates a real-world **music intelligence system**, where BPM prediction can enhance:  
- 🎧 Music Recommendation Engines  
- 🎶 Playlist Auto-Generation (e.g., workout, chill, party)  
- 🥁 Audio Signal Processing  

---

## 📂 Dataset  
- **Train Dataset** → `train.csv` (Features + Target BPM)  
- **Test Dataset** → `test.csv` (Features only, used for prediction)  
- **Sample Submission** → `sample_submission.csv` (format reference)  

🔗 Dataset provided via competition/assignment (Google Drive link).  

---

## ⚙️ Tech Stack  
- **Python 3.9+**  
- **Pandas & NumPy** → Data handling  
- **Matplotlib & Seaborn** → Visualization  
- **Scikit-learn** → Machine Learning (Regression models, preprocessing, evaluation)  

---

## 🚀 Approach  

1. **Data Preprocessing**  
   - Handled missing values  
   - Standardized features using `StandardScaler`  
   - Train-validation split for robust evaluation  

2. **Exploratory Data Analysis (EDA)**  
   - Feature distributions  
   - Correlation heatmaps  
   - Relationship of features with BPM  

3. **Model Training & Evaluation**  
   - Tried multiple models:  
     - `Linear Regression` (baseline)  
     - `Random Forest Regressor`  
     - `Gradient Boosting Regressor` (Best performing ✅)  
   - Metrics used: **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**  

4. **Prediction & Submission**  
   - Trained best model on full dataset:  
     ```python
     best_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
     best_model.fit(scaler.fit_transform(X), y)

     # Predict on test set
     test_preds = best_model.predict(X_test_scaled)
     ```
   - Created `submission.csv` with format:  
     ```
     ID,BeatsPerMinute
     524164,119.55
     524165,127.42
     524166,111.11
     ```  

---

## 📈 Visualizations  

  - **Feature Distributions**  
  - **Correlation Heatmap**  
  - **Actual vs Predicted BPM Scatter Plot**



## 📊 Results  

| Model                  | MAE ↓  | RMSE ↓  |
|-------------------------|--------|---------|
| Linear Regression       | ~XX.XX | ~XX.XX  |
| Random Forest Regressor | ~XX.XX | ~XX.XX  |
| Gradient Boosting       | **Best** | **Best** |  

✅ **Gradient Boosting Regressor** was chosen as the final model.  

---



