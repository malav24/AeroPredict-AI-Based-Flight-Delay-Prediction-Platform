import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. Load the preprocessed data
final_df = pd.read_csv("D:\\Semester 6\\Big Data Visualization and Data Analytics\\Project\\preprocessed_flight_data_final.csv")

# 2. Take 10% Sample for testing
sample_df = final_df.sample(frac=0.1, random_state=42)
print(f"Sample size for training: {sample_df.shape}")

# 3. Create the Route Feature (FIXED: Doing this before splitting X and y)
sample_df['route'] = sample_df['Dep_Airport_Encoded'] * 1000 + sample_df['Arr_Airport_Encoded']

# 4. Define Features and Target
X = sample_df.drop(columns=['is_delayed'])
y = sample_df['is_delayed']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Optimized Random Forest Training
print("Training Tuned Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 7. Apply the 0.6 Threshold (The "Precision" Game Changer)
print("Applying 0.6 Confidence Threshold...")
y_probs = model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_probs > 0.6).astype(int)

# 8. Evaluation
print(f"\nAccuracy (0.6 Threshold): {accuracy_score(y_test, y_pred_custom):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom))

# 9. Save the Final Model and the new Feature list
with open('flight_stress_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the column names so the Streamlit app knows the order
with open('model_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\nModel and Features saved successfully!")