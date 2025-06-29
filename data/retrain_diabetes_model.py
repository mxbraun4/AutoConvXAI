"""Retrain diabetes model with current sklearn version."""
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

print("🔄 Retraining diabetes model with current sklearn version...")

# Set random seed for reproducibility
np.random.seed(42)

# Load data
print("📊 Loading diabetes dataset...")
df = pd.read_csv("diabetes.csv")
X = df.drop('y', axis=1)
y = df['y']

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Split data (same as original)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Create pipeline with LogisticRegression (as expected by the system)
print("🤖 Training LogisticRegression model...")
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(C=1.0, max_iter=10000, random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate performance
train_score = model_pipeline.score(X_train, y_train)
test_score = model_pipeline.score(X_test, y_test)

print(f"✅ Training accuracy: {train_score:.3f}")
print(f"✅ Test accuracy: {test_score:.3f}")

# Make predictions for detailed evaluation
y_pred = model_pipeline.predict(X_test)
print(f"✅ Validation accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Save the model with the correct filename
model_path = "diabetes_model_logistic_regression.pkl"
with open(model_path, "wb") as f:
    pkl.dump(model_pipeline, f)

print(f"💾 Model saved to: {model_path}")
print("🎉 Retraining completed successfully!")

# Test the saved model works
print("\n🧪 Testing saved model...")
with open(model_path, "rb") as f:
    loaded_model = pkl.load(f)

test_predictions = loaded_model.predict(X_test)
final_accuracy = accuracy_score(y_test, test_predictions)
print(f"✅ Final model accuracy: {final_accuracy:.3f}")
print("✅ Model successfully retrained and ready to use!") 