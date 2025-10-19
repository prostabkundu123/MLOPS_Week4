# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data(path: str):
    """Load dataset from CSV file."""
    return pd.read_csv(path)

def train_model(data_path: str = "data/iris.csv", model_path: str = "model/iris_model.joblib"):
    """Train a RandomForest model on the Iris dataset."""
    print("Loading data...")
    df = load_data(data_path)

    # Split into features and labels
    X = df.drop("species", axis=1)
    y = df["species"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully with accuracy: {acc:.4f}")

    # Create models directory if not exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model and scaler
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    return acc

if __name__ == "__main__":
    acc = train_model()
    print("Training completed.")