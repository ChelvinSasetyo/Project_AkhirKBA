import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Membaca data bersih
df = pd.read_csv("cleaned_superstore.csv")

# Target machine learning
y = df["profit_status"]

# Fitur yang digunakan untuk prediksi
features = [
    "sales",
    "quantity",
    "discount",
    "shipping_cost",
    "shipping_days",
    "segment",
    "market",
    "region",
    "category",
    "sub_category",
    "ship_mode",
    "order_priority"
]

X = df[features]

# Kolom numerik dan kategori
numeric_features = [
    "sales",
    "quantity",
    "discount",
    "shipping_cost",
    "shipping_days"
]

categorical_features = [
    "segment",
    "market",
    "region",
    "category",
    "sub_category",
    "ship_mode",
    "order_priority"
]

# Preprocessing untuk data kategori
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# Model machine learning
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Melatih model
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)

print("Akurasi Model:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))