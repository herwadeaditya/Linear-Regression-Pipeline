import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ==============================
# LOAD DATASET
# ==============================

file_path = input("\nEnter full CSV file path: ")

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"\n❌ Error loading file: {e}")
    sys.exit()

print("\n" + "="*50)
print("📊 DATASET SUMMARY")
print("="*50)
print(f"Total Rows       : {df.shape[0]}")
print(f"Total Columns    : {df.shape[1]}")
print("="*50)


# ==============================
# REMOVE ID-LIKE COLUMNS
# ==============================

for col in df.columns:
    if df[col].nunique() == len(df):
        df = df.drop(columns=[col])


# ==============================
# REMOVE HIGHLY MISSING COLUMNS
# ==============================

df = df.dropna(thresh=len(df)*0.6, axis=1)
df = df.dropna(axis=1, how='all')


# ==============================
# HANDLE MISSING VALUES
# ==============================

# Fill numeric columns with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])


# ==============================
# ENCODE CATEGORICAL VARIABLES
# ==============================

df = pd.get_dummies(df, drop_first=True)

print("\n✅ Data cleaning completed successfully.")


# ==============================
# TARGET SELECTION
# ==============================

print("\nAvailable Columns:")
print("=" * 40)
for col in df.columns:
    print(col)
print("=" * 40)

target = input("\nEnter Target Column Name: ")

if target not in df.columns:
    print("\n❌ Invalid target column.")
    sys.exit()


# ==============================
# REMOVE OUTLIERS (IQR METHOD)
# ==============================

Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1

df = df[(df[target] > Q1 - 1.5 * IQR) &
        (df[target] < Q3 + 1.5 * IQR)]


# ==============================
# REMOVE WEAK FEATURES
# ==============================

correlation = df.corr()[target].abs()
strong_features = correlation[correlation > 0.1].index
df = df[strong_features]


# ==============================
# SPLIT FEATURES & TARGET
# ==============================

X = df.drop(columns=[target])
y = df[target]


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)


# ==============================
# MODEL TRAINING
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)


# ==============================
# FINAL REPORT
# ==============================

print("\n" + "="*50)
print("📈 MODEL PERFORMANCE REPORT")
print("="*50)

regression_type = (
    "Simple Linear Regression"
    if X.shape[1] == 1
    else "Multiple Linear Regression"
)

print(f"Model Type        : {regression_type}")
print(f"Number of Features: {X.shape[1]}")
print(f"R² Score          : {round(score, 4)}")

if score >= 0.85:
    print("Performance Level : ⭐ Excellent")
elif score >= 0.70:
    print("Performance Level : 👍 Good")
elif score >= 0.50:
    print("Performance Level : ⚠ Moderate")
else:
    print("Performance Level : ❌ Needs Improvement")

print("="*50)
print("🎯 Model training completed successfully!")
print("="*50)

