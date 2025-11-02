# CRISP-DM Example: Iris Flower Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Data Understanding
data = load_iris(as_frame=True)
df = data.frame
print("Sample data:")
print(df.head())

# 2. Data Preparation
X = df.drop(columns="target")
y = df["target"]

# تقسیم داده به train و test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modeling
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Deployment (پیش‌بینی روی داده جدید)
sample = [[5.1, 3.5, 1.4, 0.2]]  # یه گل با طول و عرض فرضی
prediction = model.predict(sample)
print("Prediction for sample flower:", data.target_names[prediction[0]])
