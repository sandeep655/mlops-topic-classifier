import os
import joblib
import mlflow
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

mlflow.set_experiment("topic-classification-logreg")

mlflow.set_tracking_uri(f"file://{tempfile.gettempdir()}/mlruns")


with mlflow.start_run():
    categories = ['comp.sys.ibm.pc.hardware', 'rec.sport.baseball']
    data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    texts = data.data
    labels = data.target

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Validation Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_artifact("models/logistic_model.pkl")
    mlflow.log_artifact("models/vectorizer.pkl")

    print("Model trained, validated, logged to MLflow, and saved to disk.")