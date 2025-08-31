# Download necessary NLTK data(once)
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# Fake Review Detection - Final Year Project Script
# Includes Preprocessing, Feature Engineering, and Model Comparison

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load dataset
df = pd.read_csv('fake reviews dataset.csv')

# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

def clean_text(text):
    tokens = tokenizer.tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['cleaned_review'] = df['review'].apply(clean_text)
df['label'] = df['label'].map({'CG': 0, 'OR': 1})

# Feature Engineering
df['sentiment'] = df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['review_length'] = df['review'].apply(lambda x: len(str(x)))
df['exaggeration'] = df['review'].apply(lambda x: str(x).count('!'))

# TF-IDF Vectorization
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_tfidf = tfidf.fit_transform(df['cleaned_review'])

# Scale numeric metadata
scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(df[['sentiment', 'review_length', 'exaggeration']])

# Combine all features
from scipy.sparse import hstack
X = hstack((X_tfidf, X_meta_scaled))
y = df['label'].values

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle Imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Separate TF-IDF for Naive Bayes (requires non-negative features)
X_nb_train, X_nb_test = X_train_res[:, :5000], X_test[:, :5000]

# 9. Define My Models
myModels = {
    "Logistic Regression": LogisticRegression(solver='saga', max_iter=5000),
    "Random Forest": RandomForestClassifier(),
    #"SVM": SVC(kernel='linear', probability=True),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
}
# 11. Define Comparison-Only Models
compareModels = {
    "Naive Bayes": MultinomialNB(),  # TF-IDF only
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier()
}
#To save the results for comparison
my_results = {}
compare_results = {}

# 10. Train and Evaluate My Models
for name, model in myModels.items():
    print(f"\n=== {name} ===")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fake"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score: {auc:.4f}")
    # Collect metrics into my_results
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    my_results[name] = {
        "Accuracy": round(report["accuracy"], 4),
        "Precision": round(report["1"]["precision"], 4),
        "Recall": round(report["1"]["recall"], 4),
        "F1-Score": round(report["1"]["f1-score"], 4),
        "ROC-AUC": round(auc, 4) if auc else "N/A"
    }
# Plot confusion matrix heatmaps for each of your models
for name, model in myModels.items():
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Fake'], yticklabels=['Genuine', 'Fake'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
# Train and evaluate all models
# 12. Train and Compare Models (No Printouts)
for name, model in compareModels.items():
    if name == "Naive Bayes" or name == "Naive Bayes":
        model.fit(X_train_res[:, :5000], y_train_res)
        y_pred = model.predict(X_test[:, :5000])
        y_proba = model.predict_proba(X_test[:, :5000])[:, 1]
    elif name == "LightGBM":
        # Convert to dense arrays to avoid warning
        X_train_res_lgbm = X_train_res.toarray()
        X_test_lgbm = X_test.toarray()
        model.fit(X_train_res_lgbm, y_train_res)
        y_pred = model.predict(X_test_lgbm)
        y_proba = model.predict_proba(X_test_lgbm)[:, 1]
    else:
        if hasattr(model, "fit"):
            model.fit(X_train_res, y_train_res)
        if hasattr(model, "predict"):
            y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    compare_results[name] = {
        "Accuracy": round(report["accuracy"], 4),
        "Precision": round(report["1"]["precision"], 4),
        "Recall": round(report["1"]["recall"], 4),
        "F1-Score": round(report["1"]["f1-score"], 4),
        "ROC-AUC": round(auc, 4) if auc else "N/A"
    }
# Show the Comparison Table
# Merge both results
all_results = {**my_results, **compare_results}
comparison_df = pd.DataFrame(all_results).T

print("\n=== Combined Performance Table: My Models vs Comparison Models ===")
print(comparison_df)

# === Plot F1-Score Comparison Bar Chart ===
import matplotlib.pyplot as plt
import numpy as np

# Clean and convert data
plot_df = comparison_df.replace("N/A", np.nan).dropna().astype(float)

# Plot F1-Score
plt.figure(figsize=(12, 6))
plot_df["F1-Score"].plot(kind="bar", color="skyblue")
plt.title("F1-Score Comparison of All Models")
plt.ylabel("F1-Score")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# Save performance table to Excel
#comparison_df.to_excel("Model_Performance_Comparison.xlsx")

# Show top performers sorted by F1-Score
print("\nTop Models by F1-Score:")
print(comparison_df.sort_values(by="F1-Score", ascending=False))
# plot Accuracy:
comparison_df.replace("N/A", np.nan).dropna().astype(float)["Accuracy"].plot(
    kind="bar", figsize=(12, 6), title="Accuracy Comparison of Models", ylabel="Accuracy"
)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#saving model
import joblib

# Save best model: SVM
best_model = myModels["MLP Neural Network"]
joblib.dump(best_model, "fake_review_model.pkl")

# Save TF-IDF vectorizer and scaler too
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")