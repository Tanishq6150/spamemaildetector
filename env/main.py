import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

# Load dataset
file_path = r"C:\Users\tanis\Desktop\E-mail spam detection\EmailSpamCollection"
df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Check if preprocessing is already done
if 'clean_text' not in df.columns:
    df['clean_text'] = df['text'].apply(clean_text)
    print("Text cleaned and saved!")

# Save preprocessed data so you don't have to clean it again
df.to_csv("cleaned_data.csv", index=False)

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['label']

print("Feature extraction completed! Shape of X:", X.shape)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Step 1: Light Undersampling of Ham (Keep more spam)
undersample = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
X_under, y_under = undersample.fit_resample(X, y)

# Step 2: Apply SMOTE to create synthetic spam samples
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_under, y_under)

# Check new class balance
print("After Resampling:", pd.Series(y_resampled).value_counts())

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

# ✅ Create a DataFrame for resampled labels
resampled_df = pd.DataFrame({'label': y_resampled})

# 📊 **Spam vs. Ham Count (Resampled)**
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=resampled_df, palette=['green', 'red'])
plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])
plt.title("Spam vs. Ham Count (After Resampling)")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

# 🥧 **Pie Chart for Resampled Data**
plt.figure(figsize=(6,6))
plt.pie(resampled_df['label'].value_counts(), labels=['Ham', 'Spam'], autopct='%1.1f%%', 
        colors=['green', 'red'], startangle=90)
plt.title("Spam vs. Ham Distribution (After Resampling)")
plt.show()

# 🔠 **Convert Resampled Features Back to Text**
resampled_text_df = pd.DataFrame(X_resampled, columns=vectorizer.get_feature_names_out())  # ✅ Fixed `.toarray()`
resampled_text_df['label'] = y_resampled  # Add labels back

# **Extract words for WordClouds**
spam_words_resampled = ' '.join(resampled_text_df[resampled_text_df['label'] == 1].drop('label', axis=1).sum().index)
ham_words_resampled = ' '.join(resampled_text_df[resampled_text_df['label'] == 0].drop('label', axis=1).sum().index)

# ☁️ **Spam WordCloud**
plt.figure(figsize=(10,5))
wordcloud_spam = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(spam_words_resampled)
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Spam Messages (After Resampling)", fontsize=14)
plt.show()

# ☁️ **Ham WordCloud**
plt.figure(figsize=(10,5))
wordcloud_ham = WordCloud(width=800, height=400, background_color='black', colormap='Greens').generate(ham_words_resampled)
plt.imshow(wordcloud_ham, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Ham Messages (After Resampling)", fontsize=14)
plt.show()

# 📝 **Message Length Analysis (Resampled)**
# ✅ Ensure `message_length` column exists
resampled_df['message_length'] = X_resampled.sum(axis=1)

plt.figure(figsize=(8,5))
sns.histplot(resampled_df[resampled_df['label'] == 0]['message_length'], bins=30, color='green', label='Ham', kde=True)
sns.histplot(resampled_df[resampled_df['label'] == 1]['message_length'], bins=30, color='red', label='Spam', kde=True)
plt.legend()
plt.title("Message Length Distribution (After Resampling)")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

# ✅ Ensure these variables are defined:
# - X_before, y_before → Original dataset (Before resampling)
# - X_resampled, y_resampled → Resampled dataset (After resampling)

X_before = X  # The full dataset's text messages
y_before = y  # The full dataset's labels (0 for Ham, 1 for Spam)


# Convert labels to DataFrames
original_df = pd.DataFrame({'label': y_before})  # Before Resampling
resampled_df = pd.DataFrame({'label': y_resampled})  # After Resampling

# 🔹 **1️⃣ Spam vs. Ham Count (Before vs. After)**
fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.countplot(x='label', data=original_df, palette=['green', 'red'], ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Ham', 'Spam'])
axes[0].set_title("Spam vs. Ham Count (Before Resampling)")
axes[0].set_xlabel("Message Type")
axes[0].set_ylabel("Count")

sns.countplot(x='label', data=resampled_df, palette=['green', 'red'], ax=axes[1])
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Ham', 'Spam'])
axes[1].set_title("Spam vs. Ham Count (After Resampling)")
axes[1].set_xlabel("Message Type")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# 🔹 **2️⃣ Pie Chart Comparison**
fig, axes = plt.subplots(1, 2, figsize=(12,6))

axes[0].pie(original_df['label'].value_counts(), labels=['Ham', 'Spam'], autopct='%1.1f%%', 
            colors=['green', 'red'], startangle=90)
axes[0].set_title("Spam vs. Ham Distribution (Before Resampling)")

axes[1].pie(resampled_df['label'].value_counts(), labels=['Ham', 'Spam'], autopct='%1.1f%%', 
            colors=['green', 'red'], startangle=90)
axes[1].set_title("Spam vs. Ham Distribution (After Resampling)")

plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ✅ **Split Data (Train & Test)**
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ **Convert Sparse Matrix to Dense for GaussianNB (Only)**
X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

# ✅ **Initialize Models**
models = {
    "MultinomialNB": MultinomialNB(),
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
}

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ✅ **Train & Evaluate Models**
results = []
for name, model in models.items():
    # Use dense matrix for GaussianNB, sparse for others
    if name == "GaussianNB":
        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='binary')

    results.append([name, accuracy, precision, recall, f1])

# ✅ **Create a DataFrame for Comparison**
df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
df_results = df_results.sort_values(by="F1 Score", ascending=False)

# ✅ **Display Results**
print(df_results)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(x="F1 Score", y="Model", data=df_results, palette="viridis")
plt.xlabel("F1 Score")
plt.ylabel("Models")
plt.title("Model Performance Comparison")
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ✅ **Split Data**
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ **Initialize Base Models**
models = {
    "Naïve Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# ✅ **Train Base Models and Get Predictions**
meta_features_train = []
meta_features_test = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Store predictions for stacking
    meta_features_train.append(y_train_pred)
    meta_features_test.append(y_test_pred)

    # Print Model Performance
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print(f"🔹 {name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

# ✅ **Stack Predictions for Meta Model**
meta_features_train = np.array(meta_features_train).T  # Transpose to match shape
meta_features_test = np.array(meta_features_test).T

# ✅ **Train Meta Model (Logistic Regression)**
meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y_train)

# ✅ **Final Predictions**
final_pred = meta_model.predict(meta_features_test)

# ✅ **Evaluate Stacked Model**
final_acc = accuracy_score(y_test, final_pred)
final_prec = precision_score(y_test, final_pred)
final_rec = recall_score(y_test, final_pred)
final_f1 = f1_score(y_test, final_pred)

print("\n🔥 **Stacked Model Performance** 🔥")
print(f"✅ Accuracy: {final_acc:.4f}")
print(f"✅ Precision: {final_prec:.4f}")
print(f"✅ Recall: {final_rec:.4f}")
print(f"✅ F1 Score: {final_f1:.4f}")

# Get the best model based on F1 Score
best_model_name = df_results.iloc[0]['Model']
best_model = models[best_model_name]  # Retrieve the best model

# Save best model & vectorizer
import joblib
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(f"✅ Best Model ({best_model_name}) saved successfully!")



