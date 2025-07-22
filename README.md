# UrbanTech Feedback Classification: NLP Text Modeling with Naive Bayes

## Project Overview

This project demonstrates a complete NLP machine learning pipeline for classifying customer feedback messages from UrbanTech into three categories:

- `app_experience`
- `navigation_issues`
- `service_updates`

Using a dataset of real-world user feedback, the objective is to build a scalable, interpretable classifier that UrbanTech can integrate into its automated feedback handling system.

---

## Objectives

- Clean and preprocess raw feedback text
- Convert text into numerical representations using Bag of Words and TF-IDF
- Train and evaluate Multinomial Naive Bayes classifiers
- Improve performance using grid search
- Create a production-ready classification function

---

## 📁 Dataset

**File:** `urban_feedback.csv`  
**Columns:**
- `feedback_id`: Unique identifier
- `feedback_text`: User-submitted message
- `department`: Labeled category (target variable)

---

## 🛠️ Methodology

### 1. 🧪 Data Exploration
- Loaded and inspected the dataset
- Checked class distribution
- Verified text length and structure

### 2. 🔍 Text Preprocessing
- Converted to lowercase
- Removed punctuation
- Tokenized and lemmatized using `spaCy`
- Removed English stopwords
- Final output: Cleaned and tokenized text for modeling

### 3. 📐 Feature Extraction
- Applied **CountVectorizer** and **TfidfVectorizer**
- Included unigrams and bigrams (`ngram_range=(1, 2)`)
- Limited vocabulary to top 50 features
- Stratified train/test split (80/20)

### 4. 🤖 Model Training
- Trained **Multinomial Naive Bayes** on both BoW and TF-IDF representations
- Extracted and interpreted most informative features per class

### 5. 📊 Model Evaluation
- Generated:
  - Accuracy scores
  - Precision, recall, F1-score (via classification report)
  - Confusion matrices
  - Bar chart comparison of model accuracy

### 6. 🔧 Model Optimization
- Performed `GridSearchCV` to tune:
  - `alpha` (smoothing parameter)
  - Vectorizer parameters: `max_features`, `min_df`, `max_df`, `ngram_range`
- Achieved improved accuracy with optimal settings

### 7. 🚀 Production Function
- Implemented `classify_feedback()`:
  - Accepts raw feedback string
  - Returns predicted department, confidence score, and class probabilities
  - Ready to integrate with live applications

---

## 🧪 Example Predictions

```python
Feedback: "The bus schedule says the next bus is in 5 minutes, but it's already been 20 minutes and no bus."
→ Predicted Department: service_updates
→ Confidence: 0.8651
