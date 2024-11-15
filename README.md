# 📰 Fake News Detection

## Overview
This project focuses on detecting fake news articles using Machine Learning techniques. The goal is to classify news articles as either *real* or *fake* using various data processing, feature engineering, and classification methods.

## 🚀 Project Highlights
- **Language Used:** Python
- **Libraries:** `pandas`, `numpy`, `re`, `nltk`, `scikit-learn`
- **Key Techniques:**
  - Data preprocessing (tokenization, stemming, stopword removal)
  - Feature extraction using TF-IDF Vectorization
  - Machine Learning model for classification

## ⚙️ Dataset
- **Training Data:** Due to the large size, the dataset wasn't uploaded here. However, it primarily consists of labeled news articles with fields such as `title`, `text`, `label` (1 for fake and 0 for real).
- **Source:** A CSV file was used containing articles and their labels.

## 📊 Data Preprocessing Steps
- **Stopword Removal:** Removed common stopwords using the NLTK library.
- **Stemming:** Applied stemming using the Porter Stemmer for normalization.
- **Tokenization:** Split text into tokens for analysis.

## 🛠️ Implementation Details
1. **Data Loading:**
   ```python
   news_dataset = pd.read_csv('/path/to/dataset.csv')
2. **Text Processing:** Handled missing values, tokenized, removed stopwords, and stemmed words.
   ```python
   def text_preprocessing(text):
    # Remove special characters, numbers, etc., using regular expressions
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()  # Convert text to lowercase
    text = text.split()  # Tokenization
    
    # Remove stopwords and apply stemming
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

    # Applying preprocessing function to text column of the dataset
    news_dataset['processed_text'] = news_dataset['text'].apply(text_preprocessing)
4. **Feature Extraction:** Used TF-IDF Vectorizer to convert text data into numerical features.
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
   X = vectorizer.fit_transform(news_dataset['text'])
5. **Data Splitting:** Using train_test_split to split into training and testing data
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, news_dataset['label'], test_size=0.2, random_state=2)

6. **Model Training:** Utilized Logistic Regression to train the model
   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression()
   model.fit(X_train, y_train)
## 💡 Future Improvements
- Integrate more sophisticated NLP models such as Transformers (e.g., BERT).
- Hyperparameter tuning to improve model performance.
- Explore additional features such as metadata.
