# Core libraries for data processing, model building, and evaluation
from sklearn.decomposition import PCA  # For feature dimensionality reduction
import pandas as pd  # For loading the file called 'bbc-text.csv'
import numpy as np  # For numerical operations
from sklearn.model_selection import KFold  # For K-fold cross-validation
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF feature extraction
from sklearn.feature_selection import SelectKBest, chi2  # For feature selection (Chi-Square test)
from sklearn.linear_model import LogisticRegression  # For Logistic Regression model
from sklearn.pipeline import Pipeline  # For integrating feature engineering and modeling
from sklearn.base import BaseEstimator, TransformerMixin  # Base classes for custom transformers
from sklearn.metrics import precision_recall_fscore_support, accuracy_score  # For evaluation metrics
from sklearn.preprocessing import StandardScaler  # For feature normalization
import re  # Regular expressions for text preprocessing
from nltk.corpus import stopwords  # For removing stopwords
from nltk.tokenize import word_tokenize  # For text tokenization
from gensim.models import Word2Vec  # For word embeddings

# Step 1
# Load dataset
# Load the dataset, which should have two columns:
dataset = pd.read_csv('bbc-text.csv')
X = dataset['text']  # Feature column containing text data
y = dataset['category']  # Target column containing labels

# Step 2
# Text preprocessing
# Function to preprocess the text data:
# Removes punctuation
# Converts text to lowercase
# Tokenizes text into words
# Removes common English stopwords
def preprocess(text):
    stop_words = set(stopwords.words('english'))  # Load standard English stopwords
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    words = word_tokenize(text)  # Tokenize text into words
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return words

# Apply preprocessing to all text in the dataset
X_processed = X.apply(preprocess)

# Step 3
# Train Word2Vec model
# Word2Vec generates word embeddings that capture semantic meaning
word2vec_model = Word2Vec(
    sentences=X_processed,  # Preprocessed sentences
    vector_size=100,  # Embedding dimension size
    window=5,  # Context window size for neighboring words
    min_count=1,  # Minimum frequency threshold for words to be included
    workers=4  # Number of parallel threads
)

# Function to compute the average Word2Vec embedding for a document
def get_avg_word2vec(words, model, vector_size):
    vectors = [model.wv[word] for word in words if word in model.wv]  # Extract valid word vectors
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)  # Compute average embedding of valid words
    else:
        return np.zeros(vector_size)  # Return a zero vector if no valid words exist

# Step 4
# Feature engineering with normalization
# Combines TF-IDF, Word2Vec, and text length into a single feature matrix
class FeatureCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf, selector, word2vec_model, vector_size, pca_components=50):
        self.tfidf = tfidf  # TF-IDF feature extractor
        self.selector = selector  # Feature selector (Chi-Square test)
        self.word2vec_model = word2vec_model  # Trained Word2Vec model
        self.vector_size = vector_size  # Word2Vec embedding dimension
        self.pca_components = pca_components  # PCA dimension
        self.pca = PCA(n_components=pca_components, random_state=0)  # PCA for dimensionality reduction
        self.scaler = StandardScaler()  # Scaler for feature normalization

    # Fit method: Prepares TF-IDF, PCA, and other tools
    def fit(self, X, y=None):
        # Extract TF-IDF features and perform feature selection
        X_tfidf = self.tfidf.fit_transform(X)
        X_tfidf_selected = self.selector.fit_transform(X_tfidf, y)
        X_tfidf_selected_dense = X_tfidf_selected.toarray()  # Convert sparse matrix to dense

        # Compute average Word2Vec embeddings for each document
        word2vec_features = np.array(
            [get_avg_word2vec(preprocess(text), self.word2vec_model, self.vector_size) for text in X]
        )

        # Compute text lengths as numerical features
        text_lengths = X.apply(len).values.reshape(-1, 1) * 10

        # Combine all features into a single matrix
        combined_features = np.hstack([X_tfidf_selected_dense, word2vec_features, text_lengths])

        # Normalize the combined features
        normalized_features = self.scaler.fit_transform(combined_features)

        # Fit PCA for dimensionality reduction
        self.pca.fit(normalized_features)
        return self

    # Transform method: Applies the same feature engineering steps to new data
    def transform(self, X):
        X_tfidf = self.tfidf.transform(X)
        X_tfidf_selected = self.selector.transform(X_tfidf)
        X_tfidf_selected_dense = X_tfidf_selected.toarray()

        word2vec_features = np.array(
            [get_avg_word2vec(preprocess(text), self.word2vec_model, self.vector_size) for text in X]
        )
        text_lengths = X.apply(len).values.reshape(-1, 1) * 10

        combined_features = np.hstack([X_tfidf_selected_dense, word2vec_features, text_lengths])

        # Normalize the combined features
        normalized_features = self.scaler.transform(combined_features)

        # Apply PCA transformation
        reduced_features = self.pca.transform(normalized_features)
        return reduced_features

# Step 5
# Create pipeline（pipeline can simplify the process of building a model）
# Integrates feature engineering and model training
tfidf = TfidfVectorizer(max_features=1000)  # Extract up to 1000 features
selector = SelectKBest(chi2, k=500)  # Select top 500 features using Chi-Square test
feature_combiner = FeatureCombiner(tfidf, selector, word2vec_model, vector_size=100, pca_components=50)
model = LogisticRegression(solver='newton-cg', max_iter=500)  # Logistic regression model
pipeline = Pipeline([
    ('features', feature_combiner),
    ('classifier', model)
])

# Step 6
# Display 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)
precision_scores, recall_scores, f1_scores, accuracy_scores = [], [], [], []

print("Cross-Validation Results:")

for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    pipeline.fit(X_train, y_train)  # Train the pipeline
    y_pred = pipeline.predict(X_test)  # Predict on the test set

    # Calculate performance metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    accuracy_scores.append(accuracy)

    # Print loop result
    print("\nFold {} cross-validation result:".format(fold))
    print("\tAccuracy: {:.2f}".format(accuracy))
    print("\tMacro Precision: {:.2f}".format(precision))
    print("\tMacro Recall: {:.2f}".format(recall))
    print("\tMacro F1-Score: {:.2f}".format(f1))
    fold += 1

# Print average result
print("\nAverage cross-validation results:")
print("\tAverage Accuracy: {:.2f}".format(np.mean(accuracy_scores)))
print("\tAverage Macro Precision: {:.2f}".format(np.mean(precision_scores)))
print("\tAverage Macro Recall: {:.2f}".format(np.mean(recall_scores)))
print("\tAverage Macro F1-Score: {:.2f}".format(np.mean(f1_scores)))
