# Libraries for text processing, feature extraction, dimensionality reduction, model training, and evaluation
from sklearn.decomposition import PCA  # Principal Component Analysis (PCA) for dimensionality reduction
from sklearn.model_selection import train_test_split  # To split dataset into training, development, and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer  # To extract TF-IDF features
from sklearn.feature_selection import SelectKBest, chi2  # For feature selection using chi-squared test
from sklearn.preprocessing import StandardScaler  # For standardizing features
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score  # Metrics for model evaluation
import pandas as pd  # For data handling
import numpy as np  # For numerical computations
import re  # For text preprocessing using regular expressions
from nltk.corpus import stopwords  # To remove stopwords
from nltk.tokenize import word_tokenize  # To tokenize text into words
from gensim.models import Word2Vec  # To train word embedding model
from sklearn.preprocessing import PolynomialFeatures  # To create polynomial features
import matplotlib.pyplot as plt  # To plot graphs

# Load dataset
# Load a dataset that contains two columns: 'text' (news content) and 'category' (labels)
dataset = pd.read_csv('bbc-text.csv')
X = dataset['text']  # Text content
y = dataset['category']  # Category labels

# Text preprocessing
# Define a function to preprocess text by cleaning and tokenizing
def preprocess(text):
    stop_words = set(stopwords.words('english'))  # Load English stopwords
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    words = word_tokenize(text)  # Tokenize text into words
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return words

# Apply preprocessing to all text data
X_processed = X.apply(preprocess)

# Train Word2Vec model
# Train a Word2Vec model on the preprocessed text to generate word embeddings
word2vec_model = Word2Vec(
    sentences=X_processed,  # Input tokenized sentences
    vector_size=300,        # Size of word embeddings (300-dimensional)
    window=5,               # Context window size (5 words before and after)
    min_count=1,            # Minimum word frequency to include in vocabulary
    workers=4               # Number of threads to use for training
)

# Define a function to compute the average word embedding for a document
def get_avg_word2vec(words, model, vector_size):
    vectors = [model.wv[word] for word in words if word in model.wv]  # Retrieve vectors for valid words
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)  # Compute mean of word vectors
    else:
        return np.zeros(vector_size)  # Return zero vector if no valid words are present

# Feature combiner class
# Define a class to combine TF-IDF, word embeddings, and text length features
class FeatureCombiner:
    def __init__(self, tfidf, selector, word2vec_model, vector_size, pca_components):
        self.tfidf = tfidf  # TF-IDF vectorizer
        self.selector = selector  # Feature selector using chi-squared test
        self.word2vec_model = word2vec_model  # Word2Vec model for embeddings
        self.vector_size = vector_size  # Size of word embedding vectors
        self.pca_components = pca_components  # Number of PCA components for dimensionality reduction
        self.pca = PCA(n_components=pca_components, random_state=0)  # PCA object
        self.scaler = StandardScaler()  # StandardScaler for normalization

    def expand_text_length_features(self, text_lengths):
        """
        Expand text length into polynomial features with size 300.
        """
        poly = PolynomialFeatures(degree=2, include_bias=False)  # Create polynomial features
        expanded_lengths = poly.fit_transform(text_lengths.reshape(-1, 1))  # Apply polynomial expansion
        return expanded_lengths[:, :300]  # Ensure the expanded features have 300 dimensions

    def fit_transform(self, X, y):
        """
        Extract features, normalize them, and apply PCA for dimensionality reduction.
        """
        # Compute TF-IDF features and select top ones using chi-squared test
        X_tfidf = self.tfidf.fit_transform(X)
        X_tfidf_selected = self.selector.fit_transform(X_tfidf, y)
        X_tfidf_selected_dense = X_tfidf_selected.toarray()  # Convert sparse matrix to dense

        # Compute Word2Vec features
        word2vec_features = np.array(
            [get_avg_word2vec(preprocess(text), self.word2vec_model, self.vector_size) for text in X]
        )

        # Expand text lengths into polynomial features
        text_lengths = X.apply(len).values.reshape(-1, 1)
        expanded_length_features = self.expand_text_length_features(text_lengths)

        # Combine all features
        combined_features = np.hstack([X_tfidf_selected_dense, word2vec_features, expanded_length_features])
        normalized_features = self.scaler.fit_transform(combined_features)  # Normalize features
        return self.pca.fit_transform(normalized_features)  # Apply PCA for dimensionality reduction

    def transform(self, X):
        """
        Transform new data using the same process as fit_transform.
        """
        X_tfidf = self.tfidf.transform(X)
        X_tfidf_selected = self.selector.transform(X_tfidf)
        X_tfidf_selected_dense = X_tfidf_selected.toarray()

        word2vec_features = np.array(
            [get_avg_word2vec(preprocess(text), self.word2vec_model, self.vector_size) for text in X]
        )

        text_lengths = X.apply(len).values.reshape(-1, 1)
        expanded_length_features = self.expand_text_length_features(text_lengths)

        combined_features = np.hstack([X_tfidf_selected_dense, word2vec_features, expanded_length_features])
        normalized_features = self.scaler.transform(combined_features)  # Normalize features
        return self.pca.transform(normalized_features)  # Apply PCA

# Split dataset into training, development, and testing sets
# Split the dataset into 70% training, 15% development, and 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
X_develop, X_test, y_develop, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Step 6: Adjust PCA dimensions using the development set
pca_dimensions = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]  # PCA dimensions to evaluate
best_f1 = 0  # Best F1-score found
best_pca_dim = None  # Corresponding best PCA dimension
pca_f1_scores = []  # Store PCA dimensions and their corresponding F1-scores

for pca_dim in pca_dimensions:
    # Create a feature combiner with the current PCA dimension
    tfidf = TfidfVectorizer(max_features=1000)  # Extract TF-IDF features
    selector = SelectKBest(chi2, k=300)  # Select top 300 features
    feature_combiner = FeatureCombiner(tfidf, selector, word2vec_model, vector_size=300, pca_components=pca_dim)

    # Transform training and development sets
    X_train_transformed = feature_combiner.fit_transform(X_train, y_train)
    X_develop_transformed = feature_combiner.transform(X_develop)

    # Train and evaluate logistic regression model
    model = LogisticRegression(solver='newton-cg', max_iter=500)  # Use 'newton-cg' solver
    model.fit(X_train_transformed, y_train)
    y_develop_pred = model.predict(X_develop_transformed)

    # Evaluate F1-score on the development set
    _, _, develop_f1, _ = precision_recall_fscore_support(y_develop, y_develop_pred, average='macro')
    print(f"PCA Dimension: {pca_dim}, Development F1-Score: {develop_f1:.5f}")

    # Store PCA dimension and F1-score for plotting
    pca_f1_scores.append((pca_dim, develop_f1))

    # Update the best PCA dimension if a better F1-score is found
    if develop_f1 > best_f1:
        best_f1 = develop_f1
        best_pca_dim = pca_dim

print("\nBest PCA Dimension: {}, Best Development set F1-Score: {:.5f}".format(best_pca_dim, best_f1))

# Plot PCA dimensions vs. Development F1-Scores
pca_dims, f1_scores = zip(*pca_f1_scores)  # Unpack PCA dimensions and F1-scores
plt.figure(figsize=(10, 6))
plt.plot(pca_dims, f1_scores, marker='o', linestyle='-', color='b')
plt.title('F1-Scores of development set in different PCA Dimensions')
plt.xlabel('PCA Dimensions')
plt.ylabel('Development set F1-Score')
plt.grid(True)
plt.show()

# Final evaluation on test set using the best PCA dimension
feature_combiner = FeatureCombiner(tfidf, selector, word2vec_model, vector_size=300, pca_components=best_pca_dim)
X_train_transformed = feature_combiner.fit_transform(X_train, y_train)
X_test_transformed = feature_combiner.transform(X_test)

model = LogisticRegression(solver='newton-cg', max_iter=500)
model.fit(X_train_transformed, y_train)
y_test_pred = model.predict(X_test_transformed)

test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro')
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print final test set results
print("\nTest Set Results with Best PCA Dimension:")
print("\tAccuracy: {:.2f}".format(test_accuracy))
print("\tMacro Precision: {:.2f}".format(test_precision))
print("\tMacro Recall: {:.2f}".format(test_recall))
print("\tMacro F1-Score: {:.2f}".format(test_f1))
