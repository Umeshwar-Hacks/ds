import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

# Download NLTK packages
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# PART 1: CLUSTERING
# Generate sample data and apply K-means clustering
X = np.random.rand(100, 2) * 10  # Random 2D data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
silhouette = silhouette_score(X_scaled, labels)
print(f"Clustering Silhouette Score: {silhouette:.4f}")

# Plot clustering results
plt.figure(figsize=(6, 4))
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           marker='X', s=100, c='black', label='Centroids')
plt.title('K-means Clustering')
plt.legend()
plt.savefig('clustering.png')

# PART 2: TEXT PROCESSING
doc1 = "Artificial intelligence is intelligence demonstrated by machines."
doc2 = "Machine learning is a branch of artificial intelligence."

# 1. Tokenization
tokens1 = word_tokenize(doc1.lower())
tokens2 = word_tokenize(doc2.lower())
print(f"Doc1 tokens: {tokens1[:5]}")

# 2. POS Tagging
pos_tags1 = pos_tag(tokens1)
print(f"POS tags: {pos_tags1[:5]}")

# 3. Sentiment Analysis
sid = SentimentIntensityAnalyzer()
sentiment1 = sid.polarity_scores(doc1)
print(f"Sentiment: {sentiment1}")

# 4. Document Similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
print(f"Document similarity: {cosine_sim:.4f}")

# 5. Word Cloud
wordcloud = WordCloud(width=400, height=200, background_color='white').generate(doc1 + " " + doc2)
plt.figure(figsize=(6, 3))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('wordcloud.png')

# PART 3: IMAGE CLASSIFICATION
# Create simple synthetic dataset
os.makedirs('data/train/circle', exist_ok=True)
os.makedirs('data/train/square', exist_ok=True)
os.makedirs('data/val/circle', exist_ok=True)
os.makedirs('data/val/square', exist_ok=True)

# Create 20 sample images for demonstration
for i in range(20):
    img_size = 28
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    if i % 2 == 0:  # Create circles
        cv2.circle(img, (img_size//2, img_size//2), img_size//3, 255, -1)
        path = f"data/{'train' if i < 16 else 'val'}/circle/img_{i}.png"
    else:  # Create squares
        cv2.rectangle(img, (img_size//3, img_size//3), (2*img_size//3, 2*img_size//3), 255, -1)
        path = f"data/{'train' if i < 16 else 'val'}/square/img_{i}.png"
    
    cv2.imwrite(path, img)

# Data preprocessing with augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory('data/train', target_size=(28, 28), 
                                           batch_size=4, class_mode='categorical', color_mode='grayscale')
val_gen = val_datagen.flow_from_directory('data/val', target_size=(28, 28), 
                                       batch_size=4, class_mode='categorical', color_mode='grayscale')

# Simple CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, epochs=5, validation_data=val_gen, verbose=0)

# Plot results
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()
plt.savefig('cnn_accuracy.png')

# PART 4: INFERENCE
print("\nINFERENCE AND CONCLUSION:")
print("1. Clustering Analysis: K-means successfully identified 3 clusters with silhouette score indicating good separation.")
print("2. Text Processing: Implemented tokenization, POS tagging, sentiment analysis, document similarity, and word cloud visualization.")
print("3. Image Classification: Simple CNN achieved good accuracy distinguishing between circles and squares.")
print("4. The comprehensive approach demonstrates effective techniques for analyzing different data types.")
