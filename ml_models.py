"""
Machine Learning Models for Cryptanalysis
This module contains ML models for cipher classification and cryptanalysis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from collections import Counter
import math
import re

class CipherClassifier:
    """Machine learning classifier for different cipher types"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 3),
            max_features=1000
        )
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'nb': MultinomialNB()
        }
        self.best_model = None

    def extract_features(self, text):
        """Extract comprehensive features from text for classification"""
        features = {}

        # Basic text statistics
        text_clean = re.sub(r'[^a-zA-Z]', '', text.lower())
        features['length'] = len(text)
        features['unique_chars'] = len(set(text_clean))
        features['entropy'] = self.calculate_entropy(text_clean)

        # Character frequency features
        freq = Counter(text_clean)
        total = len(text_clean)
        if total > 0:
            for char in 'abcdefghijklmnopqrstuvwxyz':
                features[f'freq_{char}'] = freq[char] / total
        else:
            for char in 'abcdefghijklmnopqrstuvwxyz':
                features[f'freq_{char}'] = 0

        # Pattern features
        features['has_numbers'] = int(bool(re.search(r'\d', text)))
        features['has_special'] = int(bool(re.search(r'[^a-zA-Z0-9\s]', text)))
        features['is_uppercase'] = int(text.isupper())
        features['is_lowercase'] = int(text.islower())

        # N-gram features
        features.update(self.extract_ngram_features(text_clean))

        return features

    def extract_ngram_features(self, text, n=3):
        """Extract n-gram frequency features"""
        features = {}
        if len(text) < n:
            for i in range(26**n):
                features[f'ngram_{i}'] = 0
            return features

        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        ngram_freq = Counter(ngrams)

        # Create feature vector for all possible n-grams (limited to most common)
        most_common = ngram_freq.most_common(100)
        for ngram, count in most_common:
            features[f'ngram_{ngram}'] = count / len(ngrams)

        return features

    def calculate_entropy(self, text):
        """Calculate Shannon entropy"""
        if not text:
            return 0

        freq = Counter(text)
        entropy = 0
        for count in freq.values():
            probability = count / len(text)
            entropy -= probability * math.log2(probability)
        return entropy

    def prepare_dataset(self):
        """Prepare training dataset with various cipher types"""
        data = []
        labels = []

        # Generate Caesar cipher examples
        plaintexts = [
            "the quick brown fox jumps over the lazy dog",
            "cryptography is the practice and study of techniques",
            "data science combines statistics computer science",
            "machine learning algorithms can solve complex problems",
            "artificial intelligence is transforming technology"
        ]

        for text in plaintexts:
            # Original text (plaintext)
            data.append(self.extract_features(text))
            labels.append('plaintext')

            # Caesar cipher variations
            for shift in [3, 5, 7, 10, 13]:
                cipher_text = self.caesar_cipher(text, shift)
                data.append(self.extract_features(cipher_text))
                labels.append('caesar')

        # Generate Vigenère cipher examples
        keys = ['KEY', 'SECRET', 'CRYPTO', 'ALGORITHM']
        for text in plaintexts:
            for key in keys:
                cipher_text = self.vigenere_cipher(text, key)
                data.append(self.extract_features(cipher_text))
                labels.append('vigenere')

        # Generate Base64 examples
        for text in plaintexts:
            import base64
            cipher_text = base64.b64encode(text.encode()).decode()
            data.append(self.extract_features(cipher_text))
            labels.append('base64')

        # Generate random text examples
        import random
        import string
        for _ in range(50):
            random_text = ''.join(random.choices(string.ascii_letters, k=100))
            data.append(self.extract_features(random_text))
            labels.append('random')

        return pd.DataFrame(data), labels

    def caesar_cipher(self, text, shift):
        """Simple Caesar cipher implementation"""
        result = ""
        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                result += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
            else:
                result += char
        return result

    def vigenere_cipher(self, text, key):
        """Simple Vigenère cipher implementation"""
        result = ""
        key = key.upper()
        key_index = 0

        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                key_char = key[key_index % len(key)]
                key_shift = ord(key_char) - 65

                result += chr((ord(char) - ascii_offset + key_shift) % 26 + ascii_offset)
                key_index += 1
            else:
                result += char
        return result

    def train(self):
        """Train the classifier models"""
        print("Preparing dataset...")
        X_df, y = self.prepare_dataset()

        # Convert to numpy arrays
        feature_names = list(X_df.columns)
        X = X_df.values

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training models...")
        best_accuracy = 0
        best_model_name = None

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"{name.upper()} Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                self.best_model = model

        print(f"\nBest model: {best_model_name.upper()} with accuracy: {best_accuracy:.4f}")

        # Save the best model
        if self.best_model:
            joblib.dump(self.best_model, 'cipher_classifier.pkl')
            joblib.dump(feature_names, 'feature_names.pkl')
            print("Model saved successfully!")

        return best_accuracy

    def predict(self, text):
        """Predict cipher type for given text"""
        if not self.best_model:
            # Try to load saved model
            try:
                self.best_model = joblib.load('cipher_classifier.pkl')
                feature_names = joblib.load('feature_names.pkl')
            except:
                return {'error': 'Model not trained. Please train the model first.'}

        features = self.extract_features(text)
        feature_vector = np.array([features[name] for name in feature_names])

        prediction = self.best_model.predict([feature_vector])[0]
        probabilities = self.best_model.predict_proba([feature_vector])[0]

        class_probabilities = {}
        for i, class_name in enumerate(self.best_model.classes_):
            class_probabilities[class_name] = probabilities[i]

        return {
            'prediction': prediction,
            'confidence': max(probabilities),
            'probabilities': class_probabilities
        }

class NeuralCryptanalyzer:
    """Neural network-based cryptanalysis tool"""

    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 5),
            max_features=2000
        )

    def create_training_data(self, size=1000):
        """Create synthetic training data for neural cryptanalysis"""
        data = []
        labels = []

        # Generate various types of encrypted and plain text
        for _ in range(size):
            # Plain text
            plain = self.generate_plaintext()
            data.append(plain)
            labels.append(0)  # 0 = plaintext

            # Caesar encrypted
            caesar = self.caesar_cipher(plain, np.random.randint(1, 25))
            data.append(caesar)
            labels.append(1)  # 1 = caesar

            # Vigenère encrypted
            key = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))
            vigenere = self.vigenere_cipher(plain, key)
            data.append(vigenere)
            labels.append(2)  # 2 = vigenere

        return data, labels

    def generate_plaintext(self, length=100):
        """Generate realistic plaintext"""
        words = [
            'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
            'cryptography', 'security', 'encryption', 'decryption', 'algorithm',
            'data', 'science', 'machine', 'learning', 'artificial', 'intelligence'
        ]

        text = []
        while len(' '.join(text)) < length:
            text.append(np.random.choice(words))

        return ' '.join(text)[:length]

    def caesar_cipher(self, text, shift):
        """Caesar cipher"""
        result = ""
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - base + shift) % 26 + base)
            else:
                result += char
        return result

    def vigenere_cipher(self, text, key):
        """Vigenère cipher"""
        result = ""
        key_index = 0
        key = key.upper()

        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                key_char = key[key_index % len(key)]
                shift = ord(key_char) - ord('A')
                result += chr((ord(char) - base + shift) % 26 + base)
                key_index += 1
            else:
                result += char
        return result

    def build_model(self):
        """Build neural network model (simplified version)"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            self.tokenizer = Tokenizer(char_level=True)
            self.model = Sequential([
                Embedding(1000, 64, input_length=200),
                LSTM(128, return_sequences=True),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')  # 3 classes: plaintext, caesar, vigenere
            ])

            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            return True
        except ImportError:
            print("TensorFlow not available. Neural network features disabled.")
            return False

    def train_neural_model(self, epochs=10):
        """Train the neural network"""
        if not self.model:
            if not self.build_model():
                return False

        print("Generating training data...")
        texts, labels = self.create_training_data(2000)

        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=200)
        y = np.array(labels)

        print("Training neural network...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        # Save model
        try:
            self.model.save('neural_cryptanalyzer.h5')
            print("Neural model saved!")
        except:
            print("Could not save model")

        return history

    def analyze_text(self, text):
        """Analyze text using neural network"""
        if not self.model:
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model('neural_cryptanalyzer.h5')
            except:
                return {'error': 'Neural model not available'}

        try:
            sequence = self.tokenizer.texts_to_sequences([text])
            X = pad_sequences(sequence, maxlen=200)
            prediction = self.model.predict(X)[0]

            classes = ['plaintext', 'caesar', 'vigenere']
            predicted_class = classes[np.argmax(prediction)]
            confidence = float(np.max(prediction))

            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    classes[i]: float(prediction[i]) for i in range(len(classes))
                }
            }
        except:
            return {'error': 'Analysis failed'}

class StatisticalAnalyzer:
    """Advanced statistical analysis for cryptanalysis"""

    def __init__(self):
        self.ngram_models = {}

    def build_ngram_model(self, text, n=2):
        """Build n-gram frequency model"""
        text = re.sub(r'[^a-z]', '', text.lower())
        ngrams = {}

        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            ngrams[ngram] = ngrams.get(ngram, 0) + 1

        # Convert to probabilities
        total = sum(ngrams.values())
        for ngram in ngrams:
            ngrams[ngram] /= total

        return ngrams

    def calculate_chi_squared(self, text, expected_freq=None):
        """Calculate chi-squared statistic for goodness of fit"""
        if expected_freq is None:
            # English letter frequencies
            expected_freq = {
                'a': 0.08167, 'b': 0.01492, 'c': 0.02782, 'd': 0.04253,
                'e': 0.12702, 'f': 0.02228, 'g': 0.02015, 'h': 0.06094,
                'i': 0.06966, 'j': 0.00153, 'k': 0.00772, 'l': 0.04025,
                'm': 0.02406, 'n': 0.06749, 'o': 0.07507, 'p': 0.01929,
                'q': 0.00095, 'r': 0.05987, 's': 0.06327, 't': 0.09056,
                'u': 0.02758, 'v': 0.00978, 'w': 0.02360, 'x': 0.00150,
                'y': 0.01974, 'z': 0.00074
            }

        text = re.sub(r'[^a-z]', '', text.lower())
        if not text:
            return float('inf')

        observed = Counter(text)
        total = len(text)
        chi_squared = 0

        for letter, expected_prob in expected_freq.items():
            expected_count = expected_prob * total
            observed_count = observed.get(letter, 0)
            if expected_count > 0:
                chi_squared += (observed_count - expected_count) ** 2 / expected_count

        return chi_squared

    def detect_caesar_key(self, text, max_key=25):
        """Try to detect Caesar cipher key using chi-squared"""
        best_key = 0
        best_score = float('inf')

        for key in range(max_key + 1):
            decrypted = self.caesar_cipher(text, -key)
            score = self.calculate_chi_squared(decrypted)
            if score < best_score:
                best_score = score
                best_key = key

        return best_key, best_score

    def caesar_cipher(self, text, shift):
        """Caesar cipher with shift"""
        result = ""
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - base + shift) % 26 + base)
            else:
                result += char
        return result

    def kasiski_examination(self, text, min_distance=3, max_distance=20):
        """Kasiski examination for detecting Vigenère key length"""
        text = re.sub(r'[^a-z]', '', text.lower())
        repeated_sequences = {}

        # Find repeated sequences
        for length in range(3, 8):  # Check sequences of length 3-7
            for i in range(len(text) - length + 1):
                seq = text[i:i+length]
                if seq in repeated_sequences:
                    repeated_sequences[seq].append(i)
                else:
                    repeated_sequences[seq] = [i]

        # Calculate distances between repetitions
        distances = []
        for seq, positions in repeated_sequences.items():
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    dist = positions[i+1] - positions[i]
                    if min_distance <= dist <= max_distance:
                        distances.append(dist)

        if not distances:
            return []

        # Find greatest common divisors
        from math import gcd
        from functools import reduce

        gcds = []
        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                gcds.append(gcd(distances[i], distances[j]))

        # Count most common GCDs (potential key lengths)
        gcd_counts = Counter(gcds)
        likely_key_lengths = [gcd for gcd, count in gcd_counts.most_common(5)]

        return likely_key_lengths

    def index_of_coincidence(self, text):
        """Calculate Index of Coincidence"""
        text = re.sub(r'[^a-z]', '', text.lower())
        if not text:
            return 0

        n = len(text)
        freq = Counter(text)

        ic = 0
        for count in freq.values():
            ic += count * (count - 1)

        ic = ic / (n * (n - 1)) * 26
        return ic

    def analyze_text(self, text):
        """Comprehensive statistical analysis"""
        analysis = {}

        # Basic statistics
        analysis['length'] = len(text)
        analysis['entropy'] = self.calculate_entropy(text)
        analysis['chi_squared'] = self.calculate_chi_squared(text)
        analysis['index_of_coincidence'] = self.index_of_coincidence(text)

        # Caesar key detection
        caesar_key, caesar_score = self.detect_caesar_key(text)
        analysis['likely_caesar_key'] = caesar_key
        analysis['caesar_confidence'] = 1 / (1 + caesar_score)  # Convert to confidence score

        # Vigenère analysis
        likely_key_lengths = self.kasiski_examination(text)
        analysis['likely_vigenere_lengths'] = likely_key_lengths[:3]

        # Classification based on statistics
        if analysis['entropy'] > 4.5:
            analysis['likely_type'] = 'random_or_otp'
            analysis['confidence'] = 0.8
        elif analysis['chi_squared'] < 100:
            analysis['likely_type'] = 'caesar_or_simple_substitution'
            analysis['confidence'] = 0.7
        elif analysis['index_of_coincidence'] > 1.7:
            analysis['likely_type'] = 'polyalphabetic'
            analysis['confidence'] = 0.6
        else:
            analysis['likely_type'] = 'unknown'
            analysis['confidence'] = 0.3

        return analysis

    def calculate_entropy(self, text):
        """Calculate Shannon entropy"""
        text = re.sub(r'[^a-z]', '', text.lower())
        if not text:
            return 0

        freq = Counter(text)
        entropy = 0
        for count in freq.values():
            probability = count / len(text)
            entropy -= probability * math.log2(probability)
        return entropy
