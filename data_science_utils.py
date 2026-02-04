"""
Data Science Utilities for Cryptanalysis
This module provides advanced data science tools for analyzing encrypted texts
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import networkx as nx
from collections import Counter, defaultdict
import re
import math
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class TextAnalyzer:
    """Advanced text analysis for cryptanalysis"""

    def __init__(self):
        self.english_freq = {
            'a': 0.08167, 'b': 0.01492, 'c': 0.02782, 'd': 0.04253,
            'e': 0.12702, 'f': 0.02228, 'g': 0.02015, 'h': 0.06094,
            'i': 0.06966, 'j': 0.00153, 'k': 0.00772, 'l': 0.04025,
            'm': 0.02406, 'n': 0.06749, 'o': 0.07507, 'p': 0.01929,
            'q': 0.00095, 'r': 0.05987, 's': 0.06327, 't': 0.09056,
            'u': 0.02758, 'v': 0.00978, 'w': 0.02360, 'x': 0.00150,
            'y': 0.01974, 'z': 0.00074
        }

    def analyze_text_properties(self, text):
        """Comprehensive text property analysis"""
        properties = {}

        # Basic properties
        properties['length'] = len(text)
        properties['unique_chars'] = len(set(text))
        properties['alphanumeric_ratio'] = len(re.findall(r'[a-zA-Z0-9]', text)) / len(text) if text else 0

        # Character distribution
        char_freq = self.calculate_frequency(text)
        properties['char_distribution'] = char_freq

        # Statistical measures
        properties['entropy'] = self.calculate_entropy(text)
        properties['chi_squared'] = self.calculate_chi_squared(text)
        properties['index_of_coincidence'] = self.calculate_ic(text)

        # Pattern analysis
        properties['repeated_patterns'] = self.find_repeated_patterns(text)
        properties['ngram_analysis'] = self.analyze_ngrams(text)

        # Structural analysis
        properties['word_lengths'] = self.analyze_word_lengths(text)
        properties['sentence_structure'] = self.analyze_sentence_structure(text)

        return properties

    def calculate_frequency(self, text):
        """Calculate character frequency distribution"""
        text = re.sub(r'[^a-zA-Z]', '', text.lower())
        if not text:
            return {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz'}

        freq = Counter(text)
        total = len(text)

        return {char: freq.get(char, 0) / total for char in 'abcdefghijklmnopqrstuvwxyz'}

    def calculate_entropy(self, text):
        """Calculate Shannon entropy"""
        text = re.sub(r'[^a-z]', '', text.lower())
        if not text:
            return 0

        freq = Counter(text)
        entropy = 0
        for count in freq.values():
            p = count / len(text)
            entropy -= p * math.log2(p)
        return entropy

    def calculate_chi_squared(self, text):
        """Calculate chi-squared statistic"""
        freq = self.calculate_frequency(text)
        chi_squared = 0

        for char, observed in freq.items():
            expected = self.english_freq.get(char, 0)
            if expected > 0:
                chi_squared += (observed - expected) ** 2 / expected

        return chi_squared

    def calculate_ic(self, text):
        """Calculate Index of Coincidence"""
        text = re.sub(r'[^a-z]', '', text.lower())
        if not text:
            return 0

        n = len(text)
        freq = Counter(text)

        ic = 0
        for count in freq.values():
            ic += count * (count - 1)

        return ic / (n * (n - 1)) * 26 if n > 1 else 0

    def find_repeated_patterns(self, text, min_length=3, max_length=8):
        """Find repeated patterns in text"""
        text = re.sub(r'[^a-z]', '', text.lower())
        patterns = defaultdict(list)

        for length in range(min_length, min(max_length + 1, len(text))):
            for i in range(len(text) - length + 1):
                pattern = text[i:i+length]
                patterns[pattern].append(i)

        # Filter patterns that appear more than once
        repeated = {p: positions for p, positions in patterns.items() if len(positions) > 1}

        # Sort by frequency and pattern length
        sorted_patterns = sorted(repeated.items(),
                               key=lambda x: (len(x[1]), len(x[0])),
                               reverse=True)

        return sorted_patterns[:20]  # Return top 20 patterns

    def analyze_ngrams(self, text, n=2):
        """Analyze n-gram frequencies"""
        text = re.sub(r'[^a-z]', '', text.lower())
        ngrams = defaultdict(int)

        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            ngrams[ngram] += 1

        total = sum(ngrams.values())
        ngram_freq = {ng: count/total for ng, count in ngrams.items()}

        # Return top 20 most frequent n-grams
        sorted_ngrams = sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_ngrams[:20]

    def analyze_word_lengths(self, text):
        """Analyze word length distribution"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {'average': 0, 'distribution': {}, 'most_common': []}

        lengths = [len(word) for word in words]
        length_dist = Counter(lengths)

        return {
            'average': np.mean(lengths),
            'median': np.median(lengths),
            'distribution': dict(length_dist),
            'most_common': length_dist.most_common(5)
        }

    def analyze_sentence_structure(self, text):
        """Analyze sentence structure patterns"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {'sentence_count': 0, 'avg_length': 0}

        lengths = [len(s.split()) for s in sentences]

        return {
            'sentence_count': len(sentences),
            'avg_words_per_sentence': np.mean(lengths),
            'sentence_length_distribution': Counter(lengths)
        }

class ClusteringAnalyzer:
    """Clustering analysis for text classification"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def create_feature_matrix(self, texts):
        """Create feature matrix from texts"""
        features = []

        analyzer = TextAnalyzer()

        for text in texts:
            props = analyzer.analyze_text_properties(text)

            # Extract numerical features
            feature_vector = [
                props['length'],
                props['unique_chars'],
                props['alphanumeric_ratio'],
                props['entropy'],
                props['chi_squared'],
                props['index_of_coincidence']
            ]

            # Add character frequencies
            for char in 'abcdefghijklmnopqrstuvwxyz':
                feature_vector.append(props['char_distribution'][char])

            features.append(feature_vector)

        return np.array(features)

    def perform_clustering(self, texts, method='kmeans', n_clusters=3):
        """Perform clustering analysis"""
        X = self.create_feature_matrix(texts)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Reduce dimensionality for visualization
        X_pca = self.pca.fit_transform(X_scaled)

        if method == 'kmeans':
            clusters = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            clusters = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise ValueError("Unsupported clustering method")

        labels = clusters.fit_predict(X_scaled)

        # Calculate silhouette score if possible
        if len(set(labels)) > 1 and -1 not in labels:  # DBSCAN can have noise points
            silhouette = silhouette_score(X_scaled, labels)
        else:
            silhouette = None

        return {
            'labels': labels,
            'pca_components': X_pca,
            'silhouette_score': silhouette,
            'cluster_centers': clusters.cluster_centers_ if hasattr(clusters, 'cluster_centers_') else None
        }

    def analyze_clusters(self, texts, labels):
        """Analyze characteristics of each cluster"""
        cluster_analysis = defaultdict(list)

        analyzer = TextAnalyzer()

        for text, label in zip(texts, labels):
            if label != -1:  # Skip noise points in DBSCAN
                props = analyzer.analyze_text_properties(text)
                cluster_analysis[label].append(props)

        cluster_summary = {}
        for cluster_id, properties in cluster_analysis.items():
            if properties:
                # Calculate average properties for the cluster
                avg_props = {}
                for prop_name in properties[0].keys():
                    if isinstance(properties[0][prop_name], (int, float)):
                        values = [p[prop_name] for p in properties]
                        avg_props[prop_name] = np.mean(values)
                    else:
                        avg_props[prop_name] = properties[0][prop_name]  # Keep first value for non-numeric

                cluster_summary[cluster_id] = {
                    'size': len(properties),
                    'average_properties': avg_props,
                    'entropy_range': (min(p['entropy'] for p in properties),
                                    max(p['entropy'] for p in properties))
                }

        return cluster_summary

class NetworkAnalyzer:
    """Network analysis for pattern relationships"""

    def __init__(self):
        self.graph = nx.Graph()

    def build_pattern_network(self, text, ngram_size=3):
        """Build network of pattern relationships"""
        text = re.sub(r'[^a-z]', '', text.lower())

        # Create nodes (unique n-grams)
        ngrams = set()
        for i in range(len(text) - ngram_size + 1):
            ngram = text[i:i+ngram_size]
            ngrams.add(ngram)

        # Create edges based on adjacency
        edges = []
        ngram_list = list(ngrams)

        for i in range(len(text) - ngram_size):
            current = text[i:i+ngram_size]
            next_ngram = text[i+1:i+1+ngram_size]
            if next_ngram:
                edges.append((current, next_ngram))

        # Build network
        self.graph.add_nodes_from(ngram_list)
        self.graph.add_edges_from(edges)

        return self.graph

    def analyze_network_properties(self):
        """Analyze network properties"""
        if not self.graph:
            return {}

        properties = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }

        # Calculate degree distribution
        degrees = [d for n, d in self.graph.degree()]
        properties['avg_degree'] = np.mean(degrees)
        properties['max_degree'] = max(degrees)
        properties['degree_distribution'] = Counter(degrees)

        # Find central nodes
        if properties['num_nodes'] > 0:
            centrality = nx.degree_centrality(self.graph)
            properties['most_central_nodes'] = sorted(centrality.items(),
                                                    key=lambda x: x[1],
                                                    reverse=True)[:10]

        return properties

    def find_communities(self):
        """Find communities in the pattern network"""
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(self.graph))
            return {
                'num_communities': len(communities),
                'communities': [list(c) for c in communities],
                'modularity': community.modularity(self.graph, communities)
            }
        except:
            return {'error': 'Community detection not available'}

class StatisticalVisualizer:
    """Visualization tools for statistical analysis"""

    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_frequency_distribution(self, text, title="Character Frequency Distribution"):
        """Plot character frequency distribution"""
        analyzer = TextAnalyzer()
        freq = analyzer.calculate_frequency(text)

        plt.figure(figsize=(12, 6))
        chars = list(freq.keys())
        frequencies = list(freq.values())

        bars = plt.bar(chars, frequencies, alpha=0.7)
        plt.title(title)
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)

        # Add expected English frequencies for comparison
        expected = [analyzer.english_freq.get(char, 0) for char in chars]
        plt.plot(chars, expected, 'r--', label='English Average', linewidth=2)
        plt.legend()

        plt.tight_layout()
        return plt.gcf()

    def plot_ngram_frequencies(self, text, n=2, top_n=20):
        """Plot n-gram frequency distribution"""
        analyzer = TextAnalyzer()
        ngrams = analyzer.analyze_ngrams(text, n)

        if not ngrams:
            return None

        plt.figure(figsize=(12, 8))
        ngram_labels = [ng[0] for ng in ngrams[:top_n]]
        frequencies = [ng[1] for ng in ngrams[:top_n]]

        plt.barh(range(len(ngram_labels)), frequencies)
        plt.yticks(range(len(ngram_labels)), ngram_labels)
        plt.xlabel('Frequency')
        plt.ylabel(f'{n}-grams')
        plt.title(f'Top {top_n} {n}-gram Frequencies')
        plt.gca().invert_yaxis()

        plt.tight_layout()
        return plt.gcf()

    def plot_cluster_visualization(self, pca_components, labels, title="Text Clustering"):
        """Plot clustering results"""
        plt.figure(figsize=(10, 8))

        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise points in DBSCAN
                color = 'black'
                label_name = 'Noise'
            else:
                label_name = f'Cluster {label}'

            mask = labels == label
            plt.scatter(pca_components[mask, 0], pca_components[mask, 1],
                       c=[color], label=label_name, alpha=0.7, s=50)

        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        return plt.gcf()

    def plot_network_graph(self, graph, title="Pattern Network"):
        """Plot network visualization"""
        plt.figure(figsize=(12, 8))

        pos = nx.spring_layout(graph, k=1, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=300, alpha=0.7,
                              node_color='lightblue', edgecolors='black')

        # Draw edges
        nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color='gray')

        # Draw labels for high-degree nodes only
        degrees = dict(graph.degree())
        high_degree_nodes = [n for n, d in degrees.items() if d > np.percentile(list(degrees.values()), 80)]

        labels = {node: node for node in high_degree_nodes}
        nx.draw_networkx_labels(graph, pos, labels, font_size=8)

        plt.title(title)
        plt.axis('off')

        plt.tight_layout()
        return plt.gcf()

class AdvancedCryptanalysis:
    """Advanced cryptanalysis techniques"""

    def __init__(self):
        self.analyzer = TextAnalyzer()

    def hill_climbing_attack(self, ciphertext, max_iterations=1000):
        """Hill climbing attack on substitution cipher"""
        # This is a simplified implementation
        best_key = list('abcdefghijklmnopqrstuvwxyz')
        best_score = self.score_text(self.decrypt_with_key(ciphertext, best_key))

        for _ in range(max_iterations):
            # Try swapping two random letters
            new_key = best_key.copy()
            i, j = np.random.choice(26, 2, replace=False)
            new_key[i], new_key[j] = new_key[j], new_key[i]

            decrypted = self.decrypt_with_key(ciphertext, new_key)
            score = self.score_text(decrypted)

            if score > best_score:
                best_key = new_key
                best_score = score

        return ''.join(best_key), best_score

    def genetic_algorithm_attack(self, ciphertext, population_size=50, generations=100):
        """Genetic algorithm attack on substitution cipher"""
        population = self.initialize_population(population_size)

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(ciphertext, key) for key in population]

            # Select parents
            parents = self.select_parents(population, fitness_scores)

            # Create new population
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(len(parents), 2, replace=False)
                child = self.crossover(parents[parent1], parents[parent2])
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        # Return best solution
        best_key = max(population, key=lambda k: self.evaluate_fitness(ciphertext, k))
        best_score = self.evaluate_fitness(ciphertext, best_key)

        return ''.join(best_key), best_score

    def initialize_population(self, size):
        """Initialize population of random keys"""
        population = []
        for _ in range(size):
            key = list('abcdefghijklmnopqrstuvwxyz')
            np.random.shuffle(key)
            population.append(key)
        return population

    def evaluate_fitness(self, ciphertext, key):
        """Evaluate fitness of a key"""
        decrypted = self.decrypt_with_key(ciphertext, key)
        return self.score_text(decrypted)

    def select_parents(self, population, fitness_scores, num_parents=None):
        """Select parents using tournament selection"""
        if num_parents is None:
            num_parents = len(population) // 2

        parents = []
        for _ in range(num_parents):
            # Tournament selection
            tournament = np.random.choice(len(population), 3, replace=False)
            winner = max(tournament, key=lambda i: fitness_scores[i])
            parents.append(population[winner])

        return parents

    def crossover(self, parent1, parent2):
        """Crossover two parent keys"""
        # Single point crossover
        point = np.random.randint(1, 25)
        child = parent1[:point] + parent2[point:]

        # Fix duplicates
        used = set(child[:point])
        remaining = [c for c in parent2[point:] if c not in used]

        for i in range(point, 26):
            if len(remaining) > 0:
                child[i] = remaining.pop(0)
            else:
                # Fill with unused letters
                unused = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in child]
                child[i] = unused[i - point] if i - point < len(unused) else 'a'

        return child

    def mutate(self, key, mutation_rate=0.1):
        """Mutate a key"""
        if np.random.random() < mutation_rate:
            i, j = np.random.choice(26, 2, replace=False)
            key[i], key[j] = key[j], key[i]
        return key

    def decrypt_with_key(self, ciphertext, key):
        """Decrypt using substitution key"""
        key_map = {chr(97 + i): key[i] for i in range(26)}
        result = []

        for char in ciphertext.lower():
            if char in key_map:
                result.append(key_map[char])
            else:
                result.append(char)

        return ''.join(result)

    def score_text(self, text):
        """Score text based on English language patterns"""
        # Simple scoring based on quadgram frequencies
        # This is a simplified version - real implementation would use quadgram statistics
        score = 0

        # Prefer common English patterns
        common_patterns = ['the', 'and', 'ing', 'tion', 'that', 'with', 'from']
        for pattern in common_patterns:
            score += text.count(pattern)

        # Penalize uncommon letter combinations
        uncommon = ['zx', 'xq', 'qj', 'jz', 'qb', 'vj']
        for pattern in uncommon:
            score -= text.count(pattern)

        return score

    def analyze_vigenere_key_length(self, ciphertext, max_length=20):
        """Analyze possible Vigenère key lengths"""
        ciphertext = re.sub(r'[^a-z]', '', ciphertext.lower())

        ic_values = []
        for key_length in range(1, max_length + 1):
            # Split text into groups
            groups = ['' for _ in range(key_length)]
            for i, char in enumerate(ciphertext):
                groups[i % key_length] += char

            # Calculate average IC
            avg_ic = np.mean([self.analyzer.calculate_ic(group) for group in groups])
            ic_values.append((key_length, avg_ic))

        # Find peaks in IC values (likely key lengths)
        ic_scores = [ic for _, ic in ic_values]
        peaks, _ = find_peaks(ic_scores, height=np.mean(ic_scores) + np.std(ic_scores))

        likely_lengths = [ic_values[i][0] for i in peaks]

        return likely_lengths, ic_values
