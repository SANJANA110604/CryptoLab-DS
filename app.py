from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import math
from collections import Counter
import re

# Import custom modules
from ml_models import CipherClassifier, NeuralCryptanalyzer, StatisticalAnalyzer
from data_science_utils import TextAnalyzer, ClusteringAnalyzer, NetworkAnalyzer, StatisticalVisualizer, AdvancedCryptanalysis
from crypto_utils import AdvancedCrypto, KeyDerivation, HashFunctions, SecureRandom, SecureMessaging, CryptoBenchmark

app = Flask(__name__)
CORS(app)

def caesar_cipher(text, shift, encrypt=True):
    """Implement Caesar cipher encryption/decryption"""
    result = ""
    shift = shift if encrypt else -shift
    
    for char in text:
        if char.isalpha():
            ascii_offset = 65 if char.isupper() else 97
            result += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
        else:
            result += char
    return result

def vigenere_cipher(text, key, encrypt=True):
    """Implement Vigenère cipher encryption/decryption"""
    result = ""
    key = key.upper()
    key_index = 0
    
    for char in text:
        if char.isalpha():
            ascii_offset = 65 if char.isupper() else 97
            key_char = key[key_index % len(key)]
            key_shift = ord(key_char) - 65
            
            if encrypt:
                shift = key_shift
            else:
                shift = -key_shift
                
            result += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
            key_index += 1
        else:
            result += char
    return result

def calculate_frequency(text):
    """Calculate letter frequency distribution"""
    text = text.lower()
    letters = [char for char in text if char.isalpha()]
    
    if not letters:
        return {}
    
    freq = Counter(letters)
    total = len(letters)
    
    percentages = {letter: (count / total * 100) for letter, count in freq.items()}
    
    # Fill missing letters with 0
    for char in 'abcdefghijklmnopqrstuvwxyz':
        if char not in percentages:
            percentages[char] = 0
    
    return {
        'counts': dict(freq),
        'percentages': percentages,
        'total': total
    }

def calculate_entropy(text):
    """Calculate Shannon entropy of text"""
    # Clean text - keep only letters
    clean_text = re.sub(r'[^a-zA-Z]', '', text).lower()
    
    if not clean_text:
        return 0
    
    # Calculate frequencies
    freq = Counter(clean_text)
    total = len(clean_text)
    
    # Calculate entropy
    entropy = 0
    for count in freq.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    
    return entropy

def find_patterns(text, min_length=3, max_length=10):
    """Find repeated patterns in text"""
    text = re.sub(r'[^a-z]', '', text.lower())
    patterns = {}
    
    for length in range(min_length, max_length + 1):
        for i in range(len(text) - length + 1):
            pattern = text[i:i + length]
            
            if pattern in patterns:
                patterns[pattern]['count'] += 1
                patterns[pattern]['positions'].append(i)
            else:
                patterns[pattern] = {
                    'count': 1,
                    'positions': [i]
                }
    
    # Return only patterns that appear more than once
    repeated_patterns = {p: d for p, d in patterns.items() if d['count'] > 1}
    
    # Sort by frequency
    sorted_patterns = dict(sorted(repeated_patterns.items(), 
                                  key=lambda x: x[1]['count'], 
                                  reverse=True))
    
    return sorted_patterns

@app.route('/api/caesar', methods=['POST'])
def api_caesar():
    """API endpoint for Caesar cipher"""
    data = request.json
    text = data.get('text', '')
    shift = data.get('shift', 3)
    action = data.get('action', 'encrypt')  # 'encrypt' or 'decrypt'
    
    if action == 'encrypt':
        result = caesar_cipher(text, shift, True)
    else:
        result = caesar_cipher(text, shift, False)
    
    return jsonify({'result': result})

@app.route('/api/vigenere', methods=['POST'])
def api_vigenere():
    """API endpoint for Vigenère cipher"""
    data = request.json
    text = data.get('text', '')
    key = data.get('key', 'KEY')
    action = data.get('action', 'encrypt')  # 'encrypt' or 'decrypt'
    
    if action == 'encrypt':
        result = vigenere_cipher(text, key, True)
    else:
        result = vigenere_cipher(text, key, False)
    
    return jsonify({'result': result})

@app.route('/api/base64', methods=['POST'])
def api_base64():
    """API endpoint for Base64 encoding/decoding"""
    data = request.json
    text = data.get('text', '')
    action = data.get('action', 'encode')  # 'encode' or 'decode'
    
    try:
        if action == 'encode':
            # Encode string to Base64
            encoded_bytes = base64.b64encode(text.encode('utf-8'))
            result = encoded_bytes.decode('utf-8')
        else:
            # Decode Base64 to string
            decoded_bytes = base64.b64decode(text)
            result = decoded_bytes.decode('utf-8')
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    return jsonify({'result': result})

@app.route('/api/frequency', methods=['POST'])
def api_frequency():
    """API endpoint for frequency analysis"""
    data = request.json
    text = data.get('text', '')
    
    freq_data = calculate_frequency(text)
    
    return jsonify(freq_data)

@app.route('/api/entropy', methods=['POST'])
def api_entropy():
    """API endpoint for entropy calculation"""
    data = request.json
    text = data.get('text', '')
    
    entropy = calculate_entropy(text)
    
    return jsonify({'entropy': entropy})

@app.route('/api/patterns', methods=['POST'])
def api_patterns():
    """API endpoint for pattern recognition"""
    data = request.json
    text = data.get('text', '')
    min_length = data.get('min_length', 3)
    max_length = data.get('max_length', 10)
    
    patterns = find_patterns(text, min_length, max_length)
    
    # Convert to list for JSON serialization
    patterns_list = []
    for pattern, data in patterns.items():
        patterns_list.append({
            'pattern': pattern,
            'count': data['count'],
            'positions': data['positions']
        })
    
    return jsonify({'patterns': patterns_list})

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for cipher classification"""
    data = request.json
    text = data.get('text', '')
    
    # Simplified classification logic
    entropy = calculate_entropy(text)
    has_special_chars = bool(re.search(r'[^a-zA-Z\s]', text))
    all_upper_or_lower = text.isupper() or text.islower()
    
    if entropy > 4.5:
        classification = "Random Text or One-Time Pad"
        confidence = 75
        details = "High entropy suggests random distribution of characters."
    elif not has_special_chars and all_upper_or_lower:
        classification = "Classical Cipher (Caesar, Substitution, or Vigenère)"
        confidence = 70
        details = "Text contains only letters with consistent casing, typical of classical ciphers."
    elif has_special_chars:
        classification = "Base64 or Modern Encryption"
        confidence = 65
        details = "Presence of non-alphabetic characters suggests encoding like Base64 or modern encryption."
    else:
        classification = "Unknown/Plain Text"
        confidence = 50
        details = "Unable to determine cipher type with high confidence."
    
    return jsonify({
        'classification': classification,
        'confidence': confidence,
        'details': details,
        'entropy': entropy
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'CryptoLab API'})

# Initialize ML and crypto utilities
cipher_classifier = CipherClassifier()
neural_analyzer = NeuralCryptanalyzer()
statistical_analyzer = StatisticalAnalyzer()
text_analyzer = TextAnalyzer()
clustering_analyzer = ClusteringAnalyzer()
network_analyzer = NetworkAnalyzer()
visualizer = StatisticalVisualizer()
advanced_crypto = AdvancedCrypto()
key_derivation = KeyDerivation()
hash_functions = HashFunctions()
secure_random = SecureRandom()
secure_messaging = SecureMessaging()
crypto_benchmark = CryptoBenchmark()

@app.route('/api/ml/classify-cipher', methods=['POST'])
def api_ml_classify():
    """ML-based cipher classification"""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = cipher_classifier.predict(text)
    return jsonify(result)

@app.route('/api/ml/train-classifier', methods=['POST'])
def api_train_classifier():
    """Train the ML cipher classifier"""
    try:
        accuracy = cipher_classifier.train()
        return jsonify({'message': 'Classifier trained successfully', 'accuracy': accuracy})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/neural-analysis', methods=['POST'])
def api_neural_analysis():
    """Neural network analysis of text"""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = neural_analyzer.analyze_text(text)
    return jsonify(result)

@app.route('/api/ml/train-neural', methods=['POST'])
def api_train_neural():
    """Train neural network model"""
    try:
        history = neural_analyzer.train_neural_model()
        return jsonify({'message': 'Neural model trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/statistical', methods=['POST'])
def api_statistical_analysis():
    """Comprehensive statistical analysis"""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    analysis = statistical_analyzer.analyze_text(text)
    return jsonify(analysis)

@app.route('/api/analysis/text-properties', methods=['POST'])
def api_text_properties():
    """Detailed text property analysis"""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    properties = text_analyzer.analyze_text_properties(text)
    return jsonify(properties)

@app.route('/api/analysis/cluster', methods=['POST'])
def api_cluster_analysis():
    """Clustering analysis of multiple texts"""
    data = request.json
    texts = data.get('texts', [])

    if not texts or len(texts) < 2:
        return jsonify({'error': 'At least 2 texts required for clustering'}), 400

    result = clustering_analyzer.perform_clustering(texts)
    cluster_summary = clustering_analyzer.analyze_clusters(texts, result['labels'])

    return jsonify({
        'labels': result['labels'],
        'silhouette_score': result['silhouette_score'],
        'cluster_summary': cluster_summary
    })

@app.route('/api/analysis/network', methods=['POST'])
def api_network_analysis():
    """Network analysis of text patterns"""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    graph = network_analyzer.build_pattern_network(text)
    properties = network_analyzer.analyze_network_properties()

    return jsonify(properties)

@app.route('/api/analysis/advanced-crypto', methods=['POST'])
def api_advanced_crypto_analysis():
    """Advanced cryptanalysis techniques"""
    data = request.json
    text = data.get('text', '')
    method = data.get('method', 'hill_climbing')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if method == 'hill_climbing':
        key, score = advanced_crypto.hill_climbing_attack(text)
        return jsonify({'method': 'hill_climbing', 'key': key, 'score': score})
    elif method == 'genetic':
        key, score = advanced_crypto.genetic_algorithm_attack(text)
        return jsonify({'method': 'genetic', 'key': key, 'score': score})
    elif method == 'vigenere_analysis':
        lengths, ic_values = advanced_crypto.analyze_vigenere_key_length(text)
        return jsonify({'likely_lengths': lengths, 'ic_values': ic_values})
    else:
        return jsonify({'error': 'Unknown method'}), 400

@app.route('/api/crypto/rsa/generate', methods=['POST'])
def api_rsa_generate():
    """Generate RSA key pair"""
    try:
        private_key, public_key = advanced_crypto.generate_rsa_keypair()
        return jsonify({
            'private_key': private_key,
            'public_key': public_key
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/rsa/encrypt', methods=['POST'])
def api_rsa_encrypt():
    """RSA encryption"""
    data = request.json
    public_key = data.get('public_key', '')
    text = data.get('text', '')

    if not public_key or not text:
        return jsonify({'error': 'Public key and text required'}), 400

    try:
        ciphertext = advanced_crypto.rsa_encrypt(public_key, text)
        return jsonify({'ciphertext': ciphertext})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/rsa/decrypt', methods=['POST'])
def api_rsa_decrypt():
    """RSA decryption"""
    data = request.json
    private_key = data.get('private_key', '')
    ciphertext = data.get('ciphertext', '')

    if not private_key or not ciphertext:
        return jsonify({'error': 'Private key and ciphertext required'}), 400

    try:
        plaintext = advanced_crypto.rsa_decrypt(private_key, ciphertext)
        return jsonify({'plaintext': plaintext})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/aes/generate-key', methods=['POST'])
def api_aes_generate_key():
    """Generate AES key"""
    data = request.json
    key_size = data.get('key_size', 256)

    try:
        key = advanced_crypto.generate_aes_key(key_size)
        return jsonify({'key': key})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/aes/encrypt', methods=['POST'])
def api_aes_encrypt():
    """AES encryption"""
    data = request.json
    key = data.get('key', '')
    text = data.get('text', '')
    mode = data.get('mode', 'CBC')

    if not key or not text:
        return jsonify({'error': 'Key and text required'}), 400

    try:
        ciphertext, iv = advanced_crypto.aes_encrypt(key, text, mode)
        return jsonify({'ciphertext': ciphertext, 'iv': iv})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/aes/decrypt', methods=['POST'])
def api_aes_decrypt():
    """AES decryption"""
    data = request.json
    key = data.get('key', '')
    ciphertext = data.get('ciphertext', '')
    iv = data.get('iv', '')
    mode = data.get('mode', 'CBC')

    if not key or not ciphertext or not iv:
        return jsonify({'error': 'Key, ciphertext, and IV required'}), 400

    try:
        plaintext = advanced_crypto.aes_decrypt(key, ciphertext, iv, mode)
        return jsonify({'plaintext': plaintext})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/hash', methods=['POST'])
def api_hash():
    """Cryptographic hash functions"""
    data = request.json
    algorithm = data.get('algorithm', 'sha256')
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Text required'}), 400

    try:
        if algorithm == 'sha256':
            result = hash_functions.sha256_hash(text)
        elif algorithm == 'sha3_256':
            result = hash_functions.sha3_256_hash(text)
        elif algorithm == 'blake2b':
            key = data.get('key', '')
            result = hash_functions.blake2b_hash(text, key if key else None)
        else:
            return jsonify({'error': 'Unsupported algorithm'}), 400

        return jsonify({'hash': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/hmac', methods=['POST'])
def api_hmac():
    """HMAC generation"""
    data = request.json
    key = data.get('key', '')
    message = data.get('message', '')

    if not key or not message:
        return jsonify({'error': 'Key and message required'}), 400

    try:
        hmac_result = hash_functions.hmac_sha256(key, message)
        return jsonify({'hmac': hmac_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/key-derivation', methods=['POST'])
def api_key_derivation():
    """Key derivation functions"""
    data = request.json
    method = data.get('method', 'pbkdf2')
    password = data.get('password', '')

    if not password:
        return jsonify({'error': 'Password required'}), 400

    try:
        if method == 'pbkdf2':
            key, salt = key_derivation.pbkdf2_derive(password)
        elif method == 'scrypt':
            key, salt = key_derivation.scrypt_derive(password)
        else:
            return jsonify({'error': 'Unsupported method'}), 400

        return jsonify({'key': key, 'salt': salt})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/random/token', methods=['POST'])
def api_random_token():
    """Generate secure random token"""
    data = request.json
    length = data.get('length', 32)

    try:
        token = secure_random.generate_token(length)
        return jsonify({'token': token})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/random/password', methods=['POST'])
def api_random_password():
    """Generate secure random password"""
    data = request.json
    length = data.get('length', 16)
    include_special = data.get('include_special', True)

    try:
        password = secure_random.generate_password(length, include_special)
        return jsonify({'password': password})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/benchmark', methods=['POST'])
def api_crypto_benchmark():
    """Run cryptographic benchmarks"""
    try:
        results = crypto_benchmark.run_full_benchmark()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/messaging/key-exchange', methods=['POST'])
def api_key_exchange():
    """Generate key exchange parameters"""
    try:
        key_data = secure_messaging.generate_key_exchange()
        return jsonify(key_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/messaging/encrypt', methods=['POST'])
def api_encrypt_message():
    """Encrypt a message for secure transmission"""
    data = request.json
    sender_private_key = data.get('sender_private_key', '')
    recipient_public_key = data.get('recipient_public_key', '')
    message = data.get('message', '')

    if not sender_private_key or not recipient_public_key or not message:
        return jsonify({'error': 'Sender private key, recipient public key, and message required'}), 400

    try:
        encrypted_data = secure_messaging.encrypt_message(
            sender_private_key, recipient_public_key, message
        )
        return jsonify(encrypted_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/messaging/decrypt', methods=['POST'])
def api_decrypt_message():
    """Decrypt a received message"""
    data = request.json
    recipient_private_key = data.get('recipient_private_key', '')
    sender_public_key = data.get('sender_public_key', '')
    encrypted_data = data.get('encrypted_data', {})

    if not recipient_private_key or not sender_public_key or not encrypted_data:
        return jsonify({'error': 'Recipient private key, sender public key, and encrypted data required'}), 400

    try:
        message = secure_messaging.decrypt_message(
            recipient_private_key, sender_public_key, encrypted_data
        )
        if message is None:
            return jsonify({'error': 'Message verification failed'}), 400
        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
