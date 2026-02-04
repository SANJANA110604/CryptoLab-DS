# CryptoLab DS
A comprehensive web application that combines cryptography and data science techniques to explore, analyze, and break various encryption methods. Built with Flask backend and interactive web frontend.

## 🌟 Features

### Cryptography Tools
- **Classical Ciphers**: Caesar and Vigenère cipher encryption/decryption
- **Modern Encryption**: Base64 encoding/decoding, RSA, AES encryption
- **Hash Functions**: SHA-256, SHA3-256, BLAKE2b with HMAC support
- **Secure Random Generation**: Cryptographically secure tokens and passwords

### Data Science Analysis
- **Frequency Analysis**: Letter distribution analysis with interactive charts
- **Pattern Recognition**: Identify repeated sequences in ciphertext
- **Entropy Calculation**: Measure randomness of encrypted text
- **Statistical Analysis**: Comprehensive text property analysis
- **Clustering Analysis**: Group similar texts using ML algorithms
- **Network Analysis**: Pattern network visualization

### Machine Learning Cryptanalysis
- **Cipher Classification**: ML model to identify cipher types
- **Neural Cryptanalysis**: Neural network-based cipher breaking simulation
- **Advanced Cryptanalysis**: Hill climbing and genetic algorithm attacks
- **Training Data Generation**: Automated dataset creation for ML models

### Interactive Web Interface
- Real-time encryption/decryption tools
- Interactive charts and visualizations using Chart.js
- Responsive design for desktop and mobile
- Educational content on cryptography history

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Sanjana152911/cryptolab-ds.git
cd cryptolab-ds
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## 📖 Usage

### Web Interface
The application provides an interactive web interface with several sections:

- **Cryptography Tools**: Encrypt/decrypt text using various algorithms
- **Data Science**: Analyze ciphertext using statistical methods
- **ML Cryptanalysis**: Apply machine learning to cipher breaking
- **Algorithms**: Learn about different cryptographic techniques
- **History**: Timeline of cryptography development

### API Usage
The Flask backend provides RESTful API endpoints. Here are some examples:

#### Caesar Cipher
```bash
curl -X POST http://localhost:5000/api/caesar \
  -H "Content-Type: application/json" \
  -d '{"text": "HELLO", "shift": 3, "action": "encrypt"}'
```

#### Frequency Analysis
```bash
curl -X POST http://localhost:5000/api/frequency \
  -H "Content-Type: application/json" \
  -d '{"text": "Your encrypted text here"}'
```

#### ML Cipher Classification
```bash
curl -X POST http://localhost:5000/api/ml/classify-cipher \
  -H "Content-Type: application/json" \
  -d '{"text": "Gur dhvpx oebja sbk whzcf bire gur ynml qbt."}'
```

## 🔧 API Endpoints

### Cryptography
- `POST /api/caesar` - Caesar cipher operations
- `POST /api/vigenere` - Vigenère cipher operations
- `POST /api/base64` - Base64 encoding/decoding
- `POST /api/rsa/generate` - Generate RSA key pairs
- `POST /api/rsa/encrypt` - RSA encryption
- `POST /api/rsa/decrypt` - RSA decryption
- `POST /api/aes/generate-key` - Generate AES keys
- `POST /api/aes/encrypt` - AES encryption
- `POST /api/aes/decrypt` - AES decryption
- `POST /api/hash` - Cryptographic hashing
- `POST /api/hmac` - HMAC generation

### Data Science
- `POST /api/frequency` - Frequency analysis
- `POST /api/entropy` - Entropy calculation
- `POST /api/patterns` - Pattern recognition
- `POST /api/analysis/statistical` - Statistical analysis
- `POST /api/analysis/text-properties` - Text property analysis
- `POST /api/analysis/cluster` - Clustering analysis
- `POST /api/analysis/network` - Network analysis

### Machine Learning
- `POST /api/ml/classify-cipher` - ML cipher classification
- `POST /api/ml/train-classifier` - Train classification model
- `POST /api/ml/neural-analysis` - Neural network analysis
- `POST /api/ml/train-neural` - Train neural network
- `POST /api/analysis/advanced-crypto` - Advanced cryptanalysis

### Utilities
- `GET /api/health` - Health check
- `POST /api/crypto/random/token` - Generate secure tokens
- `POST /api/crypto/random/password` - Generate secure passwords
- `POST /api/crypto/benchmark` - Run performance benchmarks

## 🛠️ Technologies Used

### Backend
- **Flask**: Web framework for API development
- **cryptography**: Modern cryptographic operations
- **scikit-learn**: Machine learning algorithms
- **TensorFlow**: Neural network implementations
- **NumPy/Pandas**: Data manipulation and analysis
- **NetworkX**: Graph and network analysis
- **Matplotlib/Seaborn**: Data visualization

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript (ES6+)**: Interactive functionality
- **Chart.js**: Data visualization in browser
- **Responsive Design**: Mobile-friendly interface

## 📁 Project Structure

```
cryptolab-ds/
├── app.py                    # Flask application main file
├── index.html               # Main web interface
├── script.js                # Frontend JavaScript logic
├── styles.css               # CSS styling
├── requirements.txt         # Python dependencies
├── ml_models.py            # Machine learning models
├── data_science_utils.py   # Data science utilities
├── crypto_utils.py         # Cryptographic utilities
└── README.md               # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used for securing sensitive information in production environments. Always use established cryptographic libraries and follow security best practices for real-world applications.

## 🙏 Acknowledgments

- Inspired by the intersection of cryptography and data science
- Built for educational purposes to demonstrate cryptographic concepts
- Thanks to the open-source community for the amazing libraries used
---
**CryptoLab DS** - Where Cryptography Meets Data Science 🔐📊
