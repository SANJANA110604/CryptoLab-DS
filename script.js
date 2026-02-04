// Initialize charts
let freqChart, entropyChart;

// Caesar Cipher Implementation
document.getElementById('caesar-encrypt').addEventListener('click', async function() {
    const text = document.getElementById('caesar-text').value;
    const shift = parseInt(document.getElementById('caesar-shift').value);
    try {
        const response = await fetch('http://localhost:5000/api/caesar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                shift: shift,
                action: 'encrypt'
            })
        });
        const data = await response.json();
        document.getElementById('caesar-result').textContent = data.result;
    } catch (error) {
        document.getElementById('caesar-result').textContent = 'Error: ' + error.message;
    }
});

document.getElementById('caesar-decrypt').addEventListener('click', async function() {
    const text = document.getElementById('caesar-text').value;
    const shift = parseInt(document.getElementById('caesar-shift').value);
    try {
        const response = await fetch('http://localhost:5000/api/caesar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                shift: shift,
                action: 'decrypt'
            })
        });
        const data = await response.json();
        document.getElementById('caesar-result').textContent = data.result;
    } catch (error) {
        document.getElementById('caesar-result').textContent = 'Error: ' + error.message;
    }
});

// Vigenère Cipher Implementation
document.getElementById('vigenere-encrypt').addEventListener('click', async function() {
    const text = document.getElementById('vigenere-text').value;
    const key = document.getElementById('vigenere-key').value.toUpperCase();
    try {
        const response = await fetch('http://localhost:5000/api/vigenere', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                key: key,
                action: 'encrypt'
            })
        });
        const data = await response.json();
        document.getElementById('vigenere-result').textContent = data.result;
    } catch (error) {
        document.getElementById('vigenere-result').textContent = 'Error: ' + error.message;
    }
});

document.getElementById('vigenere-decrypt').addEventListener('click', async function() {
    const text = document.getElementById('vigenere-text').value;
    const key = document.getElementById('vigenere-key').value.toUpperCase();
    try {
        const response = await fetch('http://localhost:5000/api/vigenere', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                key: key,
                action: 'decrypt'
            })
        });
        const data = await response.json();
        document.getElementById('vigenere-result').textContent = data.result;
    } catch (error) {
        document.getElementById('vigenere-result').textContent = 'Error: ' + error.message;
    }
});

// Base64 Implementation
document.getElementById('base64-encode').addEventListener('click', async function() {
    const text = document.getElementById('base64-text').value;
    try {
        const response = await fetch('http://localhost:5000/api/base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                action: 'encode'
            })
        });
        const data = await response.json();
        document.getElementById('base64-result').textContent = data.result;
    } catch (error) {
        document.getElementById('base64-result').textContent = 'Error: ' + error.message;
    }
});

document.getElementById('base64-decode').addEventListener('click', async function() {
    const text = document.getElementById('base64-text').value;
    try {
        const response = await fetch('http://localhost:5000/api/base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                action: 'decode'
            })
        });
        const data = await response.json();
        document.getElementById('base64-result').textContent = data.result;
    } catch (error) {
        document.getElementById('base64-result').textContent = 'Error: ' + error.message;
    }
});

// Data Science Features

// Frequency Analysis
document.getElementById('analyze-freq').addEventListener('click', async function() {
    const text = document.getElementById('freq-text').value;
    try {
        const response = await fetch('http://localhost:5000/api/frequency', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text
            })
        });
        const freqData = await response.json();
        displayFrequencyChart(freqData);
    } catch (error) {
        console.error('Error:', error);
    }
});

function displayFrequencyChart(freqData) {
    const ctx = document.getElementById('freq-chart').getContext('2d');
    
    if (freqChart) {
        freqChart.destroy();
    }
    
    const letters = Object.keys(freqData.percentages);
    const percentages = letters.map(letter => parseFloat(freqData.percentages[letter]));
    
    freqChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: letters,
            datasets: [{
                label: 'Letter Frequency (%)',
                data: percentages,
                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Frequency (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Letters'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Letter Frequency Distribution'
                }
            }
        }
    });
}

// Pattern Recognition
document.getElementById('find-patterns').addEventListener('click', async function() {
    const text = document.getElementById('pattern-text').value;
    try {
        const response = await fetch('http://localhost:5000/api/patterns', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                min_length: 3,
                max_length: 10
            })
        });
        const data = await response.json();
        displayPatterns(data.patterns);
    } catch (error) {
        document.getElementById('pattern-result').innerHTML = 'Error: ' + error.message;
    }
});

function displayPatterns(patterns) {
    let result = '<h4>Repeated Patterns Found:</h4>';
    
    if (patterns.length === 0) {
        result += '<p>No repeated patterns found.</p>';
    } else {
        patterns.forEach(pattern => {
            result += `<p><strong>"${pattern.pattern}"</strong> appears ${pattern.count} times at positions: ${pattern.positions.join(', ')}</p>`;
        });
    }
    
    document.getElementById('pattern-result').innerHTML = result;
}

// Entropy Analysis
document.getElementById('calculate-entropy').addEventListener('click', async function() {
    const text = document.getElementById('entropy-text').value;
    try {
        const response = await fetch('http://localhost:5000/api/entropy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text
            })
        });
        const data = await response.json();
        const entropy = data.entropy;
        document.getElementById('entropy-result').innerHTML = `Entropy: ${entropy.toFixed(4)} bits per character`;
        displayEntropyComparison(entropy);
    } catch (error) {
        document.getElementById('entropy-result').innerHTML = 'Error: ' + error.message;
    }
});

function displayEntropyComparison(entropy) {
    const ctx = document.getElementById('entropy-chart').getContext('2d');
    
    if (entropyChart) {
        entropyChart.destroy();
    }
    
    // Reference entropy values for comparison
    const referenceData = {
        'English Text': 4.0,
        'Random Text': 4.7,
        'Your Text': entropy
    };
    
    entropyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(referenceData),
            datasets: [{
                label: 'Entropy (bits/character)',
                data: Object.values(referenceData),
                backgroundColor: [
                    'rgba(52, 152, 219, 0.7)',
                    'rgba(46, 204, 113, 0.7)',
                    'rgba(231, 76, 60, 0.7)'
                ],
                borderColor: [
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 5,
                    title: {
                        display: true,
                        text: 'Entropy (bits/character)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Entropy Comparison'
                }
            }
        }
    });
}

// ML Cryptanalysis Features

// Cipher Classification
document.getElementById('classify-cipher').addEventListener('click', function() {
    const text = document.getElementById('cipher-sample').value;
    const classification = classifyCipher(text);
    document.getElementById('classification-result').innerHTML = `
        <h4>Cipher Classification Result:</h4>
        <p><strong>Most likely:</strong> ${classification.mostLikely}</p>
        <p><strong>Confidence:</strong> ${classification.confidence}%</p>
        <p><strong>Details:</strong> ${classification.details}</p>
    `;
});

function classifyCipher(text) {
    // This is a simplified simulation - a real implementation would use a trained ML model
    const features = {
        hasNonAlpha: /[^a-zA-Z\s]/.test(text),
        caseConsistent: text === text.toUpperCase() || text === text.toLowerCase(),
        wordLengths: text.split(/\s+/).map(word => word.length),
        charDistribution: calculateEntropy(text)
    };
    
    // Simple heuristic-based classification
    let mostLikely = "Unknown";
    let confidence = 50;
    let details = "Unable to determine with high confidence.";
    
    if (features.charDistribution > 4.5) {
        mostLikely = "Random Text or One-Time Pad";
        confidence = 75;
        details = "High entropy suggests random distribution of characters.";
    } else if (!features.hasNonAlpha && features.caseConsistent) {
        mostLikely = "Classical Cipher (Caesar, Substitution, or Vigenère)";
        confidence = 70;
        details = "Text contains only letters with consistent casing, typical of classical ciphers.";
    } else if (features.hasNonAlpha) {
        mostLikely = "Base64 or Modern Encryption";
        confidence = 65;
        details = "Presence of non-alphabetic characters suggests encoding like Base64 or modern encryption.";
    }
    
    return { mostLikely, confidence, details };
}

// Neural Cryptanalysis Simulation
document.getElementById('neural-analysis').addEventListener('click', function() {
    const text = document.getElementById('neural-input').value;
    const result = simulateNeuralAnalysis(text);
    document.getElementById('neural-result').innerHTML = `
        <h4>Neural Network Analysis:</h4>
        <p><strong>Pattern Detected:</strong> ${result.pattern}</p>
        <p><strong>Confidence Score:</strong> ${result.confidence}</p>
        <p><strong>Recommended Approach:</strong> ${result.recommendation}</p>
        <div style="margin-top: 10px;">
            <strong>Layer Activations:</strong>
            <ul>
                ${result.activations.map(activation => `<li>${activation}</li>`).join('')}
            </ul>
        </div>
    `;
});

function simulateNeuralAnalysis(text) {
    // Simulate a neural network analysis
    const patterns = [
        "Repetitive sequences detected",
        "Low entropy pattern identified",
        "Frequency distribution matches substitution cipher",
        "Possible polyalphabetic cipher characteristics"
    ];
    
    const randomPattern = patterns[Math.floor(Math.random() * patterns.length)];
    const confidence = (70 + Math.random() * 25).toFixed(2);
    
    const recommendations = [
        "Try frequency analysis with n-gram models",
        "Apply Kasiski examination for key length detection",
        "Use hill-climbing algorithm with fitness function",
        "Implement genetic algorithm for key space exploration"
    ];
    
    const randomRecommendation = recommendations[Math.floor(Math.random() * recommendations.length)];
    
    const activations = [
        "Input layer: Character encoding",
        "Hidden layer 1: Pattern recognition",
        "Hidden layer 2: Sequence analysis",
        "Hidden layer 3: Statistical modeling",
        "Output layer: Cipher classification"
    ];
    
    return {
        pattern: randomPattern,
        confidence: confidence + "%",
        recommendation: randomRecommendation,
        activations: activations
    };
}

// Training Data Generator
document.getElementById('generate-data').addEventListener('click', function() {
    const baseText = document.getElementById('training-text').value;
    const trainingData = generateTrainingData(baseText);
    document.getElementById('training-result').innerHTML = `
        <h4>Generated Training Data:</h4>
        <p><strong>Original:</strong> ${trainingData.original}</p>
        <p><strong>Caesar (shift=3):</strong> ${trainingData.caesar}</p>
        <p><strong>Vigenère (key=KEY):</strong> ${trainingData.vigenere}</p>
        <p><strong>Base64:</strong> ${trainingData.base64}</p>
        <p><strong>Entropy:</strong> ${trainingData.entropy.toFixed(4)}</p>
    `;
});

function generateTrainingData(text) {
    return {
        original: text,
        caesar: caesarCipher(text, 3),
        vigenere: vigenereCipher(text, "KEY", true),
        base64: btoa(text),
        entropy: calculateEntropy(text)
    };
}

// Initialize with some sample analyses
window.onload = function() {
    // Note: Removed automatic API calls on page load to avoid errors when backend is not running
    // Users can manually click buttons to test functionality
};