// API Base URL
const API_URL = 'http://localhost:8000';

// Chart instance
let probabilityChart = null;

// DOM Elements
const inputText = document.getElementById('inputText');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
    checkAPIHealth();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    predictBtn.addEventListener('click', makePrediction);
    clearBtn.addEventListener('click', clearResults);

    // Example buttons
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            inputText.value = e.target.dataset.text;
        });
    });

    // Enter key to predict
    inputText.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            makePrediction();
        }
    });
}

// Load model info
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_URL}/model-info`);
        const data = await response.json();

        document.getElementById('infoModelName').textContent = data.model_name;
        document.getElementById('infoF1').textContent = data.f1_macro.toFixed(4);
        document.getElementById('infoAccuracy').textContent = data.accuracy.toFixed(4);
        document.getElementById('infoClasses').textContent = data.num_classes;
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('infoModelName').textContent = 'Error loading';
    }
}

// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();

        const statusBadge = document.getElementById('apiStatus');
        if (data.status === 'healthy') {
            statusBadge.textContent = '✅ Healthy';
            statusBadge.classList.add('healthy');
        } else {
            statusBadge.textContent = '❌ Unhealthy';
        }
    } catch (error) {
        console.error('Error checking health:', error);
        document.getElementById('apiStatus').textContent = '❌ Offline';
    }
}

// Make prediction
async function makePrediction() {
    const text = inputText.value.trim();

    if (!text) {
        alert('Please enter some text to analyze');
        return;
    }

    // Show loading
    loading.style.display = 'flex';
    predictBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Error making prediction. Please try again.');
    } finally {
        loading.style.display = 'none';
        predictBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // Update prediction
    document.getElementById('predictionLabel').textContent = data.prediction;

    // Update confidence
    const confidencePct = (data.confidence * 100).toFixed(1);
    document.getElementById('confidenceValue').textContent = `${confidencePct}%`;
    document.getElementById('confidenceProgress').style.width = `${confidencePct}%`;

    // Update metadata
    document.getElementById('modelVersion').textContent = data.model_version;
    document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();

    const driftBadge = document.getElementById('driftStatus');
    if (data.drift_detected) {
        driftBadge.textContent = '⚠️ Drift Detected';
        driftBadge.classList.add('drift');
    } else {
        driftBadge.textContent = '✅ No Drift';
        driftBadge.classList.add('healthy');
    }

    // Update chart
    updateChart(data.probabilities);
}

// Update probability chart
function updateChart(probabilities) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');

    // Destroy existing chart
    if (probabilityChart) {
        probabilityChart.destroy();
    }

    // Sort probabilities by value
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(item => item[0]);
    const values = sorted.map(item => (item[1] * 100).toFixed(2));

    // Create new chart
    probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: values,
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(118, 75, 162, 0.8)',
                    'rgba(237, 100, 166, 0.8)',
                    'rgba(255, 154, 158, 0.8)',
                    'rgba(255, 198, 128, 0.8)',
                    'rgba(250, 227, 133, 0.8)'
                ],
                borderColor: [
                    'rgba(102, 126, 234, 1)',
                    'rgba(118, 75, 162, 1)',
                    'rgba(237, 100, 166, 1)',
                    'rgba(255, 154, 158, 1)',
                    'rgba(255, 198, 128, 1)',
                    'rgba(250, 227, 133, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Clear results
function clearResults() {
    inputText.value = '';
    resultsSection.style.display = 'none';
    if (probabilityChart) {
        probabilityChart.destroy();
        probabilityChart = null;
    }
}
