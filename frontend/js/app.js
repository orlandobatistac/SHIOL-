document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const generateBtn = document.getElementById('generate-btn');
    const btnText = document.getElementById('btn-text');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('error-message');
    const resultArea = document.getElementById('result-area');
    const initialText = document.getElementById('initial-text');
    const predictionDisplay = document.getElementById('prediction-display');

    // --- API Configuration ---
    const API_URL = 'http://127.0.0.1:8000/api/v1/predict';

    // --- Event Listener ---
    generateBtn.addEventListener('click', fetchPrediction);

    // --- Core Functions ---
    async function fetchPrediction() {
        setLoadingState(true);
        clearPreviousState();

        try {
            const response = await fetch(API_URL);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            displayPrediction(data.prediction);
        } catch (error) {
            console.error('Fetch error:', error);
            displayError('Failed to generate numbers. Please try again.');
        } finally {
            setLoadingState(false);
        }
    }

    // --- UI State Management ---
    function setLoadingState(isLoading) {
        if (isLoading) {
            generateBtn.disabled = true;
            btnText.textContent = 'Generating...';
            loader.classList.remove('hidden');
        } else {
            generateBtn.disabled = false;
            btnText.textContent = 'Generate Numbers';
            loader.classList.add('hidden');
        }
    }

    function displayPrediction(numbers) {
        initialText.classList.add('hidden');
        predictionDisplay.classList.remove('hidden');
        
        const whiteBalls = numbers.slice(0, 5).map(num => `<div class="bg-white text-gray-800 rounded-full w-12 h-12 flex items-center justify-center font-bold text-xl mx-1">${num}</div>`).join('');
        const powerball = `<div class="bg-red-500 text-white rounded-full w-12 h-12 flex items-center justify-center font-bold text-xl mx-1">${numbers[5]}</div>`;
        
        predictionDisplay.innerHTML = `
            <div class="flex items-center justify-center flex-wrap">
                ${whiteBalls}
                <div class="border-l-2 border-gray-300 dark:border-gray-600 h-8 mx-2"></div>
                ${powerball}
            </div>
        `;
    }

    function displayError(message) {
        errorMessage.textContent = message;
    }

    function clearPreviousState() {
        errorMessage.textContent = '';
        if (!initialText.classList.contains('hidden')) {
            predictionDisplay.classList.add('hidden');
        }
    }
});