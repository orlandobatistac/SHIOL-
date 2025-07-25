document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const generateBtn = document.getElementById('generate-btn');
    const btnText = document.getElementById('btn-text');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('error-message');
    const resultArea = document.getElementById('result-area');
    const initialText = document.getElementById('initial-text');
    const predictionDisplay = document.getElementById('prediction-display');
    const numPlaysInput = document.getElementById('num-plays-input');

    // --- API Configuration ---
    const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

    // --- Event Listener ---
    generateBtn.addEventListener('click', fetchPrediction);

    // --- Core Functions ---
    async function fetchPrediction() {
        setLoadingState(true);
        clearPreviousState();

        const numPlays = numPlaysInput.value || 5;
        const API_URL = `${API_BASE_URL}/predict-multiple?count=${numPlays}`;

        try {
            const response = await fetch(API_URL);
            if (!response.ok) {
                // If response is not ok, we try to parse the error message from the body
                const errorData = await response.json().catch(() => null); // Gracefully handle non-JSON responses
                if (errorData && errorData.detail) {
                    // Handle FastAPI's structured validation errors
                    if (Array.isArray(errorData.detail) && errorData.detail[0] && errorData.detail[0].msg) {
                        throw new Error(errorData.detail[0].msg);
                    }
                    throw new Error(errorData.detail);
                }
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            displayPrediction(data.predictions);
        } catch (error) {
            console.error('Fetch error:', error);
            displayError(error.message || 'Failed to generate plays. Please try again.');
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
            btnText.textContent = 'Generate Plays';
            loader.classList.add('hidden');
        }
    }

    function displayPrediction(predictions) {
        initialText.classList.add('hidden');
        predictionDisplay.classList.remove('hidden');

        if (!predictions || predictions.length === 0) {
            displayError('No predictions were generated.');
            return;
        }

        const playsHTML = predictions.map(numbers => {
            const whiteBalls = numbers.slice(0, 5).map(num => `<div class="bg-white text-gray-800 rounded-full w-12 h-12 flex items-center justify-center font-bold text-xl mx-1">${num}</div>`).join('');
            const powerball = `<div class="bg-red-500 text-white rounded-full w-12 h-12 flex items-center justify-center font-bold text-xl mx-1">${numbers[5]}</div>`;
            
            return `
                <div class="flex items-center justify-center flex-wrap mb-4">
                    ${whiteBalls}
                    <div class="border-l-2 border-gray-300 dark:border-gray-600 h-8 mx-2"></div>
                    ${powerball}
                </div>
            `;
        }).join('');

        predictionDisplay.innerHTML = `<div class="flex flex-col">${playsHTML}</div>`;
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