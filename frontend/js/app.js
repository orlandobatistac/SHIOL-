document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const generateBtn = document.getElementById('generate-btn');
    const btnText = document.getElementById('btn-text');
    const loader = document.getElementById('loader');
    const resultArea = document.getElementById('result-area');
    const numPlaysInput = document.getElementById('num-plays-input');
    const copyBtn = document.getElementById('copy-btn');
    const toastNotification = document.getElementById('toast-notification');

    // --- API Configuration ---
    const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

    let lastPredictions = [];

    // --- Event Listeners ---
    generateBtn.addEventListener('click', fetchPrediction);
    copyBtn.addEventListener('click', copyToClipboard);

    // --- Core Functions ---
    async function fetchPrediction() {
        setLoadingState(true);
        clearPreviousState();

        const numPlays = numPlaysInput.value || 1;
        const API_URL = `${API_BASE_URL}/predict-multiple?count=${numPlays}`;

        try {
            const response = await fetch(API_URL);
            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                if (errorData && errorData.detail) {
                    if (Array.isArray(errorData.detail) && errorData.detail[0] && errorData.detail[0].msg) {
                        throw new Error(errorData.detail[0].msg);
                    }
                    throw new Error(errorData.detail);
                }
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            lastPredictions = data.predictions;
            displayPredictions(lastPredictions);
            copyBtn.classList.remove('hidden');
        } catch (error) {
            console.error('Fetch error:', error);
            displayError(error.message || 'Failed to generate plays. Please try again.');
        } finally {
            setLoadingState(false);
        }
    }

    function displayPredictions(predictions) {
        if (!predictions || predictions.length === 0) {
            displayError('No predictions were generated.');
            return;
        }

        const playsHTML = predictions.map(numbers => {
            const whiteBalls = numbers.slice(0, 5).map(num => `
                <div class="w-12 h-12 flex items-center justify-center bg-gray-200 dark:bg-gray-700 rounded-full text-lg font-semibold text-gray-800 dark:text-white">${num}</div>
            `).join('');

            const powerball = `
                <div class="w-12 h-12 flex items-center justify-center bg-red-500 rounded-full text-lg font-semibold text-white">${numbers[5]}</div>
            `;
            
            return `
                <div class="flex items-center justify-center space-x-2 mb-4">
                    ${whiteBalls}
                    ${powerball}
                </div>
            `;
        }).join('');

        resultArea.innerHTML = `<div class="flex flex-col">${playsHTML}</div>`;
    }

    function copyToClipboard() {
        if (lastPredictions.length === 0) return;

        const textToCopy = lastPredictions.map(play => {
            const whiteBalls = play.slice(0, 5).join(', ');
            const powerball = play[5];
            return `Numbers: ${whiteBalls} | Powerball: ${powerball}`;
        }).join('\n');

        navigator.clipboard.writeText(textToCopy).then(() => {
            showToast();
        }).catch(err => {
            console.error('Failed to copy numbers: ', err);
            displayError('Could not copy numbers to clipboard.');
        });
    }

    // --- UI State Management ---
    function setLoadingState(isLoading) {
        generateBtn.disabled = isLoading;
        if (isLoading) {
            btnText.textContent = 'Generating...';
            loader.classList.remove('hidden');
        } else {
            btnText.textContent = 'Generate Plays';
            loader.classList.add('hidden');
        }
    }
    
    function displayError(message) {
        resultArea.innerHTML = `<p class="text-red-500">${message}</p>`;
    }

    function clearPreviousState() {
        resultArea.innerHTML = '';
        copyBtn.classList.add('hidden');
    }

    function showToast() {
        toastNotification.classList.add('opacity-100');
        setTimeout(() => {
            toastNotification.classList.remove('opacity-100');
        }, 3000);
    }
});