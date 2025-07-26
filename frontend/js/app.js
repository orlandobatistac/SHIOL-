document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const generateBtn = document.getElementById('generate-btn');
    const btnText = document.getElementById('btn-text');
    const loader = document.getElementById('loader');
    const resultArea = document.getElementById('result-area');
    const numPlaysInput = document.getElementById('num-plays-input');
    const copyBtn = document.getElementById('copy-btn');
    const toastNotification = document.getElementById('toast-notification');
    
    // New deterministic system elements
    const methodToggle = document.getElementById('method-toggle');
    const detailedBtn = document.getElementById('detailed-btn');
    const compareBtn = document.getElementById('compare-btn');
    const historyBtn = document.getElementById('history-btn');
    const diverseBtn = document.getElementById('diverse-btn');
    const scoreDetails = document.getElementById('score-details');
    const traceabilityInfo = document.getElementById('traceability-info');

    // Pipeline dashboard elements
    const pipelineStatusIndicator = document.getElementById('pipeline-status-indicator');
    const pipelineStatusText = document.getElementById('pipeline-status-text');
    const pipelineStatusDescription = document.getElementById('pipeline-status-description');
    const nextExecutionTime = document.getElementById('next-execution-time');
    const nextExecutionCountdown = document.getElementById('next-execution-countdown');
    const systemHealthIndicator = document.getElementById('system-health-indicator');
    const systemHealthText = document.getElementById('system-health-text');
    const systemHealthDetails = document.getElementById('system-health-details');
    const triggerPipelineBtn = document.getElementById('trigger-pipeline-btn');
    const viewLogsBtn = document.getElementById('view-logs-btn');
    const refreshPipelineBtn = document.getElementById('refresh-pipeline-btn');
    const autoRefreshCheckbox = document.getElementById('auto-refresh-checkbox');
    const executionHistoryTbody = document.getElementById('execution-history-tbody');
    const lastGeneratedPlays = document.getElementById('last-generated-plays');
    const lastPlaysContainer = document.getElementById('last-plays-container');
    const lastPlaysTimestamp = document.getElementById('last-plays-timestamp');
    const pipelineTriggerModal = document.getElementById('pipeline-trigger-modal');
    const cancelTriggerBtn = document.getElementById('cancel-trigger-btn');
    const confirmTriggerBtn = document.getElementById('confirm-trigger-btn');
    const pipelineNotification = document.getElementById('pipeline-notification');
    const pipelineNotificationIcon = document.getElementById('pipeline-notification-icon');
    const pipelineNotificationText = document.getElementById('pipeline-notification-text');

    // --- API Configuration ---
    // Enhanced dynamic API URL configuration with automatic detection
    function getApiBaseUrl() {
        // Use current origin for API calls - automatically adapts to any server IP
        const baseUrl = window.location.origin + '/api/v1';
        
        // Log the detected configuration for debugging
        console.log('API Base URL detected:', baseUrl);
        console.log('Current location:', {
            protocol: window.location.protocol,
            hostname: window.location.hostname,
            port: window.location.port,
            origin: window.location.origin
        });
        
        return baseUrl;
    }
    
    const API_BASE_URL = getApiBaseUrl();

    let lastPredictions = [];
    let lastPredictionData = null;
    let currentMethod = 'traditional';
    let autoRefreshInterval = null;
    let countdownInterval = null;

    // --- Event Listeners ---
    generateBtn.addEventListener('click', fetchPrediction);
    copyBtn.addEventListener('click', copyToClipboard);
    
    // New event listeners for deterministic features
    if (methodToggle) {
        methodToggle.addEventListener('change', handleMethodToggle);
    }
    if (detailedBtn) {
        detailedBtn.addEventListener('click', fetchDetailedPrediction);
    }
    if (compareBtn) {
        compareBtn.addEventListener('click', compareMethods);
    }
    if (historyBtn) {
        historyBtn.addEventListener('click', showPredictionHistory);
    }
    if (diverseBtn) {
        diverseBtn.addEventListener('click', fetchDiversePredictions);
    }

    // Pipeline dashboard event listeners
    if (triggerPipelineBtn) {
        triggerPipelineBtn.addEventListener('click', showTriggerModal);
    }
    if (viewLogsBtn) {
        viewLogsBtn.addEventListener('click', viewPipelineLogs);
    }
    if (refreshPipelineBtn) {
        refreshPipelineBtn.addEventListener('click', refreshPipelineStatus);
    }
    if (autoRefreshCheckbox) {
        autoRefreshCheckbox.addEventListener('change', handleAutoRefreshToggle);
    }
    if (cancelTriggerBtn) {
        cancelTriggerBtn.addEventListener('click', hideTriggerModal);
    }
    if (confirmTriggerBtn) {
        confirmTriggerBtn.addEventListener('click', triggerPipeline);
    }
    
    // Close modal when clicking outside
    if (pipelineTriggerModal) {
        pipelineTriggerModal.addEventListener('click', (e) => {
            if (e.target === pipelineTriggerModal) {
                hideTriggerModal();
            }
        });
    }

    // --- Core Functions ---
    async function fetchPrediction() {
        setLoadingState(true);
        clearPreviousState();

        const numPlays = numPlaysInput.value || 1;
        const isDeterministic = methodToggle ? methodToggle.checked : false;
        
        let API_URL;
        if (numPlays > 1) {
            // Multiple predictions - use existing endpoint
            API_URL = `${API_BASE_URL}/predict-multiple?count=${numPlays}`;
        } else {
            // Single prediction - use new endpoint with method selection
            API_URL = `${API_BASE_URL}/predict?deterministic=${isDeterministic}`;
        }

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
            
            if (data.predictions) {
                // Multiple predictions response
                lastPredictions = data.predictions;
                lastPredictionData = null;
                displayPredictions(lastPredictions);
            } else if (data.prediction) {
                // Single prediction response
                lastPredictions = [data.prediction];
                lastPredictionData = data;
                displayPredictions(lastPredictions, data);
                
                // Show additional info for deterministic predictions
                if (data.method === 'deterministic') {
                    displayTraceabilityInfo(data);
                }
            }
            
            copyBtn.classList.remove('hidden');
        } catch (error) {
            console.error('Fetch error:', error);
            displayError(error.message || 'Failed to generate plays. Please try again.');
        } finally {
            setLoadingState(false);
        }
    }

    async function fetchDetailedPrediction() {
        setLoadingState(true, 'Getting detailed prediction...');
        clearPreviousState();

        const isDeterministic = methodToggle ? methodToggle.checked : true;
        const API_URL = `${API_BASE_URL}/predict-detailed?deterministic=${isDeterministic}`;

        try {
            const response = await fetch(API_URL);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            
            lastPredictions = [data.prediction];
            lastPredictionData = data;
            displayPredictions(lastPredictions, data);
            
            if (data.method === 'deterministic') {
                displayDetailedScores(data);
                displayTraceabilityInfo(data);
            }
            
            copyBtn.classList.remove('hidden');
        } catch (error) {
            console.error('Detailed prediction error:', error);
            displayError(error.message || 'Failed to get detailed prediction.');
        } finally {
            setLoadingState(false);
        }
    }

    async function compareMethods() {
        setLoadingState(true, 'Comparing methods...');
        clearPreviousState();

        const API_URL = `${API_BASE_URL}/compare-methods`;

        try {
            const response = await fetch(API_URL);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            
            displayMethodComparison(data.comparison);
            copyBtn.classList.add('hidden'); // Hide copy for comparison view
        } catch (error) {
            console.error('Method comparison error:', error);
            displayError(error.message || 'Failed to compare methods.');
        } finally {
            setLoadingState(false);
        }
    }

    async function showPredictionHistory() {
        setLoadingState(true, 'Loading history...');
        clearPreviousState();

        const API_URL = `${API_BASE_URL}/prediction-history?limit=10`;

        try {
            const response = await fetch(API_URL);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            
            displayPredictionHistory(data.history);
            copyBtn.classList.add('hidden'); // Hide copy for history view
        } catch (error) {
            console.error('History error:', error);
            displayError(error.message || 'Failed to load prediction history.');
        } finally {
            setLoadingState(false);
        }
    }

    async function fetchDiversePredictions() {
        setLoadingState(true, 'Generating 5 diverse plays...');
        clearPreviousState();

        const API_URL = `${API_BASE_URL}/predict-diverse?num_plays=5`;

        try {
            const response = await fetch(API_URL);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            
            // Convert plays to the expected format for display
            const predictions = data.plays.map(play => play.prediction);
            lastPredictions = predictions;
            lastPredictionData = data;
            
            displayDiversePredictions(data);
            copyBtn.classList.remove('hidden');
            
        } catch (error) {
            console.error('Diverse predictions error:', error);
            displayError(error.message || 'Failed to generate diverse predictions.');
        } finally {
            setLoadingState(false);
        }
    }

    function displayDiversePredictions(data) {
        if (!data.plays || data.plays.length === 0) {
            displayError('No diverse predictions were generated.');
            return;
        }

        const playsHTML = data.plays.map((play, index) => {
            const whiteBalls = play.numbers.map(num => `
                <div class="w-12 h-12 flex items-center justify-center bg-gray-200 dark:bg-gray-700 rounded-full text-lg font-semibold text-gray-800 dark:text-white">${num}</div>
            `).join('');

            const powerball = `
                <div class="w-12 h-12 flex items-center justify-center bg-red-500 rounded-full text-lg font-semibold text-white">${play.powerball}</div>
            `;
            
            const methodBadge = `
                <div class="text-center mb-2">
                    <span class="inline-block px-3 py-1 text-xs font-semibold text-white bg-purple-500 rounded-full">
                        DIVERSE PLAY ${index + 1}
                    </span>
                </div>
            `;
            
            const scoreInfo = `
                <div class="text-center mt-2">
                    <span class="text-sm text-gray-600 dark:text-gray-400">
                        Score: ${play.score_total.toFixed(4)} | Rank: ${play.play_rank}
                    </span>
                </div>
            `;
            
            return `
                <div class="prediction-item mb-6 p-4 border border-purple-200 dark:border-purple-700 rounded-lg">
                    ${methodBadge}
                    <div class="flex items-center justify-center space-x-2 mb-2">
                        ${whiteBalls}
                        ${powerball}
                    </div>
                    ${scoreInfo}
                </div>
            `;
        }).join('');

        const summaryHTML = `
            <div class="mb-6 p-4 bg-purple-50 dark:bg-purple-900 rounded-lg">
                <h3 class="text-lg font-semibold mb-2 text-purple-800 dark:text-purple-200">5 Diverse High-Quality Plays</h3>
                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <span class="font-medium text-purple-700 dark:text-purple-300">Method:</span>
                        <span class="text-purple-600 dark:text-purple-400">Diverse Deterministic</span>
                    </div>
                    <div>
                        <span class="font-medium text-purple-700 dark:text-purple-300">Algorithm:</span>
                        <span class="text-purple-600 dark:text-purple-400">${data.generation_summary.diversity_algorithm}</span>
                    </div>
                    <div>
                        <span class="font-medium text-purple-700 dark:text-purple-300">Score Range:</span>
                        <span class="text-purple-600 dark:text-purple-400">
                            ${data.generation_summary.score_range.lowest.toFixed(4)} - ${data.generation_summary.score_range.highest.toFixed(4)}
                        </span>
                    </div>
                    <div>
                        <span class="font-medium text-purple-700 dark:text-purple-300">Candidates Evaluated:</span>
                        <span class="text-purple-600 dark:text-purple-400">${data.candidates_evaluated}</span>
                    </div>
                </div>
            </div>
        `;

        resultArea.innerHTML = summaryHTML + `<div class="flex flex-col">${playsHTML}</div>`;
    }

    function handleMethodToggle() {
        currentMethod = methodToggle.checked ? 'deterministic' : 'traditional';
        
        // Update UI to show/hide deterministic features
        const deterministicFeatures = document.querySelectorAll('.deterministic-feature');
        deterministicFeatures.forEach(element => {
            if (methodToggle.checked) {
                element.classList.remove('hidden');
            } else {
                element.classList.add('hidden');
            }
        });
        
        // Update button text
        updateButtonText();
    }

    function updateButtonText() {
        if (btnText) {
            const method = methodToggle && methodToggle.checked ? 'Deterministic' : 'Traditional';
            btnText.textContent = `Generate ${method} Plays`;
        }
    }

    function displayPredictions(predictions, additionalData = null) {
        if (!predictions || predictions.length === 0) {
            displayError('No predictions were generated.');
            return;
        }

        const playsHTML = predictions.map((numbers, index) => {
            const whiteBalls = numbers.slice(0, 5).map(num => `
                <div class="w-12 h-12 flex items-center justify-center bg-gray-200 dark:bg-gray-700 rounded-full text-lg font-semibold text-gray-800 dark:text-white">${num}</div>
            `).join('');

            const powerball = `
                <div class="w-12 h-12 flex items-center justify-center bg-red-500 rounded-full text-lg font-semibold text-white">${numbers[5]}</div>
            `;
            
            let methodBadge = '';
            let scoreInfo = '';
            
            if (additionalData && additionalData.method) {
                const badgeColor = additionalData.method === 'deterministic' ? 'bg-blue-500' : 'bg-green-500';
                methodBadge = `
                    <div class="text-center mb-2">
                        <span class="inline-block px-3 py-1 text-xs font-semibold text-white ${badgeColor} rounded-full">
                            ${additionalData.method.toUpperCase()}
                        </span>
                    </div>
                `;
                
                if (additionalData.score_total !== undefined) {
                    scoreInfo = `
                        <div class="text-center mt-2">
                            <span class="text-sm text-gray-600 dark:text-gray-400">
                                Score: ${additionalData.score_total.toFixed(4)}
                            </span>
                        </div>
                    `;
                }
            }
            
            return `
                <div class="prediction-item mb-6">
                    ${methodBadge}
                    <div class="flex items-center justify-center space-x-2 mb-2">
                        ${whiteBalls}
                        ${powerball}
                    </div>
                    ${scoreInfo}
                </div>
            `;
        }).join('');

        resultArea.innerHTML = `<div class="flex flex-col">${playsHTML}</div>`;
    }

    function displayDetailedScores(data) {
        if (!data.component_scores || !scoreDetails) return;
        
        const scores = data.component_scores;
        const weights = data.score_weights;
        
        const scoresHTML = `
            <div class="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <h3 class="text-lg font-semibold mb-4 text-gray-800 dark:text-white">Detailed Scoring</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div class="score-component">
                        <div class="flex justify-between items-center mb-1">
                            <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Probability</span>
                            <span class="text-sm text-gray-600 dark:text-gray-400">${(weights.probability * 100)}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-blue-500 h-2 rounded-full" style="width: ${scores.probability_score * 100}%"></div>
                        </div>
                        <span class="text-xs text-gray-500">${scores.probability_score.toFixed(4)}</span>
                    </div>
                    
                    <div class="score-component">
                        <div class="flex justify-between items-center mb-1">
                            <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Diversity</span>
                            <span class="text-sm text-gray-600 dark:text-gray-400">${(weights.diversity * 100)}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-green-500 h-2 rounded-full" style="width: ${scores.diversity_score * 100}%"></div>
                        </div>
                        <span class="text-xs text-gray-500">${scores.diversity_score.toFixed(4)}</span>
                    </div>
                    
                    <div class="score-component">
                        <div class="flex justify-between items-center mb-1">
                            <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Historical</span>
                            <span class="text-sm text-gray-600 dark:text-gray-400">${(weights.historical * 100)}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-yellow-500 h-2 rounded-full" style="width: ${scores.historical_score * 100}%"></div>
                        </div>
                        <span class="text-xs text-gray-500">${scores.historical_score.toFixed(4)}</span>
                    </div>
                    
                    <div class="score-component">
                        <div class="flex justify-between items-center mb-1">
                            <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Risk Adjusted</span>
                            <span class="text-sm text-gray-600 dark:text-gray-400">${(weights.risk_adjusted * 100)}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-red-500 h-2 rounded-full" style="width: ${scores.risk_adjusted_score * 100}%"></div>
                        </div>
                        <span class="text-xs text-gray-500">${scores.risk_adjusted_score.toFixed(4)}</span>
                    </div>
                </div>
                
                <div class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <div class="flex justify-between items-center">
                        <span class="font-semibold text-gray-800 dark:text-white">Total Score:</span>
                        <span class="font-bold text-lg text-blue-600 dark:text-blue-400">${data.total_score.toFixed(4)}</span>
                    </div>
                </div>
            </div>
        `;
        
        scoreDetails.innerHTML = scoresHTML;
        scoreDetails.classList.remove('hidden');
    }

    function displayTraceabilityInfo(data) {
        if (!data.traceability || !traceabilityInfo) return;
        
        const traceability = data.traceability;
        const timestamp = new Date(traceability.timestamp).toLocaleString();
        
        const traceabilityHTML = `
            <div class="mt-4 p-3 bg-blue-50 dark:bg-blue-900 rounded-lg">
                <h4 class="text-sm font-semibold mb-2 text-blue-800 dark:text-blue-200">Traceability Information</h4>
                <div class="grid grid-cols-2 gap-2 text-xs">
                    <div>
                        <span class="font-medium text-blue-700 dark:text-blue-300">Dataset Hash:</span>
                        <span class="text-blue-600 dark:text-blue-400 font-mono">${traceability.dataset_hash}</span>
                    </div>
                    <div>
                        <span class="font-medium text-blue-700 dark:text-blue-300">Model Version:</span>
                        <span class="text-blue-600 dark:text-blue-400">${traceability.model_version}</span>
                    </div>
                    <div>
                        <span class="font-medium text-blue-700 dark:text-blue-300">Generated:</span>
                        <span class="text-blue-600 dark:text-blue-400">${timestamp}</span>
                    </div>
                    <div>
                        <span class="font-medium text-blue-700 dark:text-blue-300">Candidates:</span>
                        <span class="text-blue-600 dark:text-blue-400">${traceability.candidates_evaluated}</span>
                    </div>
                </div>
            </div>
        `;
        
        traceabilityInfo.innerHTML = traceabilityHTML;
        traceabilityInfo.classList.remove('hidden');
    }

    function displayMethodComparison(comparison) {
        const comparisonHTML = `
            <div class="method-comparison">
                <h3 class="text-xl font-bold mb-6 text-center text-gray-800 dark:text-white">Method Comparison</h3>
                
                <div class="grid md:grid-cols-2 gap-6">
                    <!-- Traditional Method -->
                    <div class="method-card p-4 bg-green-50 dark:bg-green-900 rounded-lg border border-green-200 dark:border-green-700">
                        <div class="text-center mb-4">
                            <span class="inline-block px-3 py-1 text-sm font-semibold text-white bg-green-500 rounded-full mb-2">
                                TRADITIONAL
                            </span>
                            <p class="text-sm text-green-700 dark:text-green-300">${comparison.traditional.description}</p>
                        </div>
                        
                        <div class="prediction-display mb-4">
                            <div class="flex items-center justify-center space-x-2">
                                ${comparison.traditional.prediction.slice(0, 5).map(num =>
                                    `<div class="w-10 h-10 flex items-center justify-center bg-gray-200 dark:bg-gray-700 rounded-full text-sm font-semibold text-gray-800 dark:text-white">${num}</div>`
                                ).join('')}
                                <div class="w-10 h-10 flex items-center justify-center bg-red-500 rounded-full text-sm font-semibold text-white">${comparison.traditional.prediction[5]}</div>
                            </div>
                        </div>
                        
                        <div class="characteristics">
                            <h4 class="font-semibold text-green-800 dark:text-green-200 mb-2">Characteristics:</h4>
                            <ul class="text-xs text-green-700 dark:text-green-300 space-y-1">
                                ${comparison.traditional.characteristics.map(char => `<li>• ${char}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                    
                    <!-- Deterministic Method -->
                    <div class="method-card p-4 bg-blue-50 dark:bg-blue-900 rounded-lg border border-blue-200 dark:border-blue-700">
                        <div class="text-center mb-4">
                            <span class="inline-block px-3 py-1 text-sm font-semibold text-white bg-blue-500 rounded-full mb-2">
                                DETERMINISTIC
                            </span>
                            <p class="text-sm text-blue-700 dark:text-blue-300">${comparison.deterministic.description}</p>
                        </div>
                        
                        <div class="prediction-display mb-4">
                            <div class="flex items-center justify-center space-x-2">
                                ${comparison.deterministic.prediction.slice(0, 5).map(num =>
                                    `<div class="w-10 h-10 flex items-center justify-center bg-gray-200 dark:bg-gray-700 rounded-full text-sm font-semibold text-gray-800 dark:text-white">${num}</div>`
                                ).join('')}
                                <div class="w-10 h-10 flex items-center justify-center bg-red-500 rounded-full text-sm font-semibold text-white">${comparison.deterministic.prediction[5]}</div>
                            </div>
                            <div class="text-center mt-2">
                                <span class="text-sm font-semibold text-blue-600 dark:text-blue-400">
                                    Score: ${comparison.deterministic.score_total.toFixed(4)}
                                </span>
                            </div>
                        </div>
                        
                        <div class="characteristics mb-4">
                            <h4 class="font-semibold text-blue-800 dark:text-blue-200 mb-2">Characteristics:</h4>
                            <ul class="text-xs text-blue-700 dark:text-blue-300 space-y-1">
                                ${comparison.deterministic.characteristics.map(char => `<li>• ${char}</li>`).join('')}
                            </ul>
                        </div>
                        
                        <div class="traceability text-xs">
                            <h4 class="font-semibold text-blue-800 dark:text-blue-200 mb-1">Traceability:</h4>
                            <div class="text-blue-700 dark:text-blue-300">
                                <div>Hash: ${comparison.deterministic.traceability.dataset_hash}</div>
                                <div>Version: ${comparison.deterministic.traceability.model_version}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        resultArea.innerHTML = comparisonHTML;
    }

    function displayPredictionHistory(history) {
        if (!history || history.length === 0) {
            displayError('No prediction history available.');
            return;
        }
        
        const historyHTML = `
            <div class="prediction-history">
                <h3 class="text-xl font-bold mb-6 text-center text-gray-800 dark:text-white">Prediction History</h3>
                
                <div class="space-y-4 max-h-96 overflow-y-auto">
                    ${history.map(pred => {
                        const timestamp = new Date(pred.timestamp).toLocaleString();
                        return `
                            <div class="history-item p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border">
                                <div class="flex items-center justify-between mb-2">
                                    <div class="flex items-center space-x-2">
                                        ${[pred.n1, pred.n2, pred.n3, pred.n4, pred.n5].map(num =>
                                            `<div class="w-8 h-8 flex items-center justify-center bg-gray-200 dark:bg-gray-700 rounded-full text-xs font-semibold text-gray-800 dark:text-white">${num}</div>`
                                        ).join('')}
                                        <div class="w-8 h-8 flex items-center justify-center bg-red-500 rounded-full text-xs font-semibold text-white">${pred.powerball}</div>
                                    </div>
                                    <div class="text-right">
                                        <div class="text-sm font-semibold text-gray-800 dark:text-white">
                                            Score: ${pred.score_total.toFixed(4)}
                                        </div>
                                        <div class="text-xs text-gray-600 dark:text-gray-400">
                                            ${timestamp}
                                        </div>
                                    </div>
                                </div>
                                <div class="text-xs text-gray-500 dark:text-gray-400">
                                    Hash: ${pred.dataset_hash} | Version: ${pred.model_version}
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
        
        resultArea.innerHTML = historyHTML;
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
    function setLoadingState(isLoading, customText = null) {
        generateBtn.disabled = isLoading;
        if (detailedBtn) detailedBtn.disabled = isLoading;
        if (compareBtn) compareBtn.disabled = isLoading;
        if (historyBtn) historyBtn.disabled = isLoading;
        
        if (isLoading) {
            btnText.textContent = customText || 'Generating...';
            loader.classList.remove('hidden');
        } else {
            updateButtonText();
            loader.classList.add('hidden');
        }
    }
    
    function displayError(message) {
        resultArea.innerHTML = `<p class="text-red-500">${message}</p>`;
    }

    function clearPreviousState() {
        resultArea.innerHTML = '';
        copyBtn.classList.add('hidden');
        
        // Clear deterministic-specific elements
        if (scoreDetails) {
            scoreDetails.innerHTML = '';
            scoreDetails.classList.add('hidden');
        }
        if (traceabilityInfo) {
            traceabilityInfo.innerHTML = '';
            traceabilityInfo.classList.add('hidden');
        }
    }

    function showToast() {
        toastNotification.classList.add('opacity-100');
        setTimeout(() => {
            toastNotification.classList.remove('opacity-100');
        }, 3000);
    }

    // --- Pipeline Dashboard Functions ---
    async function fetchPipelineStatus() {
        try {
            const response = await fetch(`${API_BASE_URL}/pipeline/status`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            updatePipelineStatus(data);
            return data;
        } catch (error) {
            console.error('Pipeline status fetch error:', error);
            showPipelineError('Failed to fetch pipeline status');
            return null;
        }
    }

    function updatePipelineStatus(data) {
        if (!data) return;

        // Extract pipeline status data
        const pipelineStatus = data.pipeline_status || {};
        const status = pipelineStatus.current_status || 'unknown';

        // Update current status
        if (pipelineStatusIndicator && pipelineStatusText && pipelineStatusDescription) {
            pipelineStatusIndicator.className = `w-3 h-3 rounded-full status-${status}`;
            pipelineStatusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            pipelineStatusDescription.textContent = `Pipeline is currently ${status}`;
        }

        // Update next execution time
        if (nextExecutionTime && nextExecutionCountdown) {
            if (pipelineStatus.next_scheduled_execution) {
                const nextTime = new Date(pipelineStatus.next_scheduled_execution);
                nextExecutionTime.textContent = nextTime.toLocaleTimeString();
                updateCountdown(nextTime);
            } else {
                nextExecutionTime.textContent = 'Not scheduled';
                nextExecutionCountdown.textContent = 'Manual execution only';
            }
        }

        // Update system health
        if (systemHealthIndicator && systemHealthText && systemHealthDetails) {
            const health = pipelineStatus.system_health || {};
            let healthStatus = 'unknown';
            let healthDetails = 'No health information available';
            
            // Determine health status based on system metrics
            if (health.cpu_usage_percent !== undefined) {
                if (health.cpu_usage_percent < 80 && health.memory_usage_percent < 85 && health.disk_usage_percent < 90) {
                    healthStatus = 'healthy';
                    healthDetails = `CPU: ${health.cpu_usage_percent}%, Memory: ${health.memory_usage_percent}%, Disk: ${health.disk_usage_percent}%`;
                } else {
                    healthStatus = 'degraded';
                    healthDetails = `High resource usage - CPU: ${health.cpu_usage_percent}%, Memory: ${health.memory_usage_percent}%`;
                }
            }
            
            systemHealthIndicator.className = `w-3 h-3 rounded-full health-${healthStatus}`;
            systemHealthText.textContent = healthStatus.charAt(0).toUpperCase() + healthStatus.slice(1);
            systemHealthDetails.textContent = healthDetails;
        }

        // Update execution history
        if (pipelineStatus.recent_execution_history) {
            updateExecutionHistory(pipelineStatus.recent_execution_history);
        }

        // Update last generated plays
        if (data.generated_plays_last_run && data.generated_plays_last_run.length > 0) {
            updateLastGeneratedPlays({
                plays: data.generated_plays_last_run.map(play => [...play.numbers, play.powerball]),
                scores: data.generated_plays_last_run.map(play => play.score),
                timestamp: data.generated_plays_last_run[0].timestamp
            });
        }

        // Update trigger button state
        if (triggerPipelineBtn) {
            const canTrigger = status !== 'running';
            triggerPipelineBtn.disabled = !canTrigger;
        }
    }

    function updateCountdown(targetTime) {
        if (countdownInterval) {
            clearInterval(countdownInterval);
        }

        countdownInterval = setInterval(() => {
            const now = new Date();
            const diff = targetTime - now;

            if (diff <= 0) {
                nextExecutionCountdown.textContent = 'Execution due';
                clearInterval(countdownInterval);
                return;
            }

            const hours = Math.floor(diff / (1000 * 60 * 60));
            const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((diff % (1000 * 60)) / 1000);

            if (hours > 0) {
                nextExecutionCountdown.textContent = `in ${hours}h ${minutes}m`;
            } else if (minutes > 0) {
                nextExecutionCountdown.textContent = `in ${minutes}m ${seconds}s`;
            } else {
                nextExecutionCountdown.textContent = `in ${seconds}s`;
            }
        }, 1000);
    }

    function updateExecutionHistory(executions) {
        if (!executionHistoryTbody) return;

        if (!executions || executions.length === 0) {
            executionHistoryTbody.innerHTML = `
                <tr>
                    <td colspan="5" class="px-4 py-8 text-center text-gray-500 dark:text-gray-400">
                        No execution history available
                    </td>
                </tr>
            `;
            return;
        }

        const historyHTML = executions.map(execution => {
            const startTime = execution.start_time ? new Date(execution.start_time).toLocaleString() : 'N/A';
            const endTime = execution.end_time ? new Date(execution.end_time) : null;
            const startTimeObj = execution.start_time ? new Date(execution.start_time) : null;
            
            let duration = 'N/A';
            if (startTimeObj && endTime) {
                const durationMs = endTime - startTimeObj;
                duration = `${Math.round(durationMs / 1000)}s`;
            }
            
            const status = execution.status || 'unknown';
            const executionId = execution.execution_id || 'unknown';

            return `
                <tr>
                    <td class="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">${startTime}</td>
                    <td class="px-4 py-3">
                        <span class="status-badge ${status}">${status}</span>
                    </td>
                    <td class="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">${duration}</td>
                    <td class="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">${execution.steps_completed || 0}/${execution.total_steps || 7}</td>
                    <td class="px-4 py-3">
                        <button onclick="viewExecutionDetails('${executionId}')"
                                class="text-indigo-600 hover:text-indigo-900 dark:text-indigo-400 dark:hover:text-indigo-300 text-sm font-medium">
                            View Details
                        </button>
                    </td>
                </tr>
            `;
        }).join('');

        executionHistoryTbody.innerHTML = historyHTML;
    }

    function updateLastGeneratedPlays(lastRun) {
        if (!lastGeneratedPlays || !lastPlaysContainer || !lastRun.plays) return;

        const timestamp = new Date(lastRun.timestamp).toLocaleString();
        lastPlaysTimestamp.textContent = `Generated: ${timestamp}`;

        // Display all 5 plays (not just 3)
        const playsHTML = lastRun.plays.slice(0, 5).map((play, index) => {
            const whiteBalls = play.slice(0, 5).map(num => `
                <div class="w-8 h-8 flex items-center justify-center bg-gray-200 dark:bg-gray-700 rounded-full text-sm font-semibold text-gray-800 dark:text-white">${num}</div>
            `).join('');

            const powerball = `
                <div class="w-8 h-8 flex items-center justify-center bg-red-500 rounded-full text-sm font-semibold text-white">${play[5]}</div>
            `;

            return `
                <div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg mb-2">
                    <div class="flex items-center space-x-2">
                        <span class="text-sm font-medium text-gray-600 dark:text-gray-400">Play ${index + 1}:</span>
                        ${whiteBalls}
                        ${powerball}
                    </div>
                    <div class="text-xs text-gray-500 dark:text-gray-400">
                        Score: ${lastRun.scores && lastRun.scores[index] ? lastRun.scores[index].toFixed(4) : 'N/A'}
                    </div>
                </div>
            `;
        }).join('');

        lastPlaysContainer.innerHTML = playsHTML;
        lastGeneratedPlays.classList.remove('hidden');
    }

    async function triggerPipeline() {
        hideTriggerModal();
        
        try {
            triggerPipelineBtn.disabled = true;
            triggerPipelineBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Triggering...';
            
            const response = await fetch(`${API_BASE_URL}/pipeline/trigger`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            showPipelineNotification('Pipeline triggered successfully!', 'success');
            
            // Refresh status after a short delay
            setTimeout(() => {
                refreshPipelineStatus();
            }, 2000);

        } catch (error) {
            console.error('Pipeline trigger error:', error);
            showPipelineNotification('Failed to trigger pipeline', 'error');
        } finally {
            triggerPipelineBtn.disabled = false;
            triggerPipelineBtn.innerHTML = '<i class="fas fa-play mr-2"></i>Trigger Pipeline';
        }
    }

    function showTriggerModal() {
        if (pipelineTriggerModal) {
            pipelineTriggerModal.classList.remove('hidden');
        }
    }

    function hideTriggerModal() {
        if (pipelineTriggerModal) {
            pipelineTriggerModal.classList.add('hidden');
        }
    }

    async function viewPipelineLogs() {
        try {
            const response = await fetch(`${API_BASE_URL}/pipeline/logs`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            
            // Create a simple modal to display logs
            const logsModal = document.createElement('div');
            logsModal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
            logsModal.innerHTML = `
                <div class="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-4xl w-full mx-4 max-h-96 overflow-hidden">
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="text-lg font-semibold text-gray-800 dark:text-white">Pipeline Logs</h3>
                        <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-y-auto max-h-80">
                        ${data.logs ? data.logs.replace(/\n/g, '<br>') : 'No logs available'}
                    </div>
                </div>
            `;
            document.body.appendChild(logsModal);
            
        } catch (error) {
            console.error('Logs fetch error:', error);
            showPipelineNotification('Failed to fetch logs', 'error');
        }
    }

    function refreshPipelineStatus() {
        if (refreshPipelineBtn) {
            const originalHTML = refreshPipelineBtn.innerHTML;
            refreshPipelineBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>Refreshing...';
            refreshPipelineBtn.disabled = true;
            
            fetchPipelineStatus().finally(() => {
                refreshPipelineBtn.innerHTML = originalHTML;
                refreshPipelineBtn.disabled = false;
            });
        }
    }

    function handleAutoRefreshToggle() {
        if (!autoRefreshCheckbox) return;

        if (autoRefreshCheckbox.checked) {
            // Start auto-refresh every 30 seconds
            autoRefreshInterval = setInterval(fetchPipelineStatus, 30000);
            showPipelineNotification('Auto-refresh enabled', 'info');
        } else {
            // Stop auto-refresh
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
            }
            showPipelineNotification('Auto-refresh disabled', 'info');
        }
    }

    function showPipelineNotification(message, type = 'info') {
        if (!pipelineNotification || !pipelineNotificationIcon || !pipelineNotificationText) return;

        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };

        pipelineNotificationIcon.className = icons[type] || icons.info;
        pipelineNotificationText.textContent = message;
        pipelineNotification.className = `fixed bottom-5 left-5 py-3 px-4 rounded-lg shadow-xl opacity-100 transition-opacity duration-300 z-50 ${type}`;

        setTimeout(() => {
            pipelineNotification.classList.remove('opacity-100');
            pipelineNotification.classList.add('opacity-0');
        }, 5000);
    }

    function showPipelineError(message) {
        if (pipelineStatusText) {
            pipelineStatusText.textContent = 'Error';
        }
        if (pipelineStatusDescription) {
            pipelineStatusDescription.textContent = message;
        }
        if (pipelineStatusIndicator) {
            pipelineStatusIndicator.className = 'w-3 h-3 rounded-full bg-red-500';
        }
    }

    // Global function for execution details (called from HTML)
    window.viewExecutionDetails = function(executionId) {
        // This would typically fetch detailed execution information
        showPipelineNotification(`Viewing details for execution ${executionId}`, 'info');
    };

    // Initialize pipeline dashboard
    function initializePipelineDashboard() {
        // Test API connectivity first
        testApiConnectivity().then(isConnected => {
            if (isConnected) {
                console.log('✅ API connectivity confirmed');
                // Initial status fetch
                fetchPipelineStatus();
                
                // Start auto-refresh if enabled
                if (autoRefreshCheckbox && autoRefreshCheckbox.checked) {
                    handleAutoRefreshToggle();
                }
            } else {
                console.warn('⚠️ API connectivity issues detected');
                showPipelineError('Unable to connect to API server. Check if server is running.');
            }
        });
    }

    // Test API connectivity
    async function testApiConnectivity() {
        try {
            const response = await fetch(`${API_BASE_URL}/pipeline/status`, {
                method: 'GET',
                timeout: 5000
            });
            return response.ok;
        } catch (error) {
            console.error('API connectivity test failed:', error);
            return false;
        }
    }

    // Initialize UI state
    updateButtonText();
    handleMethodToggle();
    initializePipelineDashboard();
});