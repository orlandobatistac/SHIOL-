/**
 * SHIOL+ Public Interface
 * =======================
 *
 * Main JavaScript for the public interface.
 * Handles API integration, UI updates, and user interactions.
 */

class PublicInterface {
    constructor() {
        this.countdownTimer = null;
        this.currentHistoryPage = 1;
        this.historyPerPage = 30;
        this.isLoading = false;

        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    /**
     * Initialize the public interface
     */
    async init() {
        console.log('Initializing SHIOL+ Public Interface');

        try {
            // Setup event listeners
            this.setupEventListeners();

            // Initialize countdown timer
            this.initializeCountdown();

            // Load initial data
            await this.loadInitialData();

            console.log('Public interface initialized successfully');
        } catch (error) {
            console.error('Error initializing public interface:', error);
            this.showError('Failed to initialize interface');
        }
    }

    /**
     * Setup event listeners for UI interactions
     */
    setupEventListeners() {
        // Admin login button
        const adminLoginBtn = document.getElementById('admin-login-btn');
        if (adminLoginBtn) {
            adminLoginBtn.addEventListener('click', () => this.showLoginModal());
        }

        // Login modal events
        const loginModal = document.getElementById('login-modal');
        const closeLoginModal = document.getElementById('close-login-modal');
        const cancelLogin = document.getElementById('cancel-login');
        const loginForm = document.getElementById('login-form');

        if (closeLoginModal) {
            closeLoginModal.addEventListener('click', () => this.hideLoginModal());
        }

        if (cancelLogin) {
            cancelLogin.addEventListener('click', () => this.hideLoginModal());
        }

        if (loginModal) {
            loginModal.addEventListener('click', (e) => {
                if (e.target === loginModal) {
                    this.hideLoginModal();
                }
            });
        }

        if (loginForm) {
            loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        }

        // Date details modal events
        const dateModal = document.getElementById('date-details-modal');
        const closeDateModal = document.getElementById('close-date-modal');
        const closeModalBtn = document.getElementById('close-modal-btn');

        if (closeDateModal) {
            closeDateModal.addEventListener('click', () => this.hideDateDetailsModal());
        }

        if (closeModalBtn) {
            closeModalBtn.addEventListener('click', () => this.hideDateDetailsModal());
        }

        if (dateModal) {
            dateModal.addEventListener('click', (e) => {
                if (e.target === dateModal) {
                    this.hideDateDetailsModal();
                }
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideLoginModal();
                this.hideDateDetailsModal();
            }
        });
    }

    /**
     * Initialize countdown timer
     */
    initializeCountdown() {
        this.countdownTimer = new CountdownTimer('countdown-timer', 'countdown-display');
    }

    /**
     * Load initial data when page loads
     */
    async loadInitialData() {
        await this.loadNextDrawingInfo();
        await this.loadGroupedPredictionHistory();
    }

    /**
     * Load Smart AI predictions from pipeline
     */
    async loadNextDrawingInfo() {
        try {
            console.log('Loading Smart AI predictions...');

            // Show loading state
            this.showLoadingState();

            // Load 100 Smart AI predictions from the smart endpoint
            const data = await PowerballUtils.apiRequest('/predict/smart?limit=100');

            // Update last updated timestamps
            const updateTime = new Date().toLocaleString();
            const footerLastUpdated = document.getElementById('footer-last-updated');
            if (footerLastUpdated) {
                footerLastUpdated.textContent = updateTime;
            }

            // Display Smart AI predictions
            if (data.smart_predictions && data.smart_predictions.length > 0) {
                // Debug: Log first prediction structure to see available fields
                console.log('First prediction structure:', data.smart_predictions[0]);
                console.log('Available score fields:', Object.keys(data.smart_predictions[0]).filter(key =>
                    key.toLowerCase().includes('score') ||
                    key.toLowerCase().includes('confidence') ||
                    key.toLowerCase().includes('probability')
                ));

                this.displaySmartPredictions(data.smart_predictions, data.next_drawing);
            } else {
                this.showPredictionError();
            }

            console.log('Smart AI predictions loaded successfully');
        } catch (error) {
            console.error('Error loading Smart AI predictions:', error);
            this.showPredictionError();
        }
    }

    /**
     * Display Smart AI predictions
     * @param {Object} data - Smart AI prediction data
     */
    displaySmartPredictions(predictions, nextDrawing) {
        const container = document.getElementById('predictions-container');
        const loading = document.getElementById('predictions-loading');
        const error = document.getElementById('predictions-error');

        if (!container || !loading) return;

        // Hide loading and error states
        loading.classList.add('hidden');
        if (error) error.classList.add('hidden');

        // Sort predictions by confidence score in descending order (best to worst)
        const sortedPredictions = (predictions || []).sort((a, b) => {
            // Try multiple possible score field names and ensure we get numeric values
            const getScore = (pred) => {
                const possibleScores = [
                    pred.total_score,
                    pred.score_total,
                    pred.confidence,
                    pred.score,
                    pred.ai_score,
                    pred.probability
                ];

                for (let score of possibleScores) {
                    if (typeof score === 'number' && !isNaN(score)) {
                        return score;
                    }
                    if (typeof score === 'string') {
                        const numScore = parseFloat(score);
                        if (!isNaN(numScore)) {
                            return numScore;
                        }
                    }
                }
                return 0; // Default fallback
            };

            const scoreA = getScore(a);
            const scoreB = getScore(b);

            return scoreB - scoreA; // Descending order (best to worst)
        });

        // Format next drawing information
        const nextDrawingInfo = nextDrawing ? `
            <div class="next-drawing-info">
                <div class="drawing-date">
                    <span class="date-label">Next Drawing:</span>
                    <span class="date-value">${nextDrawing.formatted_date}</span>
                    ${nextDrawing.is_today ? '<span class="today-badge">TODAY</span>' : ''}
                    ${nextDrawing.days_until > 0 ? `<span class="days-until">in ${nextDrawing.days_until} day${nextDrawing.days_until > 1 ? 's' : ''}</span>` : ''}
                </div>
            </div>
        ` : '';


        const predictionsHtml = `
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                <!-- Header -->
                <div class="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30 p-6 border-b border-gray-200 dark:border-gray-700">
                    <div class="flex items-center justify-between">
                        <div>
                            <h3 class="text-xl font-bold text-gray-900 dark:text-white">
                                <i class="fas fa-brain mr-2 text-blue-600"></i>
                                Smart AI Predictions
                            </h3>
                            ${nextDrawingInfo}
                            <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                                ${sortedPredictions.length} predictions ordered from highest to lowest AI confidence score
                            </p>
                        </div>
                        <div class="text-right">
                            <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">${sortedPredictions.length}</div>
                            <div class="text-xs text-gray-500 dark:text-gray-400">Total Plays</div>
                        </div>
                    </div>
                </div>

                <!-- Predictions Table -->
                <div class="overflow-x-auto max-h-96 overflow-y-auto">
                    <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead class="bg-gray-50 dark:bg-gray-900 sticky top-0">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Position</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Numbers & Powerball</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">AI Confidence</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                            ${sortedPredictions.map((pred, index) => {
                                const isTopFive = index < 5;
                                // Get confidence score using the same logic as sorting
                                const getDisplayScore = (pred) => {
                                    const possibleScores = [
                                        pred.total_score,
                                        pred.score_total,
                                        pred.confidence,
                                        pred.score,
                                        pred.ai_score,
                                        pred.probability
                                    ];

                                    for (let score of possibleScores) {
                                        if (typeof score === 'number' && !isNaN(score)) {
                                            return score;
                                        }
                                        if (typeof score === 'string') {
                                            const numScore = parseFloat(score);
                                            if (!isNaN(numScore)) {
                                                return numScore;
                                            }
                                        }
                                    }
                                    return 0; // Default fallback
                                };

                                const confidenceScore = getDisplayScore(pred);
                                const displayRank = index + 1; // Always use 1-based ranking after sorting

                                return `
                                    <tr class="hover:bg-gray-50 dark:hover:bg-gray-700 ${isTopFive ? 'bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20' : ''}">
                                        <td class="px-6 py-4 whitespace-nowrap">
                                            <div class="flex items-center">
                                                <span class="inline-flex items-center justify-center w-8 h-8 rounded-full ${isTopFive ? 'bg-blue-100 text-blue-800 border-2 border-blue-300' : 'bg-gray-100 text-gray-700 border-2 border-gray-300'} text-sm font-bold">
                                                    ${displayRank}
                                                </span>
                                                ${isTopFive ? '<span class="ml-2 text-xs font-semibold text-blue-600 uppercase tracking-wide">Top Pick</span>' : ''}
                                            </div>
                                        </td>
                                        <td class="px-6 py-4 whitespace-nowrap">
                                            <div class="flex items-center space-x-2">
                                                ${(pred.numbers || []).map(num => `
                                                    <span class="inline-flex items-center justify-center w-10 h-10 bg-white text-gray-900 rounded-full text-sm font-bold border-2 border-gray-300 shadow-sm">${num}</span>
                                                `).join('')}
                                                <span class="text-red-500 text-lg font-bold mx-2">•</span>
                                                <span class="inline-flex items-center justify-center w-10 h-10 bg-red-600 text-white rounded-full text-sm font-bold shadow-md">${pred.powerball || pred.pb || ''}</span>
                                            </div>
                                        </td>
                                        <td class="px-6 py-4 whitespace-nowrap">
                                            <div class="flex flex-col">
                                                <div class="text-lg font-bold text-gray-900 dark:text-white">
                                                    ${(confidenceScore * 100).toFixed(1)}%
                                                </div>
                                                <div class="text-xs text-gray-500 dark:text-gray-400">
                                                    ${isTopFive ? 'Premium Quality' : 'Standard Quality'}
                                                </div>
                                            </div>
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>

                <!-- How to Read This Table -->
                <div class="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg border border-blue-200 dark:border-blue-700">
                    <h4 class="text-sm font-semibold text-blue-800 dark:text-blue-200 mb-3">
                        <i class="fas fa-lightbulb text-blue-500 mr-2"></i>
                        How to Read This Table
                    </h4>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-blue-700 dark:text-blue-300">
                        <div class="space-y-2">
                            <div class="flex items-center">
                                <span class="w-6 h-6 bg-blue-100 text-blue-800 rounded-full text-xs font-bold flex items-center justify-center mr-2">1</span>
                                <span><strong>Position:</strong> Best predictions first</span>
                            </div>
                            <div class="flex items-center">
                                <i class="fas fa-star text-blue-500 mr-2"></i>
                                <span><strong>Top Pick:</strong> AI's most confident choices</span>
                            </div>
                        </div>
                        <div class="space-y-2">
                            <div class="flex items-center">
                                <span class="w-3 h-3 bg-blue-500 rounded-full mr-2"></span>
                                <span><strong>80-100%:</strong> Premium Quality</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                                <span><strong>60-80%:</strong> High Quality</span>
                            </div>
                        </div>
                        <div class="space-y-2">
                            <div class="flex items-center">
                                <span class="w-3 h-3 bg-yellow-500 rounded-full mr-2"></span>
                                <span><strong>40-60%:</strong> Good Quality</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-3 h-3 bg-gray-500 rounded-full mr-2"></span>
                                <span><strong>Below 40%:</strong> Standard</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Display the predictions
        container.innerHTML = predictionsHtml;
        container.classList.remove('hidden');

        // Ensure loading is hidden
        loading.classList.add('hidden');
    }

    /**
     * Show loading state
     */
    showLoadingState() {
        const loading = document.getElementById('predictions-loading');
        const container = document.getElementById('predictions-container');
        const error = document.getElementById('predictions-error');

        if (loading) loading.classList.remove('hidden');
        if (container) container.classList.add('hidden');
        if (error) error.classList.add('hidden');
    }

    /**
     * Show error state
     */
    showPredictionError() {
        const loading = document.getElementById('predictions-loading');
        const container = document.getElementById('predictions-container');
        const error = document.getElementById('predictions-error');

        if (loading) loading.classList.add('hidden');
        if (container) container.classList.add('hidden');
        if (error) error.classList.remove('hidden');
    }

    /**
     * Load prediction history for display on main page
     */
    async loadPredictionHistory() {
        try {
            console.log('Loading prediction history...');

            // Show loading state
            const historyContainer = document.getElementById('prediction-history');
            if (historyContainer) {
                historyContainer.innerHTML = `
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <span>Loading prediction history...</span>
                    </div>
                `;
            }

            const response = await fetch('/api/v1/prediction-history-public?limit=25');

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Prediction history loaded:', data);

            this.displayPredictionHistory(data);

        } catch (error) {
            console.error('Error loading prediction history:', error);

            // Show error state
            const historyContainer = document.getElementById('prediction-history');
            if (historyContainer) {
                historyContainer.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>Error loading prediction history. Please try again later.</span>
                        <button onclick="publicInterface.loadPredictionHistory()" class="retry-button">
                            Retry
                        </button>
                    </div>
                `;
            }
        }
    }

    /**
     * Display prediction history in the UI
     */
    displayPredictionHistory(data) {
        const historyContainer = document.getElementById('prediction-history');
        if (!historyContainer) return;

        if (!data.history || data.history.length === 0) {
            historyContainer.innerHTML = `
                <div class="no-data">
                    <i class="fas fa-info-circle"></i>
                    <span>No prediction history available yet.</span>
                </div>
            `;
            return;
        }

        // Create history table
        let historyHTML = `
            <div class="history-header">
                <h3><i class="fas fa-history"></i> Previous Predictions</h3>
                <span class="history-count">${data.count} predictions</span>
            </div>
            <div class="history-table-container">
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Numbers</th>
                            <th>Powerball</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        data.history.forEach((prediction, index) => {
            const numbersHTML = prediction.numbers
                .map(num => `<span class="number-ball">${num}</span>`)
                .join('');

            const scorePercentage = Math.round(prediction.score * 100);
            const scoreClass = scorePercentage >= 75 ? 'score-high' :
                              scorePercentage >= 50 ? 'score-medium' : 'score-low';

            historyHTML += `
                <tr class="history-row ${index === 0 ? 'latest' : ''}">
                    <td class="date-cell">
                        <span class="formatted-date">${prediction.formatted_date}</span>
                    </td>
                    <td class="numbers-cell">
                        <div class="numbers-container">
                            ${numbersHTML}
                        </div>
                    </td>
                    <td class="powerball-cell">
                        <span class="powerball-number">${prediction.powerball}</span>
                    </td>
                    <td class="score-cell">
                        <span class="score-value ${scoreClass}">${scorePercentage}%</span>
                    </td>
                </tr>
            `;
        });

        historyHTML += `
                    </tbody>
                </table>
            </div>
            <div class="history-footer">
                <small>Last updated: ${new Date().toLocaleString('es-ES')}</small>
            </div>
        `;

        historyContainer.innerHTML = historyHTML;
    }

    /**
     * Show loading state for history section
     */
    showHistoryLoadingState() {
        const loading = document.getElementById('history-loading');
        if (loading) loading.classList.remove('hidden');
    }

    /**
     * Hide loading state for history section
     */
    hideHistoryLoadingState() {
        const loading = document.getElementById('history-loading');
        if (loading) loading.classList.add('hidden');
    }

    /**
     * Show error state for history section
     */
    showHistoryErrorState() {
        const error = document.getElementById('history-error');
        if (error) error.classList.remove('hidden');
        this.hideHistoryLoadingState();
    }

    /**
     * Hide error state for history section
     */
    hideHistoryErrorState() {
        const error = document.getElementById('history-error');
        if (error) error.classList.add('hidden');
    }

    /**
     * Show empty state for history section
     */
    showHistoryEmptyState() {
        const container = document.getElementById('history-container');
        if (!container) return;

        container.innerHTML = `
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center">
                <i class="fas fa-clock text-4xl text-gray-400 mb-4"></i>
                <h3 class="text-lg font-semibold text-gray-600 dark:text-gray-400 mb-2">No History Available</h3>
                <p class="text-sm text-gray-500 dark:text-gray-500">No previous predictions found in the database.</p>
            </div>
        `;
        container.classList.remove('hidden');
        this.hideHistoryLoadingState();
    }

    /**
     * Show login modal
     */
    showLoginModal() {
        const loginModal = document.getElementById('login-modal');
        if (loginModal) {
            loginModal.classList.remove('hidden');

            // Focus on username field
            const usernameField = document.getElementById('username');
            if (usernameField) {
                setTimeout(() => usernameField.focus(), 100);
            }
        }
    }

    /**
     * Hide login modal
     */
    hideLoginModal() {
        const loginModal = document.getElementById('login-modal');
        const loginError = document.getElementById('login-error');
        const loginForm = document.getElementById('login-form');

        if (loginModal) {
            loginModal.classList.add('hidden');
        }

        if (loginError) {
            loginError.classList.add('hidden');
        }

        if (loginForm) {
            loginForm.reset();
        }

        this.setLoginLoading(false);
    }

    /**
     * Handle login form submission
     * @param {Event} e - Form submit event
     */
    async handleLogin(e) {
        e.preventDefault();

        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value;

        if (!username || !password) {
            this.showLoginError('Please enter both username and password');
            return;
        }

        try {
            this.setLoginLoading(true);
            this.hideLoginError();

            const response = await PowerballUtils.apiRequest('/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, password })
            });

            if (response.success && response.session_token) {
                // Store session token
                sessionStorage.setItem('shiol_session_token', response.session_token);

                // Show success message
                PowerballUtils.showToast('Login successful! Redirecting to dashboard...', 'success');

                // Redirect to dashboard after short delay
                setTimeout(() => {
                    window.location.href = '/dashboard.html';
                }, 1500);
            } else {
                this.showLoginError('Login failed. Please try again.');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showLoginError(error.message || 'Login failed. Please check your credentials.');
        } finally {
            this.setLoginLoading(false);
        }
    }

    /**
     * Set login loading state
     * @param {boolean} loading - Whether login is loading
     */
    setLoginLoading(loading) {
        const loginSubmit = document.getElementById('login-submit');
        const loginBtnText = document.getElementById('login-btn-text');
        const loginSpinner = document.getElementById('login-spinner');

        if (loginSubmit) {
            loginSubmit.disabled = loading;
        }

        if (loginBtnText) {
            loginBtnText.textContent = loading ? 'Logging in...' : 'Login';
        }

        if (loginSpinner) {
            if (loading) {
                loginSpinner.classList.remove('hidden');
            } else {
                loginSpinner.classList.add('hidden');
            }
        }
    }

    /**
     * Show login error
     * @param {string} message - Error message
     */
    showLoginError(message) {
        const loginError = document.getElementById('login-error');
        if (loginError) {
            loginError.querySelector('p').textContent = message;
            loginError.classList.remove('hidden');
        }
    }

    /**
     * Hide login error
     */
    hideLoginError() {
        const loginError = document.getElementById('login-error');
        if (loginError) {
            loginError.classList.add('hidden');
        }
    }

    /**
     * Show general error message
     * @param {string} message - Error message
     */
    showError(message) {
        PowerballUtils.showToast(message, 'error');
    }

    /**
     * Load grouped prediction history by date
     */
    async loadGroupedPredictionHistory() {
        try {
            console.log('Loading grouped prediction history...');

            // Show loading state
            this.showGroupedHistoryLoadingState();

            const response = await fetch('/api/v1/prediction-history-grouped?limit_dates=25');

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Grouped prediction history loaded:', data);

            this.displayGroupedPredictionHistory(data);

        } catch (error) {
            console.error('Error loading grouped prediction history:', error);
            this.showGroupedHistoryErrorState();
        }
    }

    /**
     * Display grouped prediction history
     */
    displayGroupedPredictionHistory(data) {
        const container = document.getElementById('grouped-history-container');
        if (!container) return;

        this.hideGroupedHistoryLoadingState();

        if (!data.grouped_dates || data.grouped_dates.length === 0) {
            this.showGroupedHistoryEmptyState();
            return;
        }

        const groupedDates = data.grouped_dates;

        let historyHTML = `
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                <!-- Header with Summary -->
                <div class="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/30 p-6 border-b border-gray-200 dark:border-gray-700">
                    <div class="flex items-center justify-between mb-4">
                        <div>
                            <h3 class="text-xl font-bold text-gray-900 dark:text-white">
                                <i class="fas fa-calendar-alt mr-2 text-green-600"></i>
                                Prediction History by Date
                            </h3>
                            <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                                Last ${data.total_dates} dates with ${data.total_predictions} total predictions
                            </p>
                        </div>
                        <div class="text-right">
                            <div class="text-2xl font-bold text-green-600 dark:text-green-400">${data.overall_win_rate}</div>
                            <div class="text-xs text-gray-500 dark:text-gray-400">Overall Win Rate</div>
                        </div>
                    </div>

                    <!-- Summary Statistics -->
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div class="bg-white/50 dark:bg-gray-800/50 rounded-lg p-3 text-center">
                            <div class="font-bold text-lg text-gray-900 dark:text-white">${data.total_predictions}</div>
                            <div class="text-gray-600 dark:text-gray-400">Total Plays</div>
                        </div>
                        <div class="bg-white/50 dark:bg-gray-800/50 rounded-lg p-3 text-center">
                            <div class="font-bold text-lg text-green-600">${data.total_winning_predictions}</div>
                            <div class="text-gray-600 dark:text-gray-400">Winners</div>
                        </div>
                        <div class="bg-white/50 dark:bg-gray-800/50 rounded-lg p-3 text-center">
                            <div class="font-bold text-lg text-blue-600">${data.summary.dates_with_winners}</div>
                            <div class="text-gray-600 dark:text-gray-400">Winning Dates</div>
                        </div>
                        <div class="bg-white/50 dark:bg-gray-800/50 rounded-lg p-3 text-center">
                            <div class="font-bold text-lg text-purple-600">${data.summary.average_plays_per_date.toFixed(1)}</div>
                            <div class="text-gray-600 dark:text-gray-400">Avg per Date</div>
                        </div>
                    </div>
                </div>

                <!-- Grouped Dates Table -->
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead class="bg-gray-50 dark:bg-gray-900">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Date</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Plays Generated</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Winning Plays</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Best Prize</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
        `;

        groupedDates.forEach((dateGroup, index) => {
            const winRateClass = parseFloat(dateGroup.win_rate_percentage) > 0 ? 'text-green-600' : 'text-gray-600';
            const rowClass = dateGroup.winning_plays > 0 ? 'bg-green-50 dark:bg-green-900/20' : '';

            historyHTML += `
                <tr class="hover:bg-gray-50 dark:hover:bg-gray-700 ${rowClass}">
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="flex flex-col">
                            <div class="text-sm font-medium text-gray-900 dark:text-white">${dateGroup.formatted_date}</div>
                            <div class="text-xs text-gray-500 dark:text-gray-400">${dateGroup.date}</div>
                        </div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="text-sm font-bold text-gray-900 dark:text-white">${dateGroup.total_plays}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">predictions</div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="flex items-center">
                            <div class="text-sm font-bold ${winRateClass}">${dateGroup.winning_plays}</div>
                            <div class="ml-2 text-xs ${winRateClass}">(${dateGroup.win_rate_percentage})</div>
                        </div>
                        ${dateGroup.winning_plays > 0 ? `<div class="text-xs text-green-600">🏆 Winners found!</div>` : `<div class="text-xs text-gray-500">No matches</div>`}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="text-sm font-medium text-gray-900 dark:text-white">${dateGroup.best_prize}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">${dateGroup.total_prize_display}</div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button onclick="publicInterface.showDateDetailsModal('${dateGroup.date}')"
                                class="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors">
                            <i class="fas fa-eye mr-1"></i>
                            View Details
                        </button>
                    </td>
                </tr>
            `;
        });

        historyHTML += `
                        </tbody>
                    </table>
                </div>

                <!-- Footer -->
                <div class="bg-gray-50 dark:bg-gray-900 px-6 py-4 border-t border-gray-200 dark:border-gray-700">
                    <div class="flex items-center justify-between">
                        <div class="text-sm text-gray-600 dark:text-gray-400">
                            Showing ${data.total_dates} most recent dates
                        </div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">
                            Last updated: ${new Date().toLocaleString('es-ES')}
                        </div>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = historyHTML;
        container.classList.remove('hidden');
    }

    /**
     * Show date details modal
     */
    async showDateDetailsModal(date) {
        try {
            console.log('Loading details for date:', date);

            const modal = document.getElementById('date-details-modal');
            const modalTitle = document.getElementById('modal-title');
            const modalContent = document.getElementById('modal-content');

            if (!modal || !modalContent) return;

            // Show modal with loading state
            modalTitle.textContent = `Predictions for ${date}`;
            modalContent.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-spinner fa-spin text-3xl text-blue-600 mb-4"></i>
                    <p class="text-gray-600">Loading prediction details...</p>
                </div>
            `;
            modal.classList.remove('hidden');

            // Get grouped data again to find the specific date
            const response = await fetch('/api/v1/prediction-history-grouped?limit_dates=25');
            const data = await response.json();

            const dateGroup = data.grouped_dates.find(group => group.date === date);
            if (!dateGroup) {
                modalContent.innerHTML = `
                    <div class="text-center py-8">
                        <i class="fas fa-exclamation-triangle text-3xl text-red-500 mb-4"></i>
                        <p class="text-red-600">No data found for this date.</p>
                    </div>
                `;
                return;
            }

            this.displayDateDetails(dateGroup, modalContent);

        } catch (error) {
            console.error('Error loading date details:', error);
            const modalContent = document.getElementById('modal-content');
            if (modalContent) {
                modalContent.innerHTML = `
                    <div class="text-center py-8">
                        <i class="fas fa-exclamation-triangle text-3xl text-red-500 mb-4"></i>
                        <p class="text-red-600">Error loading prediction details.</p>
                    </div>
                `;
            }
        }
    }

    /**
     * Display date details in modal
     */
    displayDateDetails(dateGroup, modalContent) {
        const predictions = dateGroup.predictions || [];

        let detailsHTML = `
            <!-- Header Stats -->
            <div class="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30 rounded-lg p-4 mb-6">
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div>
                        <div class="text-2xl font-bold text-blue-600">${dateGroup.total_plays}</div>
                        <div class="text-xs text-gray-600 dark:text-gray-400">Total Plays</div>
                    </div>
                    <div>
                        <div class="text-2xl font-bold text-green-600">${dateGroup.winning_plays}</div>
                        <div class="text-xs text-gray-600 dark:text-gray-400">Winners</div>
                    </div>
                    <div>
                        <div class="text-2xl font-bold text-purple-600">${dateGroup.win_rate_percentage}</div>
                        <div class="text-xs text-gray-600 dark:text-gray-400">Win Rate</div>
                    </div>
                    <div>
                        <div class="text-2xl font-bold text-orange-600">${dateGroup.total_prize_display}</div>
                        <div class="text-xs text-gray-600 dark:text-gray-400">Total Prizes</div>
                    </div>
                </div>
            </div>

            <!-- Predictions List -->
            <div class="space-y-3">
        `;

        predictions.forEach((prediction, index) => {
            const hasWin = prediction.has_prize;
            const cardClass = hasWin ? 'border-green-200 bg-green-50 dark:bg-green-900/20' : 'border-gray-200 bg-white dark:bg-gray-800';
            // Show both creation date and target date for context
            const createdDate = prediction.created_at ? new Date(prediction.created_at).toLocaleDateString() : '';
            const targetDate = prediction.target_draw_date || dateGroup.date;

            detailsHTML += `
                <div class="border ${cardClass} rounded-lg p-4">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-4">
                            <div class="flex items-center justify-center w-8 h-8 rounded-full ${hasWin ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'} text-sm font-bold">
                                ${index + 1}
                            </div>
                            <div class="flex items-center space-x-2">
                                ${prediction.numbers.map(num => `
                                    <span class="inline-flex items-center justify-center w-8 h-8 bg-white text-gray-900 rounded-full text-sm font-bold border-2 border-gray-300">${num}</span>
                                `).join('')}
                                <span class="text-red-500 text-lg font-bold mx-2">•</span>
                                <span class="inline-flex items-center justify-center w-8 h-8 bg-red-600 text-white rounded-full text-sm font-bold">${prediction.powerball}</span>
                            </div>
                        </div>
                        <div class="text-right">
                            ${hasWin ? `
                                <div class="text-green-600 font-bold">${prediction.prize_description}</div>
                                <div class="text-sm text-green-500">$${prediction.prize_amount}</div>
                                <div class="text-xs text-gray-500">${prediction.matches_main} matches ${prediction.powerball_match ? '+ PB' : ''}</div>
                            ` : `
                                <div class="text-gray-500">No matches</div>
                                <div class="text-xs text-gray-400">${prediction.matches_main} matches ${prediction.powerball_match ? '+ PB' : ''}</div>
                            `}
                            <div class="text-xs text-gray-400 mt-1">Score: ${(prediction.score * 100).toFixed(1)}%</div>
                            <div class="text-xs text-gray-400 mt-1">
                                Generated: ${createdDate} | For: ${targetDate}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        detailsHTML += '</div>';

        modalContent.innerHTML = detailsHTML;
    }

    /**
     * Hide date details modal
     */
    hideDateDetailsModal() {
        const modal = document.getElementById('date-details-modal');
        if (modal) {
            modal.classList.add('hidden');
        }
    }

    /**
     * Show loading state for grouped history
     */
    showGroupedHistoryLoadingState() {
        const loading = document.getElementById('grouped-history-loading');
        const container = document.getElementById('grouped-history-container');
        const error = document.getElementById('grouped-history-error');

        if (loading) loading.classList.remove('hidden');
        if (container) container.classList.add('hidden');
        if (error) error.classList.add('hidden');
    }

    /**
     * Hide loading state for grouped history
     */
    hideGroupedHistoryLoadingState() {
        const loading = document.getElementById('grouped-history-loading');
        if (loading) loading.classList.add('hidden');
    }

    /**
     * Show error state for grouped history
     */
    showGroupedHistoryErrorState() {
        const loading = document.getElementById('grouped-history-loading');
        const container = document.getElementById('grouped-history-container');
        const error = document.getElementById('grouped-history-error');

        if (loading) loading.classList.add('hidden');
        if (container) container.classList.add('hidden');
        if (error) error.classList.remove('hidden');
    }

    /**
     * Show empty state for grouped history
     */
    showGroupedHistoryEmptyState() {
        const container = document.getElementById('grouped-history-container');
        if (!container) return;

        container.innerHTML = `
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center">
                <i class="fas fa-calendar-times text-4xl text-gray-400 mb-4"></i>
                <h3 class="text-lg font-semibold text-gray-600 dark:text-gray-400 mb-2">No History Available</h3>
                <p class="text-sm text-gray-500 dark:text-gray-500">No previous predictions found grouped by date.</p>
            </div>
        `;
        container.classList.remove('hidden');
        this.hideGroupedHistoryLoadingState();
    }

    /**
     * Refresh all data
     */
    async refreshData() {
        await this.loadNextDrawingInfo();
        await this.loadGroupedPredictionHistory();
    }
}

// Initialize the public interface
const publicInterface = new PublicInterface();

// Make it available globally for other scripts
window.PublicInterface = publicInterface;

// Auto-refresh data every 5 minutes
setInterval(() => {
    if (document.visibilityState === 'visible') {
        publicInterface.refreshData();
    }
}, 5 * 60 * 1000);

// Refresh when page becomes visible
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        publicInterface.refreshData();
    }
});