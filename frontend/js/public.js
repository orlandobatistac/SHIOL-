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

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideLoginModal();
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
     * Load initial data for the interface
     */
    async loadInitialData() {
        await this.loadNextDrawingInfo();
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

                this.displaySmartPredictions(data);
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
    displaySmartPredictions(data) {
        const container = document.getElementById('predictions-container');
        const loading = document.getElementById('predictions-loading');
        const error = document.getElementById('predictions-error');
        
        if (!container || !loading) return;

        // Hide loading and error states
        loading.classList.add('hidden');
        if (error) error.classList.add('hidden');

        // Sort predictions by confidence score in descending order (best to worst)
        const predictions = (data.smart_predictions || []).sort((a, b) => {
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

        const predictionsHtml = `
            <div class="bg-white rounded-lg border border-gray-200 overflow-hidden">
                <!-- Mobile-first Header -->
                <div class="bg-gradient-to-r from-blue-50 to-indigo-50 p-3 sm:p-4 border-b border-gray-200">
                    <div class="text-center sm:flex sm:items-center sm:justify-between">
                        <div class="mb-2 sm:mb-0">
                            <h3 class="text-lg sm:text-xl font-bold text-gray-900">
                                <i class="fas fa-brain mr-1 sm:mr-2 text-blue-600"></i>
                                AI Predictions
                            </h3>
                        </div>
                        <div class="text-lg sm:text-2xl font-bold text-blue-600">${predictions.length}</div>
                    </div>
                </div>

                <!-- Mobile Cards / Desktop Table -->
                <div class="block sm:hidden">
                    <!-- Mobile Card Layout -->
                    <div class="max-h-96 overflow-y-auto">
                        ${predictions.map((pred, index) => {
                            const isTopFive = index < 5;
                            const getDisplayScore = (pred) => {
                                const possibleScores = [
                                    pred.total_score, pred.score_total, pred.confidence,
                                    pred.score, pred.ai_score, pred.probability
                                ];
                                for (let score of possibleScores) {
                                    if (typeof score === 'number' && !isNaN(score)) return score;
                                    if (typeof score === 'string') {
                                        const numScore = parseFloat(score);
                                        if (!isNaN(numScore)) return numScore;
                                    }
                                }
                                return 0;
                            };
                            const confidenceScore = getDisplayScore(pred);
                            const displayRank = index + 1;

                            return `
                                <div class="p-3 border-b border-gray-100 ${isTopFive ? 'bg-blue-50' : ''}">
                                    <div class="flex items-center justify-between mb-2">
                                        <span class="inline-flex items-center justify-center w-6 h-6 rounded-full ${isTopFive ? 'bg-blue-100 text-blue-800 border border-blue-300' : 'bg-gray-100 text-gray-700 border border-gray-300'} text-xs font-bold">
                                            ${displayRank}
                                        </span>
                                        <div class="text-right">
                                            <div class="text-sm font-bold text-gray-900">${(confidenceScore * 100).toFixed(1)}%</div>
                                            ${isTopFive ? '<div class="text-xs text-blue-600 font-semibold">TOP</div>' : ''}
                                        </div>
                                    </div>
                                    <div class="flex items-center justify-center space-x-1">
                                        ${(pred.numbers || []).map(num => `
                                            <span class="inline-flex items-center justify-center w-8 h-8 bg-white text-gray-900 rounded-full text-xs font-bold border border-gray-300">${num}</span>
                                        `).join('')}
                                        <span class="text-red-500 text-sm font-bold mx-1">•</span>
                                        <span class="inline-flex items-center justify-center w-8 h-8 bg-red-600 text-white rounded-full text-xs font-bold">${pred.powerball || pred.pb || ''}</span>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>

                <!-- Desktop Table Layout -->
                <div class="hidden sm:block overflow-x-auto max-h-96 overflow-y-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50 sticky top-0">
                            <tr>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">#</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Numbers</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Score</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
                            ${predictions.map((pred, index) => {
                                const isTopFive = index < 5;
                                const getDisplayScore = (pred) => {
                                    const possibleScores = [
                                        pred.total_score, pred.score_total, pred.confidence,
                                        pred.score, pred.ai_score, pred.probability
                                    ];
                                    for (let score of possibleScores) {
                                        if (typeof score === 'number' && !isNaN(score)) return score;
                                        if (typeof score === 'string') {
                                            const numScore = parseFloat(score);
                                            if (!isNaN(numScore)) return numScore;
                                        }
                                    }
                                    return 0;
                                };
                                const confidenceScore = getDisplayScore(pred);
                                const displayRank = index + 1;

                                return `
                                    <tr class="hover:bg-gray-50 ${isTopFive ? 'bg-blue-50' : ''}">
                                        <td class="px-4 py-3">
                                            <span class="inline-flex items-center justify-center w-7 h-7 rounded-full ${isTopFive ? 'bg-blue-100 text-blue-800 border border-blue-300' : 'bg-gray-100 text-gray-700 border border-gray-300'} text-sm font-bold">
                                                ${displayRank}
                                            </span>
                                            ${isTopFive ? '<span class="ml-2 text-xs font-semibold text-blue-600">TOP</span>' : ''}
                                        </td>
                                        <td class="px-4 py-3">
                                            <div class="flex items-center space-x-2">
                                                ${(pred.numbers || []).map(num => `
                                                    <span class="inline-flex items-center justify-center w-9 h-9 bg-white text-gray-900 rounded-full text-sm font-bold border border-gray-300">${num}</span>
                                                `).join('')}
                                                <span class="text-red-500 text-lg font-bold mx-1">•</span>
                                                <span class="inline-flex items-center justify-center w-9 h-9 bg-red-600 text-white rounded-full text-sm font-bold">${pred.powerball || pred.pb || ''}</span>
                                            </div>
                                        </td>
                                        <td class="px-4 py-3">
                                            <div class="text-lg font-bold text-gray-900">${(confidenceScore * 100).toFixed(1)}%</div>
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>

                <!-- Simplified Guide -->
                <div class="p-3 sm:p-4 bg-blue-50 border-t border-blue-200">
                    <div class="text-center">
                        <h4 class="text-sm font-semibold text-blue-800 mb-2">
                            <i class="fas fa-info-circle text-blue-500 mr-1"></i>
                            Score Guide
                        </h4>
                        <div class="flex flex-wrap justify-center gap-2 text-xs text-blue-700">
                            <span class="flex items-center"><span class="w-2 h-2 bg-blue-500 rounded-full mr-1"></span>80-100% Premium</span>
                            <span class="flex items-center"><span class="w-2 h-2 bg-green-500 rounded-full mr-1"></span>60-80% High</span>
                            <span class="flex items-center"><span class="w-2 h-2 bg-yellow-500 rounded-full mr-1"></span>40-60% Good</span>
                            <span class="flex items-center"><span class="w-2 h-2 bg-gray-500 rounded-full mr-1"></span>&lt;40% Standard</span>
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
     * Refresh all data
     */
    async refreshData() {
        try {
            await this.loadInitialData();
            PowerballUtils.showToast('Data refreshed successfully', 'success');
        } catch (error) {
            console.error('Error refreshing data:', error);
            PowerballUtils.showToast('Failed to refresh data', 'error');
        }
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