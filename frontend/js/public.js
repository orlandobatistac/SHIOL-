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
            
            // Load 100 Smart AI predictions
            const data = await PowerballUtils.apiRequest('/predict/smart?num_plays=100');
            
            // Update last updated timestamps
            const updateTime = new Date().toLocaleString();
            const footerLastUpdated = document.getElementById('footer-last-updated');
            if (footerLastUpdated) {
                footerLastUpdated.textContent = updateTime;
            }

            // Display Smart AI predictions
            if (data.smart_predictions && data.smart_predictions.length > 0) {
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
        if (!container || !loading) return;

        const predictions = data.smart_predictions || [];
        
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
                            <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                                ${predictions.length} predictions ranked by AI confidence score
                            </p>
                        </div>
                        <div class="text-right">
                            <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">${predictions.length}</div>
                            <div class="text-xs text-gray-500 dark:text-gray-400">Total Plays</div>
                        </div>
                    </div>
                </div>

                <!-- Predictions Table -->
                <div class="overflow-x-auto max-h-96 overflow-y-auto">
                    <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead class="bg-gray-50 dark:bg-gray-900 sticky top-0">
                            <tr>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Rank</th>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Numbers</th>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">PB</th>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">AI Score</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                            ${predictions.map((pred, index) => {
                                const isTopTen = index < 10;
                                return `
                                    <tr class="hover:bg-gray-50 dark:hover:bg-gray-700 ${isTopTen ? 'bg-yellow-50 dark:bg-yellow-900/20' : ''}">
                                        <td class="px-4 py-3 whitespace-nowrap">
                                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full ${isTopTen ? 'bg-yellow-100 text-yellow-800 border-2 border-yellow-300' : 'bg-gray-100 text-gray-800'} text-sm font-bold">
                                                ${pred.rank || (index + 1)}
                                            </span>
                                        </td>
                                        <td class="px-4 py-3 whitespace-nowrap">
                                            <div class="flex space-x-1">
                                                ${pred.numbers.map(num => `
                                                    <span class="inline-flex items-center justify-center w-7 h-7 bg-blue-600 text-white rounded-full text-xs font-semibold">${num}</span>
                                                `).join('')}
                                            </div>
                                        </td>
                                        <td class="px-4 py-3 whitespace-nowrap">
                                            <span class="inline-flex items-center justify-center w-7 h-7 bg-red-600 text-white rounded-full text-xs font-semibold">${pred.powerball}</span>
                                        </td>
                                        <td class="px-4 py-3 whitespace-nowrap">
                                            <div class="text-sm font-medium text-gray-900 dark:text-white">
                                                ${(pred.total_score * 100).toFixed(1)}%
                                            </div>
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>

                <!-- Footer Info -->
                <div class="bg-gray-50 dark:bg-gray-900 px-6 py-4 text-center">
                    <p class="text-xs text-gray-500 dark:text-gray-400">
                        Generated by SHIOL+ AI Pipeline • Last updated: ${new Date().toLocaleString()}
                    </p>
                </div>
            </div>
        `;
        
        loading.classList.add('hidden');
        container.innerHTML = predictionsHtml;
        container.classList.remove('hidden');
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