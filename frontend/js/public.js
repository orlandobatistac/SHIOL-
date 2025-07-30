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
     * Load next drawing information and featured prediction
     */
    async loadNextDrawingInfo() {
        try {
            console.log('Loading next drawing info...');
            
            const data = await PowerballUtils.apiRequest('/public/next-drawing');
            
            // Update last updated timestamps
            const updateTime = new Date().toLocaleString();
            const footerLastUpdated = document.getElementById('footer-last-updated');
            if (footerLastUpdated) {
                footerLastUpdated.textContent = updateTime;
            }

            // Display featured prediction
            if (data.featured_prediction) {
                this.displayDashboardStylePredictions(data.featured_prediction);
            }

            console.log('Next drawing info loaded successfully');
        } catch (error) {
            console.error('Error loading next drawing info:', error);
            this.showFeaturedPredictionError();
        }
    }

    /**
     * Display multiple predictions in the dashboard style
     * @param {Object} data - Prediction data
     */
    displayDashboardStylePredictions(data) {
        const aiPredictions = document.getElementById('ai-predictions');
        if (!aiPredictions) return;

        const predictions = data.predictions || [data];
        
        const predictionsHtml = `
            <div id="last-generated-plays" class="mt-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-gray-800 dark:text-white">
                        <i class="fas fa-star mr-2 text-yellow-500"></i>
                        AI-Generated Plays
                    </h3>
                    <span id="last-plays-timestamp" class="text-sm text-gray-500 dark:text-gray-400">Generated: ${new Date().toLocaleString()}</span>
                </div>
                <div id="last-plays-container" class="space-y-3">
                    ${predictions.map((prediction, index) => {
                        const numbersHtml = PowerballUtils.createNumbersDisplay(
                            PowerballUtils.sortNumbers(prediction.numbers),
                            prediction.powerball,
                            'small'
                        );

                        return `
                            <div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                <div class="flex items-center space-x-2">
                                    <span class="text-sm font-medium text-gray-600 dark:text-gray-400">Play ${index + 1}:</span>
                                    ${numbersHtml}
                                </div>
                                <div class="text-xs text-gray-500 dark:text-gray-400">
                                    Confidence: ${(prediction.confidence_score * 100).toFixed(1)}%
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
        
        aiPredictions.innerHTML = predictionsHtml;
    }

    /**
     * Show error in featured prediction area
     */
    showFeaturedPredictionError() {
        const aiPredictions = document.getElementById('ai-predictions');
        if (aiPredictions) {
            aiPredictions.innerHTML = PowerballUtils.createErrorPlaceholder('Unable to load prediction').innerHTML;
        }
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