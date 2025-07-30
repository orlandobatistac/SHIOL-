/**
 * SHIOL+ Simple Authentication JavaScript
 * ======================================
 * 
 * Simplified authentication system without auto-refresh loops.
 * Handles login functionality and basic dashboard protection.
 */

class SimpleAuthManager {
    constructor() {
        this.sessionToken = null;
        this.user = null;
        this.isLoading = false;
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    /**
     * Initialize authentication manager
     */
    init() {
        console.log('Initializing SHIOL+ Simple Authentication');
        
        // Setup event listeners for login page
        this.setupLoginEventListeners();
        
        // Check if we're on dashboard and need protection
        this.checkDashboardAccess();
    }

    /**
     * Setup event listeners for login page
     */
    setupLoginEventListeners() {
        // Only setup login listeners if we're on login page
        const loginForm = document.getElementById('login-form');
        if (!loginForm) return;

        loginForm.addEventListener('submit', (e) => this.handleLogin(e));

        // Password toggle
        const togglePassword = document.getElementById('toggle-password');
        if (togglePassword) {
            togglePassword.addEventListener('click', () => this.togglePasswordVisibility());
        }

        // Auto-focus username field
        setTimeout(() => {
            const usernameInput = document.getElementById('username');
            if (usernameInput && !usernameInput.value) {
                usernameInput.focus();
            }
        }, 100);
    }

    /**
     * Check dashboard access (single check, no loops)
     */
    async checkDashboardAccess() {
        // Only run on dashboard pages
        if (!this.isDashboardPage()) {
            return;
        }

        const token = sessionStorage.getItem('shiol_session_token') || localStorage.getItem('shiol_session_token');
        
        if (!token) {
            this.redirectToLogin();
            return;
        }

        try {
            const isValid = await this.verifySessionOnce(token);
            if (!isValid) {
                this.redirectToLogin();
            } else {
                console.log('Dashboard access granted');
                this.setupLogoutButton();
            }
        } catch (error) {
            console.error('Dashboard access verification failed:', error);
            this.redirectToLogin();
        }
    }

    /**
     * Check if current page is dashboard
     */
    isDashboardPage() {
        return window.location.pathname.includes('dashboard') || 
               document.title.includes('Dashboard') ||
               document.querySelector('[data-page="dashboard"]');
    }

    /**
     * Verify session token (single check, no intervals)
     */
    async verifySessionOnce(token) {
        try {
            const response = await PowerballUtils.apiRequest('/auth/verify', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.valid && response.user) {
                this.sessionToken = token;
                this.user = response.user;
                return true;
            }

            return false;
        } catch (error) {
            console.error('Session verification failed:', error);
            return false;
        }
    }

    /**
     * Handle login form submission
     */
    async handleLogin(e) {
        e.preventDefault();
        
        if (this.isLoading) return;

        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value;
        const rememberMe = document.getElementById('remember-me').checked;

        // Validate inputs
        if (!this.validateLoginInputs(username, password)) {
            return;
        }

        try {
            this.setLoadingState(true);
            this.hideMessages();

            console.log('Attempting login for user:', username);

            const response = await PowerballUtils.apiRequest('/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, password })
            });

            if (response.success && response.session_token) {
                this.handleLoginSuccess(response, rememberMe);
            } else {
                this.showError('Login failed. Please check your credentials.');
            }

        } catch (error) {
            console.error('Login error:', error);
            this.handleLoginError(error);
        } finally {
            this.setLoadingState(false);
        }
    }

    /**
     * Validate login inputs
     */
    validateLoginInputs(username, password) {
        if (!username) {
            this.showError('Please enter your username');
            return false;
        }

        if (!password) {
            this.showError('Please enter your password');
            return false;
        }

        if (username.length < 3) {
            this.showError('Username must be at least 3 characters');
            return false;
        }

        if (password.length < 6) {
            this.showError('Password must be at least 6 characters');
            return false;
        }

        return true;
    }

    /**
     * Handle successful login
     */
    handleLoginSuccess(response, rememberMe) {
        console.log('Login successful for user:', response.user.username);

        // Store session token
        this.sessionToken = response.session_token;
        this.user = response.user;

        // Store token based on remember me preference
        if (rememberMe) {
            localStorage.setItem('shiol_session_token', response.session_token);
            localStorage.setItem('shiol_user', JSON.stringify(response.user));
        } else {
            sessionStorage.setItem('shiol_session_token', response.session_token);
            sessionStorage.setItem('shiol_user', JSON.stringify(response.user));
        }

        // Show success message
        this.showSuccess('Login successful! Redirecting to dashboard...');

        // Redirect after delay
        setTimeout(() => {
            window.location.href = '/dashboard.html';
        }, 1500);
    }

    /**
     * Handle login error
     */
    handleLoginError(error) {
        let errorMessage = 'Login failed. Please try again.';

        if (error.message) {
            if (error.message.includes('Invalid credentials')) {
                errorMessage = 'Invalid username or password. Please try again.';
            } else if (error.message.includes('401')) {
                errorMessage = 'Invalid credentials. Please check your username and password.';
            } else if (error.message.includes('500')) {
                errorMessage = 'Server error. Please try again later.';
            } else if (error.message.includes('Network')) {
                errorMessage = 'Network error. Please check your connection.';
            }
        }

        this.showError(errorMessage);
    }

    /**
     * Setup logout button functionality
     */
    setupLogoutButton() {
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.logout());
        }

        // Display user info
        const usernameDisplay = document.getElementById('username-display');
        if (usernameDisplay) {
            const storedUser = sessionStorage.getItem('shiol_user') || localStorage.getItem('shiol_user');
            if (storedUser) {
                try {
                    const user = JSON.parse(storedUser);
                    usernameDisplay.textContent = user.username || 'Admin';
                } catch (e) {
                    usernameDisplay.textContent = 'Admin';
                }
            } else {
                usernameDisplay.textContent = 'Admin';
            }
        }
    }

    /**
     * Logout user
     */
    async logout() {
        try {
            if (this.sessionToken) {
                await PowerballUtils.apiRequest('/auth/logout', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.sessionToken}`
                    }
                });
            }
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            this.clearSession();
            window.location.href = '/login.html';
        }
    }

    /**
     * Clear session data
     */
    clearSession() {
        this.sessionToken = null;
        this.user = null;
        sessionStorage.removeItem('shiol_session_token');
        sessionStorage.removeItem('shiol_user');
        localStorage.removeItem('shiol_session_token');
        localStorage.removeItem('shiol_user');
    }

    /**
     * Redirect to login page
     */
    redirectToLogin() {
        this.clearSession();
        window.location.href = '/login.html';
    }

    /**
     * Toggle password visibility
     */
    togglePasswordVisibility() {
        const passwordInput = document.getElementById('password');
        const passwordEye = document.getElementById('password-eye');

        if (!passwordInput || !passwordEye) return;

        if (passwordInput.type === 'password') {
            passwordInput.type = 'text';
            passwordEye.className = 'fas fa-eye-slash';
        } else {
            passwordInput.type = 'password';
            passwordEye.className = 'fas fa-eye';
        }
    }

    /**
     * Set loading state
     */
    setLoadingState(loading) {
        this.isLoading = loading;

        const loginSubmit = document.getElementById('login-submit');
        const loginBtnText = document.getElementById('login-btn-text');
        const loginSpinner = document.getElementById('login-spinner');

        if (loginSubmit) {
            loginSubmit.disabled = loading;
        }

        if (loginBtnText) {
            loginBtnText.textContent = loading ? 'Signing In...' : 'Sign In';
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
     * Show error message
     */
    showError(message) {
        const errorDiv = document.getElementById('login-error');
        const errorMessage = document.getElementById('error-message');
        const successDiv = document.getElementById('login-success');

        if (successDiv) {
            successDiv.classList.add('hidden');
        }

        if (errorDiv && errorMessage) {
            errorMessage.textContent = message;
            errorDiv.classList.remove('hidden');
        }
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        const successDiv = document.getElementById('login-success');
        const errorDiv = document.getElementById('login-error');

        if (errorDiv) {
            errorDiv.classList.add('hidden');
        }

        if (successDiv) {
            successDiv.querySelector('p').textContent = message;
            successDiv.classList.remove('hidden');
        }
    }

    /**
     * Hide all messages
     */
    hideMessages() {
        const errorDiv = document.getElementById('login-error');
        const successDiv = document.getElementById('login-success');

        if (errorDiv) {
            errorDiv.classList.add('hidden');
        }
        if (successDiv) {
            successDiv.classList.add('hidden');
        }
    }
}

// Initialize simple authentication manager
const simpleAuthManager = new SimpleAuthManager();

// Make available globally
window.SimpleAuthManager = simpleAuthManager;