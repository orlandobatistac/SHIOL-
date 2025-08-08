/**
 * SHIOL+ Powerball Utilities
 * ==========================
 * 
 * Reusable utilities and components for Powerball number display and formatting.
 * Implements the PowerballCard component with conditional rendering as specified.
 */

class PowerballUtils {
    /**
     * Create a PowerballCard component with conditional rendering
     * @param {Object} options - Configuration options
     * @param {Array} options.numbers - Array of 5 white ball numbers
     * @param {number} options.powerball - Powerball number
     * @param {string} options.type - 'prediction' or 'result'
     * @param {string} options.date - Date string for the card
     * @param {Object} options.metadata - Additional metadata (score, jackpot, etc.)
     * @param {boolean} options.featured - Whether this is a featured/highlighted card
     * @returns {HTMLElement} - The created card element
     */
    static createPowerballCard(options) {
        const {
            numbers = [],
            powerball = 0,
            type = 'prediction',
            date = '',
            metadata = {},
            featured = false
        } = options;

        // Create main card container
        const card = document.createElement('div');
        card.className = `powerball-card ${featured ? 'featured-card' : 'history-card'} bg-white rounded-xl p-6 shadow-md transition-all duration-300 hover:shadow-lg`;

        // Create date badge if provided
        let dateBadge = '';
        if (date) {
            const badgeClass = type === 'prediction' ? 'bg-blue-600' : 'bg-gray-600';
            dateBadge = `
                <div class="date-badge ${badgeClass} text-white text-sm font-semibold px-3 py-1 rounded-full inline-block mb-4">
                    ${this.formatDate(date)}
                </div>
            `;
        }

        // Create numbers display
        const numbersHtml = this.createNumbersDisplay(numbers, powerball);

        // Create metadata section based on type
        let metadataHtml = '';
        if (type === 'prediction' && metadata.confidence_score) {
            metadataHtml = `
                <div class="mt-4 flex items-center justify-center space-x-4 text-sm text-gray-600">
                    <div class="flex items-center">
                        <i class="fas fa-chart-line text-green-600 mr-1"></i>
                        <span>Confidence: <span class="font-semibold">${(metadata.confidence_score * 100).toFixed(1)}%</span></span>
                    </div>
                    <div class="flex items-center">
                        <i class="fas fa-robot text-blue-600 mr-1"></i>
                        <span>Method: <span class="font-semibold">${metadata.method || 'AI'}</span></span>
                    </div>
                </div>
            `;
        } else if (type === 'result' && metadata.jackpot_amount) {
            metadataHtml = `
                <div class="mt-4 text-center">
                    <div class="jackpot-amount text-green-600 font-bold text-lg">
                        Jackpot: ${metadata.jackpot_amount}
                    </div>
                    ${metadata.multiplier ? `<div class="text-sm text-gray-600 mt-1">Multiplier: ${metadata.multiplier}x</div>` : ''}
                </div>
            `;
        }

        // Assemble the card
        card.innerHTML = `
            ${dateBadge}
            <div class="text-center">
                ${numbersHtml}
                ${metadataHtml}
            </div>
        `;

        return card;
    }

    /**
     * Create the numbers display component
     * @param {Array} numbers - Array of 5 white ball numbers
     * @param {number} powerball - Powerball number
     * @returns {string} - HTML string for numbers display
     */
    static createNumbersDisplay(numbers, powerball) {
        if (!numbers || numbers.length !== 5 || !powerball) {
            return '<div class="text-gray-500">Invalid numbers</div>';
        }

        const whiteBalls = numbers.map(num => 
            `<div class="powerball-number white-ball">${num}</div>`
        ).join('');

        const powerBall = `<div class="powerball-number power-ball">${powerball}</div>`;

        return `
            <div class="flex items-center justify-center space-x-3 mb-2">
                ${whiteBalls}
                <div class="w-4 h-4 flex items-center justify-center">
                    <div class="w-2 h-2 bg-red-600 rounded-full"></div>
                </div>
                ${powerBall}
            </div>
        `;
    }

    /**
     * Format date for display
     * @param {string|Date} date - Date to format
     * @returns {string} - Formatted date string
     */
    static formatDate(date) {
        try {
            const dateObj = typeof date === 'string' ? new Date(date) : date;
            const options = { 
                weekday: 'short', 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric' 
            };
            return dateObj.toLocaleDateString('en-US', options);
        } catch (error) {
            console.error('Error formatting date:', error);
            return date.toString();
        }
    }

    /**
     * Format date for next drawing display
     * @param {string|Date} date - Date to format
     * @returns {string} - Formatted date string for next drawing
     */
    static formatNextDrawingDate(date) {
        try {
            const dateObj = typeof date === 'string' ? new Date(date) : date;
            const today = new Date();
            const tomorrow = new Date(today);
            tomorrow.setDate(tomorrow.getDate() + 1);

            // Convert to same timezone for comparison
            const todayUTC = new Date(today.getFullYear(), today.getMonth(), today.getDate());
            const tomorrowUTC = new Date(tomorrow.getFullYear(), tomorrow.getMonth(), tomorrow.getDate());
            const dateUTC = new Date(dateObj.getFullYear(), dateObj.getMonth(), dateObj.getDate());
            
            // Check if it's today (and before drawing time)
            if (dateUTC.getTime() === todayUTC.getTime()) {
                const currentHour = today.getHours();
                // Only show "Today" if it's before 11 PM (23:00)
                if (currentHour < 23) {
                    return 'Today';
                } else {
                    // After 11 PM, the next drawing is effectively tomorrow or later
                    return 'Tomorrow';
                }
            }
            
            // Check if it's tomorrow
            if (dateUTC.getTime() === tomorrowUTC.getTime()) {
                return 'Tomorrow';
            }

            // Otherwise, show day of week and date
            const options = { 
                weekday: 'long', 
                month: 'short', 
                day: 'numeric' 
            };
            return dateObj.toLocaleDateString('en-US', options);
        } catch (error) {
            console.error('Error formatting next drawing date:', error);
            return date.toString();
        }
    }

    /**
     * Validate Powerball numbers
     * @param {Array} numbers - Array of 5 white ball numbers
     * @param {number} powerball - Powerball number
     * @returns {boolean} - Whether numbers are valid
     */
    static validateNumbers(numbers, powerball) {
        // Check if we have exactly 5 white ball numbers
        if (!Array.isArray(numbers) || numbers.length !== 5) {
            return false;
        }

        // Check white ball range (1-69) and uniqueness
        const uniqueNumbers = new Set(numbers);
        if (uniqueNumbers.size !== 5) {
            return false; // Duplicate numbers
        }

        for (const num of numbers) {
            if (!Number.isInteger(num) || num < 1 || num > 69) {
                return false;
            }
        }

        // Check powerball range (1-26)
        if (!Number.isInteger(powerball) || powerball < 1 || powerball > 26) {
            return false;
        }

        return true;
    }

    /**
     * Sort numbers in ascending order (for display consistency)
     * @param {Array} numbers - Array of numbers to sort
     * @returns {Array} - Sorted array
     */
    static sortNumbers(numbers) {
        return [...numbers].sort((a, b) => a - b);
    }

    /**
     * Create a loading placeholder for cards
     * @param {string} message - Loading message
     * @returns {HTMLElement} - Loading placeholder element
     */
    static createLoadingPlaceholder(message = 'Loading...') {
        const placeholder = document.createElement('div');
        placeholder.className = 'loading-placeholder text-center py-12';
        placeholder.innerHTML = `
            <i class="fas fa-spinner fa-spin text-gray-400 text-3xl mb-4 loading-spinner"></i>
            <p class="text-gray-500">${message}</p>
        `;
        return placeholder;
    }

    /**
     * Create an error placeholder for cards
     * @param {string} message - Error message
     * @returns {HTMLElement} - Error placeholder element
     */
    static createErrorPlaceholder(message = 'Error loading data') {
        const placeholder = document.createElement('div');
        placeholder.className = 'error-placeholder text-center py-12';
        placeholder.innerHTML = `
            <i class="fas fa-exclamation-triangle text-red-400 text-3xl mb-4"></i>
            <p class="text-red-500">${message}</p>
            <button onclick="location.reload()" class="mt-4 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                <i class="fas fa-redo mr-2"></i>
                Retry
            </button>
        `;
        return placeholder;
    }

    /**
     * Animate card entrance
     * @param {HTMLElement} card - Card element to animate
     * @param {number} delay - Animation delay in milliseconds
     */
    static animateCardEntrance(card, delay = 0) {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.5s ease-out';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, delay);
    }

    /**
     * Get API base URL with automatic detection
     * @returns {string} - API base URL
     */
    static getApiBaseUrl() {
        const baseUrl = window.location.origin + '/api/v1';
        console.log('API Base URL detected:', baseUrl);
        return baseUrl;
    }

    /**
     * Make API request with error handling
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Fetch options
     * @returns {Promise} - API response promise
     */
    static async apiRequest(endpoint, options = {}) {
        const baseUrl = this.getApiBaseUrl();
        const url = `${baseUrl}${endpoint}`;
        
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.detail || `HTTP error! Status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }

    /**
     * Show toast notification
     * @param {string} message - Message to show
     * @param {string} type - Type of notification (success, error, warning, info)
     * @param {number} duration - Duration in milliseconds
     */
    static showToast(message, type = 'info', duration = 5000) {
        const toast = document.getElementById('toast-notification');
        const icon = document.getElementById('toast-icon');
        const messageEl = document.getElementById('toast-message');

        if (!toast || !icon || !messageEl) {
            console.warn('Toast elements not found');
            return;
        }

        // Set icon based on type
        const icons = {
            success: 'fas fa-check',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };

        icon.className = icons[type] || icons.info;
        messageEl.textContent = message;
        toast.className = `fixed bottom-5 right-5 text-white py-3 px-4 rounded-lg shadow-xl opacity-100 transition-opacity duration-300 z-50 ${type}`;

        // Auto-hide after duration
        setTimeout(() => {
            toast.classList.remove('opacity-100');
            toast.classList.add('opacity-0');
        }, duration);
    }
}

// Export for use in other scripts
window.PowerballUtils = PowerballUtils;