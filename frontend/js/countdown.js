/**
 * SHIOL+ Countdown Timer
 * ======================
 * 
 * Countdown timer component for next Powerball drawing.
 * Handles real-time countdown with automatic updates.
 */

class CountdownTimer {
    constructor(targetElementId, displayElementId) {
        this.targetElement = document.getElementById(targetElementId);
        this.displayElement = document.getElementById(displayElementId);
        this.targetDate = null;
        this.intervalId = null;
        this.isRunning = false;
    }

    /**
     * Start countdown to specific date
     * @param {Date|string} targetDate - Target date for countdown
     */
    start(targetDate) {
        try {
            this.targetDate = typeof targetDate === 'string' ? new Date(targetDate) : targetDate;
            
            if (isNaN(this.targetDate.getTime())) {
                throw new Error('Invalid target date');
            }

            // Clear any existing interval
            this.stop();

            // Start the countdown
            this.isRunning = true;
            this.update();
            this.intervalId = setInterval(() => this.update(), 1000);

            console.log('Countdown started for:', this.targetDate);
        } catch (error) {
            console.error('Error starting countdown:', error);
            this.displayError('Invalid date');
        }
    }

    /**
     * Stop the countdown timer
     */
    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.isRunning = false;
    }

    /**
     * Update countdown display
     */
    update() {
        if (!this.targetDate || !this.displayElement) {
            return;
        }

        const now = new Date();
        const timeDiff = this.targetDate.getTime() - now.getTime();

        if (timeDiff <= 0) {
            this.onCountdownComplete();
            return;
        }

        const timeString = this.formatTime(timeDiff);
        this.displayElement.textContent = timeString;

        // Update any additional elements
        this.updateAdditionalElements(timeDiff);
    }

    /**
     * Format time difference into readable string
     * @param {number} timeDiff - Time difference in milliseconds
     * @returns {string} - Formatted time string
     */
    formatTime(timeDiff) {
        const days = Math.floor(timeDiff / (1000 * 60 * 60 * 24));
        const hours = Math.floor((timeDiff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((timeDiff % (1000 * 60)) / 1000);

        if (days > 0) {
            return `${days}d ${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        } else {
            return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    /**
     * Handle countdown completion
     */
    onCountdownComplete() {
        this.stop();
        this.displayElement.textContent = 'Drawing in progress...';
        
        // Add visual indication that drawing is happening
        if (this.displayElement.parentElement) {
            this.displayElement.parentElement.classList.add('drawing-active');
        }

        // Trigger refresh of drawing info after a delay
        setTimeout(() => {
            this.refreshDrawingInfo();
        }, 5000);

        console.log('Countdown completed - drawing time reached');
    }

    /**
     * Display error message
     * @param {string} message - Error message to display
     */
    displayError(message) {
        if (this.displayElement) {
            this.displayElement.textContent = message;
            this.displayElement.classList.add('text-red-500');
        }
    }

    /**
     * Update additional elements based on countdown
     * @param {number} timeDiff - Time difference in milliseconds
     */
    updateAdditionalElements(timeDiff) {
        // Update urgency styling based on time remaining
        const hours = timeDiff / (1000 * 60 * 60);
        
        if (this.displayElement.parentElement) {
            const parent = this.displayElement.parentElement;
            
            // Remove existing urgency classes
            parent.classList.remove('countdown-urgent', 'countdown-critical');
            
            if (hours <= 1) {
                parent.classList.add('countdown-critical');
            } else if (hours <= 6) {
                parent.classList.add('countdown-urgent');
            }
        }
    }

    /**
     * Refresh drawing information (to be called after countdown completes)
     */
    async refreshDrawingInfo() {
        try {
            // This would typically trigger a refresh of the next drawing info
            if (window.PublicInterface && typeof window.PublicInterface.loadNextDrawingInfo === 'function') {
                await window.PublicInterface.loadNextDrawingInfo();
            } else {
                // Fallback: reload the page
                console.log('Refreshing page to get updated drawing info');
                window.location.reload();
            }
        } catch (error) {
            console.error('Error refreshing drawing info:', error);
        }
    }

    /**
     * Get current countdown status
     * @returns {Object} - Status object with time remaining and formatted string
     */
    getStatus() {
        if (!this.targetDate) {
            return { active: false, timeRemaining: 0, formatted: 'No target set' };
        }

        const now = new Date();
        const timeDiff = this.targetDate.getTime() - now.getTime();

        return {
            active: this.isRunning,
            timeRemaining: Math.max(0, timeDiff),
            formatted: timeDiff > 0 ? this.formatTime(timeDiff) : 'Completed',
            targetDate: this.targetDate
        };
    }
}

/**
 * Utility functions for countdown management
 */
class CountdownUtils {
    /**
     * Calculate next Powerball drawing date
     * @returns {Date} - Next drawing date
     */
    static getNextDrawingDate() {
        const now = new Date();
        let nextDrawing = new Date(now);
        
        // Powerball drawings are on Wednesdays (3) and Saturdays (6)
        // JavaScript: Sunday = 0, Monday = 1, ..., Saturday = 6
        const currentDay = now.getDay();
        const currentHour = now.getHours();
        const currentMinute = now.getMinutes();
        
        let daysToAdd = 0;
        
        if (currentDay < 3) {
            // Before Wednesday
            daysToAdd = 3 - currentDay;
        } else if (currentDay === 3) {
            // Wednesday - check if before 10:59 PM
            if (currentHour < 22 || (currentHour === 22 && currentMinute < 59)) {
                daysToAdd = 0; // Today
            } else {
                daysToAdd = 3; // Next Saturday
            }
        } else if (currentDay < 6) {
            // Thursday or Friday
            daysToAdd = 6 - currentDay;
        } else if (currentDay === 6) {
            // Saturday - check if before 10:59 PM
            if (currentHour < 22 || (currentHour === 22 && currentMinute < 59)) {
                daysToAdd = 0; // Today
            } else {
                daysToAdd = 4; // Next Wednesday
            }
        } else {
            // Sunday
            daysToAdd = 3; // Next Wednesday
        }
        
        nextDrawing.setDate(now.getDate() + daysToAdd);
        nextDrawing.setHours(22, 59, 0, 0); // 10:59 PM
        
        return nextDrawing;
    }

    /**
     * Format countdown for display in different contexts
     * @param {number} timeDiff - Time difference in milliseconds
     * @param {string} format - Format type ('full', 'compact', 'minimal')
     * @returns {string} - Formatted time string
     */
    static formatCountdown(timeDiff, format = 'full') {
        if (timeDiff <= 0) {
            return 'Drawing time!';
        }

        const days = Math.floor(timeDiff / (1000 * 60 * 60 * 24));
        const hours = Math.floor((timeDiff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((timeDiff % (1000 * 60)) / 1000);

        switch (format) {
            case 'compact':
                if (days > 0) {
                    return `${days}d ${hours}h ${minutes}m`;
                } else if (hours > 0) {
                    return `${hours}h ${minutes}m ${seconds}s`;
                } else {
                    return `${minutes}m ${seconds}s`;
                }
            
            case 'minimal':
                if (days > 0) {
                    return `${days}d ${hours}h`;
                } else if (hours > 0) {
                    return `${hours}h ${minutes}m`;
                } else {
                    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
                }
            
            case 'full':
            default:
                if (days > 0) {
                    return `${days}d ${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                } else {
                    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                }
        }
    }

    /**
     * Get time until next drawing in seconds
     * @returns {number} - Seconds until next drawing
     */
    static getSecondsUntilNextDrawing() {
        const nextDrawing = this.getNextDrawingDate();
        const now = new Date();
        return Math.max(0, Math.floor((nextDrawing.getTime() - now.getTime()) / 1000));
    }

    /**
     * Check if drawing is happening soon (within specified minutes)
     * @param {number} minutes - Minutes threshold
     * @returns {boolean} - Whether drawing is soon
     */
    static isDrawingSoon(minutes = 30) {
        const secondsUntil = this.getSecondsUntilNextDrawing();
        return secondsUntil <= (minutes * 60) && secondsUntil > 0;
    }

    /**
     * Check if drawing is happening now (within 5 minutes of drawing time)
     * @returns {boolean} - Whether drawing is happening now
     */
    static isDrawingNow() {
        const secondsUntil = this.getSecondsUntilNextDrawing();
        return secondsUntil <= 300; // 5 minutes
    }
}

// Export classes for use in other scripts
window.CountdownTimer = CountdownTimer;
window.CountdownUtils = CountdownUtils;