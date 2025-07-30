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

        // Load more button
        const loadMoreBtn = document.getElementById('load-more-btn');
        if (loadMoreBtn) {
            loadMoreBtn.addEventListener('click', () => this.loadMoreHistory());
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
        await Promise.all([
            this.loadNextDrawingInfo(),
            this.loadHistoricalResults()
        ]);
    }

    /**
     * Load next drawing information and featured prediction
     */
    async loadNextDrawingInfo() {
        try {
            console.log('Loading next drawing info...');
            
            const data = await PowerballUtils.apiRequest('/public/next-drawing');
            
            // Update drawing date and time
            const nextDrawingDate = document.getElementById('next-drawing-date');
            const nextDrawingTime = document.getElementById('next-drawing-time');
            const lastUpdated = document.getElementById('last-updated');
            const footerLastUpdated = document.getElementById('footer-last-updated');

            if (nextDrawingDate && data.next_drawing) {
                nextDrawingDate.textContent = PowerballUtils.formatNextDrawingDate(data.next_drawing.date);
            }

            if (nextDrawingTime && data.next_drawing) {
                nextDrawingTime.textContent = data.next_drawing.time + ' ET';
            }

            // Update last updated timestamps
            const updateTime = new Date().toLocaleString();
            if (lastUpdated) {
                lastUpdated.textContent = 'Just now';
            }
            if (footerLastUpdated) {
                footerLastUpdated.textContent = updateTime;
            }

            // Start countdown timer
            if (data.next_drawing && data.next_drawing.datetime_iso) {
                this.countdownTimer.start(data.next_drawing.datetime_iso);
            }

            // Display featured prediction
            if (data.featured_prediction) {
                this.displayFeaturedPrediction(data.featured_prediction);
            }

            console.log('Next drawing info loaded successfully');
        } catch (error) {
            console.error('Error loading next drawing info:', error);
            this.showFeaturedPredictionError();
        }
    }

    /**
     * Display featured prediction(s) - supports both single and multiple ADAPTIVE predictions
     * @param {Object} data - Prediction data (can contain single prediction or array of predictions)
     */
    displayFeaturedPrediction(data) {
        const featuredNumbers = document.getElementById('featured-numbers');
        const confidenceScore = document.getElementById('confidence-score');
        const predictionMethod = document.getElementById('prediction-method');

        if (!featuredNumbers) return;

        // Handle both single prediction (fallback) and multiple predictions
        const predictions = data.predictions || [data];
        const isMultiple = data.predictions && data.predictions.length > 1;

        if (isMultiple) {
            // Display multiple ADAPTIVE predictions in a grid
            this.displayMultiplePredictions(predictions, data);
        } else {
            // Display single prediction (legacy format)
            const prediction = predictions[0];
            
            // Validate prediction data
            if (!PowerballUtils.validateNumbers(prediction.numbers, prediction.powerball)) {
                console.error('Invalid prediction numbers:', prediction);
                this.showFeaturedPredictionError();
                return;
            }

            // Create numbers display
            const numbersHtml = PowerballUtils.createNumbersDisplay(
                PowerballUtils.sortNumbers(prediction.numbers),
                prediction.powerball
            );

            featuredNumbers.innerHTML = numbersHtml;

            // Update confidence score
            if (confidenceScore && prediction.confidence_score) {
                const percentage = (prediction.confidence_score * 100).toFixed(1);
                confidenceScore.textContent = `${percentage}%`;
            }

            // Update prediction method
            if (predictionMethod && prediction.method) {
                predictionMethod.textContent = prediction.method === 'ai_optimized' ? 'AI Optimized' :
                    prediction.method.replace(/_/g, ' ').toUpperCase();
            }

            // Animate the numbers
            const numberElements = featuredNumbers.querySelectorAll('.powerball-number');
            numberElements.forEach((el, index) => {
                PowerballUtils.animateCardEntrance(el, index * 100);
            });
        }
    }

    /**
     * Display multiple ADAPTIVE predictions in a grid layout
     * @param {Array} predictions - Array of prediction objects
     * @param {Object} data - Full response data
     */
    displayMultiplePredictions(predictions, data) {
        const featuredNumbers = document.getElementById('featured-numbers');
        const confidenceScore = document.getElementById('confidence-score');
        const predictionMethod = document.getElementById('prediction-method');

        // Create grid layout for multiple predictions
        const gridHtml = `
            <div class="adaptive-predictions-grid">
                <div class="predictions-header">
                    <h4>🧠 ${predictions.length} ADAPTIVE AI Predictions</h4>
                    <p class="predictions-subtitle">Self-optimizing predictions with auto-tracking</p>
                </div>
                <div class="predictions-grid">
                    ${predictions.map((prediction, index) => {
                        // Validate each prediction
                        if (!PowerballUtils.validateNumbers(prediction.numbers, prediction.powerball)) {
                            console.warn(`Invalid prediction ${index + 1}:`, prediction);
                            return '';
                        }

                        const numbersHtml = PowerballUtils.createNumbersDisplay(
                            PowerballUtils.sortNumbers(prediction.numbers),
                            prediction.powerball,
                            'small'
                        );

                        return `
                            <div class="prediction-card ${index === 0 ? 'primary' : ''}" data-rank="${prediction.rank || index + 1}">
                                <div class="prediction-rank">#${prediction.rank || index + 1}</div>
                                <div class="prediction-numbers-small">
                                    ${numbersHtml}
                                </div>
                                <div class="prediction-stats">
                                    <div class="confidence">${(prediction.confidence_score * 100).toFixed(1)}%</div>
                                    ${prediction.prediction_id ? `<div class="tracking-id" title="Prediction ID: ${prediction.prediction_id}">ID: ${prediction.prediction_id.substring(0, 8)}...</div>` : ''}
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
                <div class="predictions-footer">
                    <div class="tracking-status">
                        <span class="status-icon">✅</span>
                        <span class="status-text">All predictions automatically tracked and will be compared with official results</span>
                    </div>
                    <div class="method-info">
                        <span class="method-icon">🧠</span>
                        <span class="method-text">ADAPTIVE method uses self-optimizing AI with continuous learning</span>
                    </div>
                </div>
            </div>
        `;

        featuredNumbers.innerHTML = gridHtml;

        // Update overall stats
        if (confidenceScore) {
            const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence_score, 0) / predictions.length;
            confidenceScore.textContent = `${(avgConfidence * 100).toFixed(1)}% avg`;
        }

        if (predictionMethod) {
            predictionMethod.textContent = 'ADAPTIVE AI SYSTEM';
        }

        // Animate the prediction cards
        const predictionCards = featuredNumbers.querySelectorAll('.prediction-card');
        predictionCards.forEach((card, index) => {
            PowerballUtils.animateCardEntrance(card, index * 150);
        });
    }

    /**
     * Show error in featured prediction area
     */
    showFeaturedPredictionError() {
        const featuredNumbers = document.getElementById('featured-numbers');
        if (featuredNumbers) {
            featuredNumbers.innerHTML = PowerballUtils.createErrorPlaceholder('Unable to load prediction').innerHTML;
        }
    }

    /**
     * Load historical results with AI predictions comparison
     */
    async loadHistoricalResults() {
        if (this.isLoading) return;
        
        try {
            this.isLoading = true;
            console.log('Loading AI predictions performance data...');
            
            const historicalResults = document.getElementById('historical-results');
            if (!historicalResults) return;

            // Show loading if this is the first load
            if (this.currentHistoryPage === 1) {
                historicalResults.innerHTML = '';
                historicalResults.appendChild(PowerballUtils.createLoadingPlaceholder('Loading AI predictions vs results...'));
            }

            // Load AI predictions performance data instead of basic results
            const data = await PowerballUtils.apiRequest(`/public/predictions-performance?limit=10`);
            
            if (data.prediction_groups && data.prediction_groups.length > 0) {
                this.displayHybridPredictionsPerformance(data, this.currentHistoryPage === 1);
            } else if (data.predictions && data.predictions.length > 0) {
                // Fallback to old format if needed
                this.displayPredictionsPerformance(data, this.currentHistoryPage === 1);
            } else {
                this.showHistoryError('No AI predictions data available yet');
            }

            console.log(`Loaded ${data.prediction_groups?.length || data.predictions?.length || 0} AI prediction comparisons`);
        } catch (error) {
            console.error('Error loading predictions performance:', error);
            this.showHistoryError('Failed to load AI predictions performance');
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Display historical results
     * @param {Array} results - Array of historical results
     * @param {boolean} replace - Whether to replace existing results
     */
    displayHistoricalResults(results, replace = false) {
        const historicalResults = document.getElementById('historical-results');
        if (!historicalResults) return;

        if (replace) {
            historicalResults.innerHTML = '';
        }

        // Create grid container if it doesn't exist
        let gridContainer = historicalResults.querySelector('.results-grid');
        if (!gridContainer) {
            gridContainer = document.createElement('div');
            gridContainer.className = 'results-grid grid gap-4 md:gap-6 lg:grid-cols-2 xl:grid-cols-3';
            historicalResults.appendChild(gridContainer);
        }

        // Add each result as a card
        results.forEach((result, index) => {
            const card = PowerballUtils.createPowerballCard({
                numbers: result.winning_numbers,
                powerball: result.powerball,
                type: 'result',
                date: result.draw_date,
                metadata: {
                    jackpot_amount: result.jackpot_amount,
                    multiplier: result.multiplier
                },
                featured: false
            });

            // Add animation delay
            PowerballUtils.animateCardEntrance(card, index * 50);
            
            gridContainer.appendChild(card);
        });
    }

    /**
     * Display AI predictions performance with visual comparisons and prize amounts
     * @param {Object} data - Performance data with predictions, comparisons, and summary
     * @param {boolean} replace - Whether to replace existing content
     */
    displayPredictionsPerformance(data, replace = false) {
        const historicalResults = document.getElementById('historical-results');
        if (!historicalResults) return;

        if (replace) {
            historicalResults.innerHTML = '';
        }

        // Create performance summary header
        const summaryHtml = `
            <div class="performance-summary">
                <div class="summary-header">
                    <h3>🧠 AI Predictions vs Official Results</h3>
                    <p class="summary-subtitle">Real performance tracking with prize calculations</p>
                </div>
                <div class="summary-stats">
                    <div class="stat-card prize-total">
                        <div class="stat-icon">💰</div>
                        <div class="stat-content">
                            <div class="stat-value">${data.summary.total_prize_display}</div>
                            <div class="stat-label">Total Prizes Won</div>
                        </div>
                    </div>
                    <div class="stat-card win-rate">
                        <div class="stat-icon">🎯</div>
                        <div class="stat-content">
                            <div class="stat-value">${data.summary.win_rate_percentage}%</div>
                            <div class="stat-label">Win Rate</div>
                        </div>
                    </div>
                    <div class="stat-card predictions-count">
                        <div class="stat-icon">📊</div>
                        <div class="stat-content">
                            <div class="stat-value">${data.summary.predictions_with_prizes}</div>
                            <div class="stat-label">Winning Predictions</div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Create predictions comparison grid
        const predictionsHtml = `
            <div class="predictions-performance-grid">
                ${data.predictions.map((prediction, index) => {
                    return this.createPredictionComparisonCard(prediction, index);
                }).join('')}
            </div>
        `;

        // Combine summary and predictions
        const fullHtml = summaryHtml + predictionsHtml;
        
        if (replace) {
            historicalResults.innerHTML = fullHtml;
        } else {
            historicalResults.insertAdjacentHTML('beforeend', fullHtml);
        }

        // Animate the cards
        const cards = historicalResults.querySelectorAll('.prediction-comparison-card');
        cards.forEach((card, index) => {
            PowerballUtils.animateCardEntrance(card, index * 100);
        });

        // Animate the summary stats
        const statCards = historicalResults.querySelectorAll('.stat-card');
        statCards.forEach((card, index) => {
            PowerballUtils.animateCardEntrance(card, index * 50);
        });
    }

    /**
     * Display HYBRID AI predictions performance with grouped format
     * @param {Object} data - Hybrid performance data with prediction groups
     * @param {boolean} replace - Whether to replace existing content
     */
    displayHybridPredictionsPerformance(data, replace = false) {
        const historicalResults = document.getElementById('historical-results');
        if (!historicalResults) return;

        if (replace) {
            historicalResults.innerHTML = '';
        }

        // Create performance summary header for hybrid format
        const summaryHtml = `
            <div class="performance-summary hybrid-format">
                <div class="summary-header">
                    <h3>🧠 HYBRID AI Predictions vs Official Results</h3>
                    <p class="summary-subtitle">5 ADAPTIVE predictions per official result with prize calculations</p>
                </div>
                <div class="summary-stats">
                    <div class="stat-card prize-total">
                        <div class="stat-icon">💰</div>
                        <div class="stat-content">
                            <div class="stat-value">${data.summary.total_prize_display}</div>
                            <div class="stat-label">Total Prizes Won</div>
                        </div>
                    </div>
                    <div class="stat-card win-rate">
                        <div class="stat-icon">🎯</div>
                        <div class="stat-content">
                            <div class="stat-value">${data.summary.overall_win_rate_percentage}%</div>
                            <div class="stat-label">Overall Win Rate</div>
                        </div>
                    </div>
                    <div class="stat-card groups-count">
                        <div class="stat-icon">📊</div>
                        <div class="stat-content">
                            <div class="stat-value">${data.summary.total_official_results}</div>
                            <div class="stat-label">Official Results</div>
                        </div>
                    </div>
                    <div class="stat-card predictions-count">
                        <div class="stat-icon">🔮</div>
                        <div class="stat-content">
                            <div class="stat-value">${data.summary.total_predictions}</div>
                            <div class="stat-label">Total Predictions</div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Create hybrid prediction groups
        const groupsHtml = `
            <div class="hybrid-predictions-grid">
                ${data.prediction_groups.map((group, index) => {
                    return this.createHybridGroupCard(group, index);
                }).join('')}
            </div>
        `;

        // Combine summary and groups
        const fullHtml = summaryHtml + groupsHtml;
        
        if (replace) {
            historicalResults.innerHTML = fullHtml;
        } else {
            historicalResults.insertAdjacentHTML('beforeend', fullHtml);
        }

        // Animate the cards
        const groupCards = historicalResults.querySelectorAll('.hybrid-group-card');
        groupCards.forEach((card, index) => {
            PowerballUtils.animateCardEntrance(card, index * 150);
        });

        // Animate the summary stats
        const statCards = historicalResults.querySelectorAll('.stat-card');
        statCards.forEach((card, index) => {
            PowerballUtils.animateCardEntrance(card, index * 50);
        });
    }

    /**
     * Create a hybrid group card showing official result + 5 predictions
     * @param {Object} group - Group data with official result and predictions
     * @param {number} index - Card index for animation
     * @returns {string} HTML string for the hybrid group card
     */
    createHybridGroupCard(group, index) {
        const officialResult = group.official_result;
        const predictions = group.predictions;
        const summary = group.group_summary;
        
        const hasWinnings = summary.predictions_with_prizes > 0;
        const prizeClass = hasWinnings ? 'has-prize' : '';
        
        // Create official result display for header (white balls + powerball)
        const officialNumbersHtml = officialResult.winning_numbers.map(num => {
            return `<span class="powerball-number white-ball" data-number="${num}">${num.toString().padStart(2, '0')}</span>`;
        }).join('');
        
        const officialPowerballHtml = `<span class="powerball-number power-ball" data-number="${officialResult.winning_powerball}">${officialResult.winning_powerball.toString().padStart(2, '0')}</span>`;

        // Create prediction rows (horizontal layout like demo)
        const predictionRowsHtml = predictions.map((prediction, predIndex) => {
            const predHasWinnings = prediction.has_prize;
            
            // Create numbers with match highlighting
            const numbersHtml = prediction.number_matches.map(match => {
                const matchClass = match.is_match ? 'white-ball golden-glow' : 'white-ball';
                return `<span class="powerball-number ${matchClass}" data-number="${match.number}">${match.number.toString().padStart(2, '0')}</span>`;
            }).join('');
            
            const powerballClass = prediction.powerball_match ? 'power-ball golden-glow' : 'power-ball';
            const powerballHtml = `<span class="powerball-number ${powerballClass}" data-number="${prediction.prediction_powerball}">${prediction.prediction_powerball.toString().padStart(2, '0')}</span>`;
            
            // Create match badge
            const totalMatches = prediction.total_matches;
            const hasPowerball = prediction.powerball_match;
            let matchText = `${totalMatches} match${totalMatches !== 1 ? 'es' : ''}`;
            if (hasPowerball) {
                matchText += ' + PB';
            }
            const matchBadgeClass = (totalMatches > 0 || hasPowerball) ? 'match-badge has-matches' : 'match-badge';
            
            // Special case for jackpot
            if (totalMatches === 5 && hasPowerball) {
                matchText = '🏆 JACKPOT!';
            }
            
            const prizeAmountClass = predHasWinnings ? 'prize-amount has-prize' : 'prize-amount';
            
            return `
                <div class="prediction-row">
                    <div class="play-label">Play ${prediction.play_number}</div>
                    <div class="numbers-container">
                        ${numbersHtml}
                        ${powerballHtml}
                    </div>
                    <div class="stats-container">
                        <span class="${matchBadgeClass}">${matchText}</span>
                        <span class="${prizeAmountClass}">${prediction.prize_display}</span>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div class="comparison-group ${prizeClass}">
                <!-- Winning Numbers Header -->
                <div class="winning-header">
                    <div class="text-sm opacity-90 mb-2">📅 DRAW: ${PowerballUtils.formatDate(officialResult.draw_date)}</div>
                    <div class="text-lg font-bold mb-3">🎯 WINNING NUMBERS</div>
                    <div class="flex justify-center gap-3 mb-2">
                        ${officialNumbersHtml}
                        ${officialPowerballHtml}
                    </div>
                    <div class="text-sm opacity-90">5 ADAPTIVE AI Predictions</div>
                </div>

                <!-- Prediction Rows -->
                ${predictionRowsHtml}

                <!-- Summary Footer -->
                <div class="summary-footer">
                    <div class="summary-stats">
                        <div class="summary-stat">
                            <div class="summary-stat-value text-yellow-600">${summary.group_prize_display}</div>
                            <div class="summary-stat-label">Total Prizes</div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-stat-value text-green-600">${summary.best_result}</div>
                            <div class="summary-stat-label">Best Result</div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-stat-value text-blue-600">${summary.average_matches}</div>
                            <div class="summary-stat-label">Avg Matches</div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-stat-value text-purple-600">${summary.group_win_rate}%</div>
                            <div class="summary-stat-label">Win Rate</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Create a prediction comparison card with visual highlighting
     * @param {Object} prediction - Prediction data with comparison results
     * @param {number} index - Card index for animation
     * @returns {string} HTML string for the comparison card
     */
    createPredictionComparisonCard(prediction, index) {
        const hasWinnings = prediction.has_prize;
        const prizeClass = hasWinnings ? 'has-prize' : 'no-prize';
        const prizeIcon = hasWinnings ? '🏆' : '📊';
        
        // Create visual number comparison
        const numbersComparisonHtml = this.createNumbersComparison(
            prediction.prediction_numbers,
            prediction.winning_numbers,
            prediction.number_matches
        );
        
        const powerballComparisonHtml = this.createPowerballComparison(
            prediction.prediction_powerball,
            prediction.winning_powerball,
            prediction.powerball_match
        );

        return `
            <div class="prediction-comparison-card ${prizeClass}" data-index="${index}">
                <div class="card-header">
                    <div class="prediction-date">
                        <span class="date-label">Prediction:</span>
                        <span class="date-value">${PowerballUtils.formatDate(prediction.prediction_date)}</span>
                    </div>
                    <div class="draw-date">
                        <span class="date-label">Draw:</span>
                        <span class="date-value">${PowerballUtils.formatDate(prediction.draw_date)}</span>
                    </div>
                </div>
                
                <div class="numbers-comparison">
                    <div class="comparison-section">
                        <div class="section-label">Main Numbers</div>
                        <div class="numbers-row prediction-row">
                            <span class="row-label">AI Predicted:</span>
                            ${numbersComparisonHtml}
                        </div>
                        <div class="numbers-row result-row">
                            <span class="row-label">Official Result:</span>
                            ${this.createOfficialNumbersDisplay(prediction.winning_numbers)}
                        </div>
                    </div>
                    
                    <div class="comparison-section powerball-section">
                        <div class="section-label">Powerball</div>
                        ${powerballComparisonHtml}
                    </div>
                </div>
                
                <div class="comparison-results">
                    <div class="matches-summary">
                        <span class="matches-count">${prediction.total_matches} matches</span>
                        ${prediction.powerball_match ? '<span class="powerball-hit">+ Powerball</span>' : ''}
                    </div>
                    
                    <div class="prize-display ${prizeClass}">
                        <div class="prize-icon">${prizeIcon}</div>
                        <div class="prize-content">
                            <div class="prize-amount">${prediction.prize_display}</div>
                            <div class="prize-description">${prediction.prize_description || 'No matches'}</div>
                        </div>
                    </div>
                </div>
                
                ${hasWinnings ? '<div class="winning-glow"></div>' : ''}
            </div>
        `;
    }

    /**
     * Create visual comparison for main numbers with golden highlighting
     * @param {Array} predicted - Predicted numbers
     * @param {Array} winning - Winning numbers
     * @param {Array} matches - Match information for each predicted number
     * @returns {string} HTML for numbers comparison
     */
    createNumbersComparison(predicted, winning, matches) {
        return predicted.map((num, index) => {
            const matchInfo = matches[index];
            const isMatch = matchInfo && matchInfo.is_match;
            const matchClass = isMatch ? 'number-match golden-glow' : 'number-no-match';
            
            return `<span class="powerball-number small ${matchClass}" data-number="${num}">${num}</span>`;
        }).join('');
    }

    /**
     * Create visual comparison for powerball with highlighting
     * @param {number} predicted - Predicted powerball
     * @param {number} winning - Winning powerball
     * @param {boolean} isMatch - Whether powerball matches
     * @returns {string} HTML for powerball comparison
     */
    createPowerballComparison(predicted, winning, isMatch) {
        const matchClass = isMatch ? 'powerball-match golden-glow' : 'powerball-no-match';
        
        return `
            <div class="powerball-comparison">
                <div class="powerball-row prediction-row">
                    <span class="row-label">Predicted:</span>
                    <span class="powerball-number small ${matchClass}" data-number="${predicted}">${predicted}</span>
                </div>
                <div class="powerball-row result-row">
                    <span class="row-label">Official:</span>
                    <span class="powerball-number small official" data-number="${winning}">${winning}</span>
                </div>
            </div>
        `;
    }

    /**
     * Create display for official winning numbers
     * @param {Array} numbers - Official winning numbers
     * @returns {string} HTML for official numbers
     */
    createOfficialNumbersDisplay(numbers) {
        return numbers.map(num => {
            return `<span class="powerball-number small official" data-number="${num}">${num}</span>`;
        }).join('');
    }

    /**
     * Show error in historical results area
     * @param {string} message - Error message
     */
    showHistoryError(message) {
        const historicalResults = document.getElementById('historical-results');
        if (historicalResults) {
            historicalResults.innerHTML = '';
            historicalResults.appendChild(PowerballUtils.createErrorPlaceholder(message));
        }
    }

    /**
     * Load more historical results
     */
    async loadMoreHistory() {
        // This would implement pagination for loading more results
        // For now, we'll just show a message since the API returns the most recent 30
        PowerballUtils.showToast('Showing most recent 30 results', 'info');
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