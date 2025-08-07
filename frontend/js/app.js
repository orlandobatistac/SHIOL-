// Suppress Chrome extension errors
(function() {
    'use strict';
    
    try {
        // Check if we're in a browser environment
        if (typeof window !== 'undefined' && typeof chrome !== 'undefined') {
            // Override console.error to filter chrome extension errors
            const originalConsoleError = console.error;
            console.error = function(...args) {
                const message = args.join(' ');
                if (message.includes('Unchecked runtime.lastError') || 
                    message.includes('message port closed') ||
                    message.includes('Extension context invalidated') ||
                    message.includes('runtime.lastError')) {
                    // Suppress these specific chrome extension errors
                    return;
                }
                originalConsoleError.apply(console, args);
            };

            // Handle runtime errors at window level
            window.addEventListener('error', function(event) {
                if (event.message && (
                    event.message.includes('runtime.lastError') ||
                    event.message.includes('message port closed')
                )) {
                    event.preventDefault();
                    return true;
                }
            });

            // Clear any existing runtime errors
            if (chrome.runtime && chrome.runtime.lastError) {
                void chrome.runtime.lastError;
            }
        }
    } catch (e) {
        // Silently fail if chrome object manipulation fails
        console.debug('Chrome extension error suppression setup failed:', e);
    }
})();

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
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
    let syndicateData = null;

    // --- Event Listeners ---
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

    // Syndicate predictions event listeners
    const generateSyndicateBtn = document.getElementById('generate-syndicate-btn');
    const exportSyndicateBtn = document.getElementById('export-syndicate-btn');

    if (generateSyndicateBtn) {
        generateSyndicateBtn.addEventListener('click', generateSyndicatePredictions);
    }
    if (exportSyndicateBtn) {
        exportSyndicateBtn.addEventListener('click', exportSyndicateData);
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
            // Start auto-refresh every 60 seconds
            autoRefreshInterval = setInterval(fetchPipelineStatus, 60000);
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

    // --- Syndicate Predictions Functions ---
    async function generateSyndicatePredictions() {
        const playsInput = document.getElementById('syndicate-plays');
        const loadingDiv = document.getElementById('syndicate-loading');
        const resultsDiv = document.getElementById('syndicate-results');
        const btn = document.getElementById('generate-syndicate-btn');

        if (!playsInput || !loadingDiv || !resultsDiv || !btn) {
            console.error('Required syndicate elements not found');
            return;
        }

        const numPlays = parseInt(playsInput.value);

        if (numPlays < 10 || numPlays > 500) {
            showToast('Number of plays must be between 10 and 500', 'error');
            return;
        }

        try {
            // Show loading state
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Generating...';
            loadingDiv.classList.remove('hidden');
            resultsDiv.classList.add('hidden');

            // Make API request to Smart AI endpoint
            const response = await fetch(`${API_BASE_URL}/predict/smart?num_plays=${numPlays}`);

            if (!response.ok) {
                let errorMessage = `HTTP error! status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    if (errorData?.detail) {
                        errorMessage = errorData.detail;
                    } else if (errorData?.message) {
                        errorMessage = errorData.message;
                    }
                } catch (parseError) {
                    console.warn('Could not parse error response:', parseError);
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();

            // Process and display results
            if (data.smart_predictions && data.smart_predictions.length > 0) {
                syndicateData = data; // Store data for export
                displaySmartAIResults(data);
                showToast(`AI generated ${data.total_predictions} smart predictions successfully!`, 'success');
            } else {
                throw new Error('No predictions generated');
            }

        } catch (error) {
            console.error('Syndicate generation error:', error);
            loadingDiv.classList.add('hidden');

            let errorMessage = 'Failed to generate syndicate predictions';
            if (error.message && error.message !== '[object Object]') {
                errorMessage += ': ' + error.message;
            } else if (error.detail) {
                errorMessage += ': ' + error.detail;
            } else {
                errorMessage += '. Please check server logs.';
            }

            showToast(errorMessage, 'error');
        } finally {
            // Reset button state
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-brain mr-2"></i>Generate Smart AI Predictions';
            loadingDiv.classList.add('hidden');
        }
    }

    function displaySmartAIResults(data) {
        const resultsDiv = document.getElementById('syndicate-results');
        if (!resultsDiv) return;

        const predictions = data.smart_predictions;
        const analysis = data.ai_analysis;

        let html = `
            <div class="space-y-6">
                <!-- AI Analysis Summary -->
                <div class="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30 p-6 rounded-lg border border-blue-200 dark:border-blue-800">
                    <div class="flex items-center space-x-3 mb-4">
                        <i class="fas fa-brain text-2xl text-blue-600 dark:text-blue-400"></i>
                        <h3 class="text-lg font-bold text-blue-800 dark:text-blue-200">Smart AI Analysis</h3>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div class="bg-white dark:bg-gray-800 p-3 rounded-lg">
                            <div class="font-semibold text-gray-700 dark:text-gray-300">Candidates Evaluated</div>
                            <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">${analysis.candidates_evaluated.toLocaleString()}</div>
                        </div>
                        <div class="bg-white dark:bg-gray-800 p-3 rounded-lg">
                            <div class="font-semibold text-gray-700 dark:text-gray-300">AI Methods Used</div>
                            <div class="text-sm text-blue-600 dark:text-blue-400">${analysis.methods_used.join(', ').toUpperCase()}</div>
                        </div>
                        <div class="bg-white dark:bg-gray-800 p-3 rounded-lg">
                            <div class="font-semibold text-gray-700 dark:text-gray-300">Average AI Score</div>
                            <div class="text-2xl font-bold text-green-600 dark:text-green-400">${(analysis.score_range.average * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                    <div class="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                        <p class="text-sm text-green-800 dark:text-green-200">
                            <i class="fas fa-lightbulb mr-2"></i>
                            <strong>AI Recommendation:</strong> ${data.recommendation}
                        </p>
                    </div>
                </div>

                <!-- Predictions Table -->
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                    <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                        <h4 class="text-lg font-semibold text-gray-900 dark:text-white">
                            Smart AI Predictions (Ranked by AI Score)
                        </h4>
                        <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                            Top ${predictions.length} predictions automatically selected and ranked by AI
                        </p>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                            <thead class="bg-gray-50 dark:bg-gray-900">
                                <tr>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Rank</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Numbers</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">PB</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">AI Score</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Tier</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Method</th>
                                </tr>
                            </thead>
                            <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
        `;

        predictions.forEach((pred, index) => {
            const tierColor = getTierColor(pred.tier);
            const methodBadge = getMethodBadge(pred.ai_method);

            html += `
                <tr class="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="flex items-center">
                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full ${index < 10 ? 'bg-yellow-100 text-yellow-800 border-2 border-yellow-300' : 'bg-gray-100 text-gray-800'} text-sm font-bold">
                                ${pred.rank}
                            </span>
                        </div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="flex space-x-1">
                            ${pred.numbers.map(num => `
                                <span class="powerball-number small bg-blue-600 text-white">${num}</span>
                            `).join('')}
                        </div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <span class="powerball-number small bg-red-600 text-white">${pred.powerball}</span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="text-sm font-medium text-gray-900 dark:text-white">
                            ${(pred.smart_ai_score * 100).toFixed(1)}%
                        </div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">
                            Base: ${(pred.base_score * 100).toFixed(1)}%
                        </div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${tierColor}">
                            ${pred.tier}
                        </span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        ${methodBadge}
                    </td>
                </tr>
            `;
        });

        html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;

        resultsDiv.innerHTML = html;
        resultsDiv.classList.remove('hidden');
    }

    function getTierColor(tier) {
        const colors = {
            'Premium': 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
            'High': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
            'Medium': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
            'Standard': 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
        };
        return colors[tier] || colors['Standard'];
    }

    function getMethodBadge(method) {
        const badges = {
            'ensemble': '<span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">Ensemble</span>',
            'deterministic': '<span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">Deterministic</span>',
            'adaptive': '<span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">Adaptive</span>'
        };
        return badges[method] || '<span class="text-xs text-gray-500">Unknown</span>';
    }

    function updateSyndicateResults(data) {
        if (!data || !data.syndicate_predictions) {
            console.error('Invalid syndicate data received');
            return;
        }

        // Update coverage analysis
        const coverage = data.coverage_analysis || {};
        updateElementText('premium-tier-count', coverage.premium_tier || 0);
        updateElementText('high-tier-count', coverage.high_tier || 0);
        updateElementText('medium-tier-count', coverage.medium_tier || 0);
        updateElementText('standard-tier-count', coverage.standard_tier || 0);

        // Update total plays
        updateElementText('syndicate-total-plays', `${data.total_plays} plays`);

        // Update plays table
        const tbody = document.getElementById('syndicate-plays-tbody');
        if (!tbody) return;

        const playsHTML = data.syndicate_predictions.map((prediction, index) => {
            const numbers = prediction.numbers || [];
            const powerball = prediction.powerball || 0;
            const score = prediction.score || 0;
            const tier = prediction.tier || 'Standard';
            const method = prediction.method || data.method;
            const rank = prediction.rank || (index + 1);

            // Create number balls display
            const whiteBalls = numbers.map(num => 
                `<span class="inline-flex items-center justify-center w-6 h-6 bg-gray-200 dark:bg-gray-600 rounded-full text-xs font-semibold text-gray-800 dark:text-white mr-1">${num}</span>`
            ).join('');

            const powerBall = `<span class="inline-flex items-center justify-center w-6 h-6 bg-red-500 rounded-full text-xs font-semibold text-white">${powerball}</span>`;

            // Tier color coding
            const tierColors = {
                'Premium': 'text-purple-600 bg-purple-100 dark:bg-purple-900 dark:text-purple-300',
                'High': 'text-blue-600 bg-blue-100 dark:bg-blue-900 dark:text-blue-300',
                'Medium': 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-300',
                'Standard': 'text-gray-600 bg-gray-100 dark:bg-gray-900 dark:text-gray-100 dark:text-gray-300'
            };

            const tierClass = tierColors[tier] || tierColors['Standard'];

            return `
                <tr class="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td class="px-4 py-3 text-sm font-medium text-gray-900 dark:text-gray-100">${rank}</td>
                    <td class="px-4 py-3">
                        <div class="flex items-center">
                            ${whiteBalls}
                            <span class="mx-2 text-red-500">•</span>
                            ${powerBall}
                        </div>
                    </td>
                    <td class="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">${(score * 100).toFixed(2)}%</td>
                    <td class="px-4 py-3">
                        <span class="px-2 py-1 text-xs font-medium rounded-full ${tierClass}">${tier}</span>
                    </td>
                    <td class="px-4 py-3 text-sm text-gray-500 dark:text-gray-400">${method}</td>
                </tr>
            `;
        }).join('');

        tbody.innerHTML = playsHTML;
    }

    function exportSyndicateData() {
        if (!syndicateData || !syndicateData.syndicate_predictions) {
            PowerballUtils.showToast('No syndicate data to export', 'warning');
            return;
        }

        try {
            // Create CSV content
            const headers = ['Rank', 'Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Powerball', 'Score', 'Tier', 'Method'];
            const csvRows = [headers.join(',')];

            syndicateData.syndicate_predictions.forEach((prediction, index) => {
                const numbers = prediction.numbers || [];
                const powerball = prediction.powerball || 0;
                const score = prediction.score || 0;
                const tier = prediction.tier || 'Standard';
                const method = prediction.method || syndicateData.method;
                const rank = prediction.rank || (index + 1);

                const row = [
                    rank,
                    ...numbers,
                    powerball,
                    (score * 100).toFixed(2),
                    tier,
                    method
                ];
                csvRows.push(row.join(','));
            });

            // Create and download file
            const csvContent = csvRows.join('\n');
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');

            if (link.download !== undefined) {
                const url = URL.createObjectURL(blob);
                link.setAttribute('href', url);
                link.setAttribute('download', `shiol_syndicate_${syndicateData.total_plays}_plays_${new Date().toISOString().slice(0, 10)}.csv`);
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                PowerballUtils.showToast('Syndicate data exported successfully!', 'success');
            }
        } catch (error) {
            console.error('Export error:', error);
            PowerballUtils.showToast('Failed to export syndicate data', 'error');
        }
    }

    function updateElementText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
        }
    }

    // Initialize UI state
    initializePipelineDashboard();

    // Function to show toast notifications
    function showToast(message, type = 'info') {
        const toast = document.getElementById('toast-notification');
        if (!toast) return;

        const typeClasses = {
            success: 'bg-green-500',
            error: 'bg-red-500',
            warning: 'bg-yellow-500',
            info: 'bg-blue-500'
        };

        toast.className = `fixed bottom-5 right-5 text-white py-2 px-4 rounded-lg shadow-xl opacity-100 transition-opacity duration-300 z-50 ${typeClasses[type] || typeClasses.info}`;
        toast.innerHTML = `<i class="fas fa-info mr-2"></i>${message}`;

        setTimeout(() => {
            toast.classList.remove('opacity-100');
            toast.classList.add('opacity-0');
        }, 3000);
    }
});