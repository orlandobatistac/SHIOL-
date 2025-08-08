// Suppress Chrome extension errors
try {
    if (typeof chrome !== 'undefined' && chrome.runtime) {
        // Override console.error to filter chrome extension errors
        const originalConsoleError = console.error;
        console.error = function(...args) {
            const message = args.join(' ');
            if (message.includes('Unchecked runtime.lastError') ||
                message.includes('message port closed') ||
                message.includes('Extension context invalidated')) {
                // Suppress these specific chrome extension errors
                return;
            }
            originalConsoleError.apply(console, args);
        };

        // Also handle the runtime.lastError directly
        if (chrome.runtime.lastError) {
            // Clear the error silently
            const error = chrome.runtime.lastError;
            if (error.message && error.message.includes('message port closed')) {
                // Acknowledge the error to clear it
                void chrome.runtime.lastError;
            }
        }
    }
} catch (e) {
    // If chrome object manipulation fails, just continue
}

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
            executionHistoryTbody.textContent = ''; // Clear existing content safely
            const row = document.createElement('tr');
            const cell = document.createElement('td');
            cell.colSpan = 5;
            cell.className = 'px-4 py-8 text-center text-gray-500 dark:text-gray-400';
            cell.textContent = 'No execution history available';
            row.appendChild(cell);
            executionHistoryTbody.appendChild(row);
            return;
        }

        executionHistoryTbody.textContent = ''; // Clear previous content safely

        executions.forEach(execution => {
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

            const row = document.createElement('tr');

            // Start time cell
            const startTimeCell = document.createElement('td');
            startTimeCell.className = 'px-4 py-3 text-sm text-gray-900 dark:text-gray-100';
            startTimeCell.textContent = startTime;
            row.appendChild(startTimeCell);

            // Status cell
            const statusCell = document.createElement('td');
            statusCell.className = 'px-4 py-3';
            const statusSpan = document.createElement('span');
            statusSpan.className = `status-badge ${status}`;
            statusSpan.textContent = status;
            statusCell.appendChild(statusSpan);
            row.appendChild(statusCell);

            // Duration cell
            const durationCell = document.createElement('td');
            durationCell.className = 'px-4 py-3 text-sm text-gray-900 dark:text-gray-100';
            durationCell.textContent = duration;
            row.appendChild(durationCell);

            // Steps cell
            const stepsCell = document.createElement('td');
            stepsCell.className = 'px-4 py-3 text-sm text-gray-900 dark:text-gray-100';
            stepsCell.textContent = `${execution.steps_completed || 0}/${execution.total_steps || 7}`;
            row.appendChild(stepsCell);

            // Actions cell
            const actionsCell = document.createElement('td');
            actionsCell.className = 'px-4 py-3';
            const detailsButton = document.createElement('button');
            detailsButton.className = 'text-indigo-600 hover:text-indigo-900 dark:text-indigo-400 dark:hover:text-indigo-300 text-sm font-medium';
            detailsButton.textContent = 'View Details';
            detailsButton.onclick = () => window.viewExecutionDetails(executionId);
            actionsCell.appendChild(detailsButton);
            row.appendChild(actionsCell);

            executionHistoryTbody.appendChild(row);
        });
    }

    function updateLastGeneratedPlays(lastRun) {
        if (!lastGeneratedPlays || !lastPlaysContainer || !lastRun.plays) return;

        const timestamp = new Date(lastRun.timestamp).toLocaleString();
        lastPlaysTimestamp.textContent = `Generated: ${timestamp}`;

        // Clear container safely
        while (lastPlaysContainer.firstChild) {
            lastPlaysContainer.removeChild(lastPlaysContainer.firstChild);
        }

        // Display all 5 plays (not just 3) using safe DOM methods
        lastRun.plays.slice(0, 5).forEach((play, index) => {
            // Create main container
            const playContainer = document.createElement('div');
            playContainer.className = 'flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg mb-2';

            // Create left side container (play label and numbers)
            const leftContainer = document.createElement('div');
            leftContainer.className = 'flex items-center space-x-2';

            // Create play label
            const playLabel = document.createElement('span');
            playLabel.className = 'text-sm font-medium text-gray-600 dark:text-gray-400';
            playLabel.textContent = `Play ${index + 1}:`;
            leftContainer.appendChild(playLabel);

            // Create white balls safely
            play.slice(0, 5).forEach(num => {
                const sanitizedNum = parseInt(num, 10);
                const ballSpan = document.createElement('span');
                ballSpan.className = 'number-ball';
                
                if (isNaN(sanitizedNum) || sanitizedNum < 1 || sanitizedNum > 69) {
                    ballSpan.textContent = '?';
                } else {
                    ballSpan.textContent = sanitizedNum.toString();
                }
                leftContainer.appendChild(ballSpan);
            });

            // Create powerball safely
            const powerballSpan = document.createElement('span');
            powerballSpan.className = 'powerball-number';
            const sanitizedPowerball = parseInt(play[5], 10);
            powerballSpan.textContent = isNaN(sanitizedPowerball) ? '?' : sanitizedPowerball.toString();
            leftContainer.appendChild(powerballSpan);

            // Create right side container (score)
            const rightContainer = document.createElement('div');
            rightContainer.className = 'text-xs text-gray-500 dark:text-gray-400';
            
            const scoreText = document.createElement('span');
            if (lastRun.scores && lastRun.scores[index]) {
                const score = Math.round(parseFloat(lastRun.scores[index]) * 100);
                scoreText.textContent = `Score: ${score}%`;
            } else {
                scoreText.textContent = 'Score: N/A';
            }
            rightContainer.appendChild(scoreText);

            // Assemble the play container
            playContainer.appendChild(leftContainer);
            playContainer.appendChild(rightContainer);

            // Add to main container
            lastPlaysContainer.appendChild(playContainer);
        });
        lastGeneratedPlays.classList.remove('hidden');
    }

    async function triggerPipeline() {
        hideTriggerModal();

        try {
            triggerPipelineBtn.disabled = true;
            triggerPipelineBtn.textContent = 'Triggering...';
            triggerPipelineBtn.className += ' opacity-75';

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
            triggerPipelineBtn.textContent = 'Trigger Pipeline';
            triggerPipelineBtn.className = triggerPipelineBtn.className.replace(' opacity-75', '');
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

            // Create a simple modal to display logs safely
            const logsModal = document.createElement('div');
            logsModal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
            
            const modalContent = document.createElement('div');
            modalContent.className = 'bg-white dark:bg-gray-800 rounded-lg p-6 max-w-4xl w-full mx-4 max-h-96 overflow-hidden';

            const headerDiv = document.createElement('div');
            headerDiv.className = 'flex justify-between items-center mb-4';

            const title = document.createElement('h3');
            title.className = 'text-lg font-semibold text-gray-800 dark:text-white';
            title.textContent = 'Pipeline Logs';

            const closeButton = document.createElement('button');
            closeButton.className = 'text-gray-500 hover:text-gray-700';
            closeButton.onclick = () => logsModal.remove();

            const closeIcon = document.createElement('i');
            closeIcon.className = 'fas fa-times';
            closeButton.appendChild(closeIcon);

            headerDiv.appendChild(title);
            headerDiv.appendChild(closeButton);

            const logsContainer = document.createElement('div');
            logsContainer.className = 'bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-y-auto max-h-80';

            // Safely display logs without innerHTML - prevent XSS
            if (data.logs) {
                const lines = data.logs.split('\n');
                lines.forEach((line, index) => {
                    if (index > 0) {
                        logsContainer.appendChild(document.createElement('br'));
                    }
                    // Sanitize each line before displaying
                    const sanitizedLine = line.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                    logsContainer.appendChild(document.createTextNode(sanitizedLine));
                });
            } else {
                logsContainer.textContent = 'No logs available';
            }

            modalContent.appendChild(headerDiv);
            modalContent.appendChild(logsContainer);
            logsModal.appendChild(modalContent);
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

        // Fix XSS: Use safe DOM manipulation instead of innerHTML
        resultsDiv.textContent = ''; // Clear existing content safely

        // Create main container
        const mainContainer = document.createElement('div');
        mainContainer.className = 'space-y-6';

        // Create AI Analysis Summary section
        const analysisDiv = document.createElement('div');
        analysisDiv.className = 'bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/30 dark:to-indigo-900/30 p-6 rounded-lg border border-blue-200 dark:border-blue-800';

        // Header with brain icon
        const headerDiv = document.createElement('div');
        headerDiv.className = 'flex items-center space-x-3 mb-4';

        const brainIcon = document.createElement('i');
        brainIcon.className = 'fas fa-brain text-2xl text-blue-600 dark:text-blue-400';

        const headerTitle = document.createElement('h3');
        headerTitle.className = 'text-lg font-bold text-blue-800 dark:text-blue-200';
        headerTitle.textContent = 'Smart AI Analysis';

        headerDiv.appendChild(brainIcon);
        headerDiv.appendChild(headerTitle);

        // Stats grid
        const statsGrid = document.createElement('div');
        statsGrid.className = 'grid grid-cols-1 md:grid-cols-3 gap-4 text-sm';

        // Candidates Evaluated stat
        const candidatesDiv = document.createElement('div');
        candidatesDiv.className = 'bg-white dark:bg-gray-800 p-3 rounded-lg';

        const candidatesLabel = document.createElement('div');
        candidatesLabel.className = 'font-semibold text-gray-700 dark:text-gray-300';
        candidatesLabel.textContent = 'Candidates Evaluated';

        const candidatesValue = document.createElement('div');
        candidatesValue.className = 'text-2xl font-bold text-blue-600 dark:text-blue-400';
        candidatesValue.textContent = analysis.candidates_evaluated.toLocaleString();

        candidatesDiv.appendChild(candidatesLabel);
        candidatesDiv.appendChild(candidatesValue);

        // AI Methods stat
        const methodsDiv = document.createElement('div');
        methodsDiv.className = 'bg-white dark:bg-gray-800 p-3 rounded-lg';

        const methodsLabel = document.createElement('div');
        methodsLabel.className = 'font-semibold text-gray-700 dark:text-gray-300';
        methodsLabel.textContent = 'AI Methods Used';

        const methodsValue = document.createElement('div');
        methodsValue.className = 'text-sm text-blue-600 dark:text-blue-400';
        methodsValue.textContent = analysis.methods_used.join(', ').toUpperCase();

        methodsDiv.appendChild(methodsLabel);
        methodsDiv.appendChild(methodsValue);

        // Average Score stat
        const scoreDiv = document.createElement('div');
        scoreDiv.className = 'bg-white dark:bg-gray-800 p-3 rounded-lg';

        const scoreLabel = document.createElement('div');
        scoreLabel.className = 'font-semibold text-gray-700 dark:text-gray-300';
        scoreLabel.textContent = 'Average AI Score';

        const scoreValue = document.createElement('div');
        scoreValue.className = 'text-2xl font-bold text-green-600 dark:text-green-400';
        scoreValue.textContent = (analysis.score_range.average * 100).toFixed(1) + '%';

        scoreDiv.appendChild(scoreLabel);
        scoreDiv.appendChild(scoreValue);

        statsGrid.appendChild(candidatesDiv);
        statsGrid.appendChild(methodsDiv);
        statsGrid.appendChild(scoreDiv);

        // Recommendation box
        const recBox = document.createElement('div');
        recBox.className = 'mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800';

        const recText = document.createElement('p');
        recText.className = 'text-sm text-green-800 dark:text-green-200';

        const lightbulbIcon = document.createElement('i');
        lightbulbIcon.className = 'fas fa-lightbulb mr-2';

        const strongText = document.createElement('strong');
        strongText.textContent = 'AI Recommendation: ';

        recText.appendChild(lightbulbIcon);
        recText.appendChild(strongText);
        recText.appendChild(document.createTextNode(data.recommendation));

        recBox.appendChild(recText);

        analysisDiv.appendChild(headerDiv);
        analysisDiv.appendChild(statsGrid);
        analysisDiv.appendChild(recBox);

        // Create predictions table section
        const tableSection = document.createElement('div');
        tableSection.className = 'bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700';

        // Table header
        const tableHeader = document.createElement('div');
        tableHeader.className = 'px-6 py-4 border-b border-gray-200 dark:border-gray-700';

        const tableTitle = document.createElement('h4');
        tableTitle.className = 'text-lg font-semibold text-gray-900 dark:text-white';
        tableTitle.textContent = 'Smart AI Predictions (Ranked by AI Score)';

        const tableSubtitle = document.createElement('p');
        tableSubtitle.className = 'text-sm text-gray-600 dark:text-gray-400 mt-1';
        tableSubtitle.textContent = `Top ${predictions.length} predictions automatically selected and ranked by AI`;

        tableHeader.appendChild(tableTitle);
        tableHeader.appendChild(tableSubtitle);

        // Create table
        const tableContainer = document.createElement('div');
        tableContainer.className = 'overflow-x-auto';

        const table = document.createElement('table');
        table.className = 'min-w-full divide-y divide-gray-200 dark:divide-gray-700';

        // Table head
        const thead = document.createElement('thead');
        thead.className = 'bg-gray-50 dark:bg-gray-900';

        const headerRow = document.createElement('tr');
        const headers = ['Rank', 'Numbers', 'PB', 'AI Score', 'Tier', 'Method'];

        headers.forEach(headerText => {
            const th = document.createElement('th');
            th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider';
            th.textContent = headerText;
            headerRow.appendChild(th);
        });

        thead.appendChild(headerRow);

        // Table body
        const tbody = document.createElement('tbody');
        tbody.className = 'bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700';

        predictions.forEach((pred, index) => {
            const row = document.createElement('tr');
            row.className = 'hover:bg-gray-50 dark:hover:bg-gray-700';

            // Rank cell
            const rankCell = document.createElement('td');
            rankCell.className = 'px-6 py-4 whitespace-nowrap';

            const rankContainer = document.createElement('div');
            rankContainer.className = 'flex items-center';

            const rankSpan = document.createElement('span');
            rankSpan.className = `inline-flex items-center justify-center w-8 h-8 rounded-full ${index < 10 ? 'bg-yellow-100 text-yellow-800 border-2 border-yellow-300' : 'bg-gray-100 text-gray-800'} text-sm font-bold`;
            rankSpan.textContent = pred.rank;

            rankContainer.appendChild(rankSpan);
            rankCell.appendChild(rankContainer);

            // Numbers cell
            const numbersCell = document.createElement('td');
            numbersCell.className = 'px-6 py-4 whitespace-nowrap';

            const numbersContainer = document.createElement('div');
            numbersContainer.className = 'flex space-x-1';

            pred.numbers.forEach(num => {
                const numSpan = document.createElement('span');
                numSpan.className = 'powerball-number small bg-blue-600 text-white';
                numSpan.textContent = num;
                numbersContainer.appendChild(numSpan);
            });

            numbersCell.appendChild(numbersContainer);

            // Powerball cell
            const pbCell = document.createElement('td');
            pbCell.className = 'px-6 py-4 whitespace-nowrap';

            const pbSpan = document.createElement('span');
            pbSpan.className = 'powerball-number small bg-red-600 text-white';
            pbSpan.textContent = pred.powerball;

            pbCell.appendChild(pbSpan);

            // Score cell
            const scoreCell = document.createElement('td');
            scoreCell.className = 'px-6 py-4 whitespace-nowrap';

            const scoreMain = document.createElement('div');
            scoreMain.className = 'text-sm font-medium text-gray-900 dark:text-white';
            scoreMain.textContent = (pred.smart_ai_score * 100).toFixed(1) + '%';

            const scoreBase = document.createElement('div');
            scoreBase.className = 'text-xs text-gray-500 dark:text-gray-400';
            scoreBase.textContent = 'Base: ' + (pred.base_score * 100).toFixed(1) + '%';

            scoreCell.appendChild(scoreMain);
            scoreCell.appendChild(scoreBase);

            // Tier cell
            const tierCell = document.createElement('td');
            tierCell.className = 'px-6 py-4 whitespace-nowrap';

            const tierSpan = document.createElement('span');
            tierSpan.className = `inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getTierColor(pred.tier)}`;
            tierSpan.textContent = pred.tier;

            tierCell.appendChild(tierSpan);

            // Method cell
            const methodCell = document.createElement('td');
            methodCell.className = 'px-6 py-4 whitespace-nowrap';
            methodCell.innerHTML = getMethodBadge(pred.ai_method); // This is safe as getMethodBadge returns predefined HTML

            row.appendChild(rankCell);
            row.appendChild(numbersCell);
            row.appendChild(pbCell);
            row.appendChild(scoreCell);
            row.appendChild(tierCell);
            row.appendChild(methodCell);

            tbody.appendChild(row);
        });

        table.appendChild(thead);
        table.appendChild(tbody);
        tableContainer.appendChild(table);

        tableSection.appendChild(tableHeader);
        tableSection.appendChild(tableContainer);

        mainContainer.appendChild(analysisDiv);
        mainContainer.appendChild(tableSection);

        resultsDiv.appendChild(mainContainer);
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

        // Fix XSS: Clear tbody safely and use DOM methods
        tbody.textContent = ''; // Clear existing content safely

        data.syndicate_predictions.forEach((prediction, index) => {
            const numbers = prediction.numbers || [];
            const powerball = prediction.powerball || 0;
            const score = prediction.score || 0;
            const tier = prediction.tier || 'Standard';
            const method = prediction.method || data.method;
            const rank = prediction.rank || (index + 1);

            const row = document.createElement('tr');
            row.className = 'hover:bg-gray-50 dark:hover:bg-gray-700';

            // Rank cell
            const rankCell = document.createElement('td');
            rankCell.className = 'px-4 py-3 text-sm font-medium text-gray-900 dark:text-gray-100';
            rankCell.textContent = rank;

            // Numbers cell
            const numbersCell = document.createElement('td');
            numbersCell.className = 'px-4 py-3';

            const numbersContainer = document.createElement('div');
            numbersContainer.className = 'flex items-center';

            // Create white balls safely
            numbers.forEach(num => {
                const ball = document.createElement('span');
                ball.className = 'inline-flex items-center justify-center w-6 h-6 bg-gray-200 dark:bg-gray-600 rounded-full text-xs font-semibold text-gray-800 dark:text-white mr-1';
                ball.textContent = num;
                numbersContainer.appendChild(ball);
            });

            // Add separator
            const separator = document.createElement('span');
            separator.className = 'mx-2 text-red-500';
            separator.textContent = '•';
            numbersContainer.appendChild(separator);

            // Create powerball safely
            const powerBall = document.createElement('span');
            powerBall.className = 'inline-flex items-center justify-center w-6 h-6 bg-red-500 rounded-full text-xs font-semibold text-white';
            powerBall.textContent = powerball;
            numbersContainer.appendChild(powerBall);

            numbersCell.appendChild(numbersContainer);

            // Score cell
            const scoreCell = document.createElement('td');
            scoreCell.className = 'px-4 py-3 text-sm text-gray-900 dark:text-gray-100';
            scoreCell.textContent = (score * 100).toFixed(2) + '%';

            // Tier cell
            const tierCell = document.createElement('td');
            tierCell.className = 'px-4 py-3';

            const tierSpan = document.createElement('span');
            const tierColors = {
                'Premium': 'text-purple-600 bg-purple-100 dark:bg-purple-900 dark:text-purple-300',
                'High': 'text-blue-600 bg-blue-100 dark:bg-blue-900 dark:text-blue-300',
                'Medium': 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-300',
                'Standard': 'text-gray-600 bg-gray-100 dark:bg-gray-900 dark:text-gray-100 dark:text-gray-300'
            };

            tierSpan.className = `px-2 py-1 text-xs font-medium rounded-full ${tierColors[tier] || tierColors['Standard']}`;
            tierSpan.textContent = tier;
            tierCell.appendChild(tierSpan);

            // Method cell
            const methodCell = document.createElement('td');
            methodCell.className = 'px-4 py-3 text-sm text-gray-500 dark:text-gray-400';
            methodCell.textContent = method;

            row.appendChild(rankCell);
            row.appendChild(numbersCell);
            row.appendChild(scoreCell);
            row.appendChild(tierCell);
            row.appendChild(methodCell);

            tbody.appendChild(row);
        });
    }

    function exportSyndicateData() {
        if (!syndicateData || !syndicateData.syndicate_predictions) {
            showToast('No syndicate data to export', 'warning');
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

                showToast('Syndicate data exported successfully!', 'success');
            }
        } catch (error) {
            console.error('Export error:', error);
            showToast('Failed to export syndicate data', 'error');
        }
    }

    function updateElementText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
        }
    }

    /**
     * Sanitize text content to prevent XSS
     * @param {string} text - Text to sanitize
     * @returns {string} - Sanitized text
     */
    function sanitizeText(text) {
        if (typeof text !== 'string') {
            return String(text || '');
        }

        // Create a temporary element to use browser's built-in HTML escaping
        const tempElement = document.createElement('div');
        tempElement.textContent = text;
        return tempElement.innerHTML;
    }

    /**
     * Sanitize DOM element to prevent XSS attacks
     * @param {HTMLElement} element - Element to sanitize
     */
    function sanitizeElement(element) {
        // Remove potentially dangerous attributes
        const dangerousAttrs = ['onclick', 'onload', 'onerror', 'onmouseover', 'onfocus', 'onblur', 'onchange', 'onsubmit'];
        dangerousAttrs.forEach(attr => {
            if (element.hasAttribute(attr)) {
                element.removeAttribute(attr);
            }
        });

        // Recursively sanitize child elements
        for (let child of element.children) {
            this.sanitizeElement(child);
        }

        // Sanitize text content (remove script tags and dangerous content)
        if (element.textContent) {
            element.textContent = element.textContent.replace(/<script[^>]*>.*?<\/script>/gi, '');
        }
    }

    /**
     * Show notification message
     * @param {string} message - Message to display
     * @param {string} type - Type of notification (success, error, warning, info)
     */
    function showNotification(message, type = 'info') {
        const toast = document.getElementById('toast-notification');
        if (!toast) return;

        const typeClasses = {
            success: 'bg-green-500',
            error: 'bg-red-500',
            warning: 'bg-yellow-500',
            info: 'bg-blue-500'
        };

        toast.className = `fixed bottom-5 right-5 text-white py-2 px-4 rounded-lg shadow-xl opacity-100 transition-opacity duration-300 z-50 ${typeClasses[type] || typeClasses.info}`;
        // Create content safely without innerHTML
        toast.textContent = ''; // Clear existing content
        
        const icon = document.createElement('i');
        icon.className = 'fas fa-info mr-2';
        
        const messageSpan = document.createElement('span');
        messageSpan.textContent = message; // Safe text content, prevents XSS
        
        toast.appendChild(icon);
        toast.appendChild(messageSpan);

        setTimeout(() => {
            toast.classList.remove('opacity-100');
            toast.classList.add('opacity-0');
        }, 3000);
    }


    // Initialize UI state
    initializePipelineDashboard();

});