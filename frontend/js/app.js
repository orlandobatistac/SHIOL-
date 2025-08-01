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
            // Start auto-refresh every 30 seconds
            autoRefreshInterval = setInterval(fetchPipelineStatus, 30000);
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

    // Initialize UI state
    initializePipelineDashboard();
});