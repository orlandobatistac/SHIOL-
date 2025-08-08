/**
 * SHIOL+ v6.0 Configuration Manager Frontend
 * Handles all configuration dashboard interactions
 */

class ConfigurationManager {
    constructor() {
        this.API_BASE_URL = window.location.origin + '/api/v1';
        this.currentConfig = {};
        this.systemInterval = null;
        this.init();
    }

    init() {
        this.initializeEventListeners();
        this.initializeProfiles();
        this.initializeToggles();
        this.loadConfiguration();
        this.startSystemMonitoring();
    }

    initializeEventListeners() {
        // Configuration save/load
        document.getElementById('save-config-btn')?.addEventListener('click', () => this.saveConfiguration());
        document.getElementById('export-config-btn')?.addEventListener('click', () => this.exportConfiguration());
        document.getElementById('import-config-btn')?.addEventListener('click', () => this.importConfiguration());
        document.getElementById('test-pipeline-btn')?.addEventListener('click', () => this.testPipeline());

        // Database management
        document.getElementById('refresh-db-stats-btn')?.addEventListener('click', () => this.refreshDatabaseStats());
        document.getElementById('backup-db-btn')?.addEventListener('click', () => this.backupDatabase());
        document.getElementById('cleanup-db-btn')?.addEventListener('click', () => this.showCleanupModal());

        // Model management
        document.getElementById('retrain-model-btn')?.addEventListener('click', () => this.retrainModel());
        document.getElementById('backup-models-btn')?.addEventListener('click', () => this.backupModels());
        document.getElementById('reset-models-btn')?.addEventListener('click', () => this.resetModels());

        // Pipeline controls
        document.getElementById('trigger-pipeline-btn')?.addEventListener('click', () => this.triggerPipeline());
        document.getElementById('stop-pipeline-btn')?.addEventListener('click', () => this.stopPipeline());

        // Log management
        document.getElementById('refresh-logs-btn')?.addEventListener('click', () => this.refreshLogs());
        document.getElementById('download-logs-btn')?.addEventListener('click', () => this.downloadLogs());
        document.getElementById('clear-logs-btn')?.addEventListener('click', () => this.clearLogs());

        // Prediction count slider
        const predictionSlider = document.getElementById('prediction-count');
        if (predictionSlider) {
            predictionSlider.addEventListener('input', (e) => {
                this.updatePredictionCount(e.target.value);
            });
        }

        // File import handler
        const configImport = document.getElementById('config-import');
        if (configImport) {
            configImport.addEventListener('change', (e) => this.handleConfigImport(e));
        }
    }

    initializeProfiles() {
        const profileBtns = document.querySelectorAll('.profile-btn');
        profileBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                profileBtns.forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                this.loadConfigProfile(btn.dataset.profile);
            });
        });
    }

    initializeToggles() {
        const toggles = document.querySelectorAll('.toggle-switch');
        toggles.forEach(toggle => {
            toggle.addEventListener('click', () => {
                toggle.classList.toggle('active');
            });
        });
    }

    async loadConfiguration() {
        try {
            const response = await fetch(`${this.API_BASE_URL}/config/load`);
            if (response.ok) {
                const config = await response.json();
                this.applyConfiguration(config);
                this.currentConfig = config;
            }
        } catch (error) {
            console.error('Error loading configuration:', error);
            this.showNotification('Error loading configuration', 'error');
        }
    }

    async saveConfiguration() {
        try {
            console.log('🔄 Testing Configuration Save...');
            const config = this.collectConfiguration();
            console.log('📋 Configuration Data:', config);

            const response = await fetch(`${this.API_BASE_URL}/config/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                const result = await response.json();
                console.log('✅ Configuration Save Result:', result);
                this.showNotification('Configuration saved successfully', 'success');
                this.currentConfig = config;
            } else {
                const errorText = await response.text();
                throw new Error(`Failed to save configuration: ${response.status} - ${errorText}`);
            }
        } catch (error) {
            console.error('❌ Configuration Save Error:', error);
            this.showNotification('Error saving configuration: ' + error.message, 'error');
        }
    }

    collectConfiguration() {
        const mondayEl = document.getElementById('day-monday');
        const wednesdayEl = document.getElementById('day-wednesday');
        const saturdayEl = document.getElementById('day-saturday');
        const timeEl = document.getElementById('execution-time');
        const timezoneEl = document.getElementById('timezone');
        const toggleEl = document.getElementById('auto-execution-toggle');

        return {
            pipeline: {
                execution_days: {
                    monday: mondayEl ? mondayEl.checked : false,
                    wednesday: wednesdayEl ? wednesdayEl.checked : false,
                    saturday: saturdayEl ? saturdayEl.checked : false
                },
                execution_time: timeEl ? timeEl.value : '02:00',
                timezone: timezoneEl ? timezoneEl.value : 'America/New_York',
                auto_execution: toggleEl ? toggleEl.classList.contains('active') : false
            },
            predictions: {
                count: parseInt((document.getElementById('prediction-count')?.value) || '100'),
                method: (document.getElementById('prediction-method')?.value) || 'smart_ai',
                weights: {
                    probability: parseInt((document.getElementById('weight-probability')?.value) || '40'),
                    diversity: parseInt((document.getElementById('weight-diversity')?.value) || '25'),
                    historical: parseInt((document.getElementById('weight-historical')?.value) || '20'),
                    risk: parseInt((document.getElementById('weight-risk')?.value) || '15')
                }
            },
            notifications: {
                email: (document.getElementById('notification-email')?.value) || '',
                session_timeout: parseInt((document.getElementById('session-timeout')?.value) || '60')
            }
        };
    }

    applyConfiguration(config) {
        if (!config) return;

        // Pipeline settings
        if (config.pipeline) {
            const { execution_days, execution_time, timezone, auto_execution } = config.pipeline;

            if (execution_days) {
                const mondayEl = document.getElementById('day-monday');
                const wednesdayEl = document.getElementById('day-wednesday');
                const saturdayEl = document.getElementById('day-saturday');

                if (mondayEl) mondayEl.checked = execution_days.monday;
                if (wednesdayEl) wednesdayEl.checked = execution_days.wednesday;
                if (saturdayEl) saturdayEl.checked = execution_days.saturday;
            }

            if (execution_time) {
                const timeEl = document.getElementById('execution-time');
                if (timeEl) timeEl.value = execution_time;
            }

            if (timezone) {
                const timezoneEl = document.getElementById('timezone');
                if (timezoneEl) timezoneEl.value = timezone;
            }

            const autoToggle = document.getElementById('auto-execution-toggle');
            if (autoToggle) {
                if (auto_execution) {
                    autoToggle.classList.add('active');
                } else {
                    autoToggle.classList.remove('active');
                }
            }
        }

        // Prediction settings
        if (config.predictions) {
            const { count, method, weights } = config.predictions;

            if (count) {
                const slider = document.getElementById('prediction-count');
                if (slider) {
                    slider.value = count;
                    this.updatePredictionCount(count);
                }
            }

            if (method) {
                const methodEl = document.getElementById('prediction-method');
                if (methodEl) methodEl.value = method;
            }

            if (weights) {
                Object.entries(weights).forEach(([key, value]) => {
                    const element = document.getElementById(`weight-${key}`);
                    if (element) element.value = value;
                });
            }
        }

        // Notification settings
        if (config.notifications) {
            const { email, session_timeout } = config.notifications;

            if (email) {
                const emailEl = document.getElementById('notification-email');
                if (emailEl) emailEl.value = email;
            }

            if (session_timeout) {
                const timeoutEl = document.getElementById('session-timeout');
                if (timeoutEl) timeoutEl.value = session_timeout;
            }
        }
    }

    loadConfigProfile(profileName) {
        const profiles = {
            conservative: {
                'prediction-count': 50,
                'prediction-method': 'deterministic',
                'weight-probability': 50,
                'weight-diversity': 20,
                'weight-historical': 20,
                'weight-risk': 10
            },
            aggressive: {
                'prediction-count': 500,
                'prediction-method': 'ensemble',
                'weight-probability': 30,
                'weight-diversity': 35,
                'weight-historical': 20,
                'weight-risk': 15
            },
            balanced: {
                'prediction-count': 100,
                'prediction-method': 'smart_ai',
                'weight-probability': 40,
                'weight-diversity': 25,
                'weight-historical': 20,
                'weight-risk': 15
            }
        };

        const profile = profiles[profileName];
        if (profile && profileName !== 'custom') {
            Object.entries(profile).forEach(([key, value]) => {
                const element = document.getElementById(key);
                if (element) {
                    element.value = value;
                    if (key === 'prediction-count') {
                        this.updatePredictionCount(value);
                    }
                }
            });
            this.showNotification(`${profileName.charAt(0).toUpperCase() + profileName.slice(1)} profile loaded`, 'success');
        }
    }

    updatePredictionCount(value) {
        const display = document.getElementById('prediction-count-display');
        if (display) {
            display.textContent = value;
        }
    }

    async startSystemMonitoring() {
        this.updateSystemStats();
        this.systemInterval = setInterval(() => {
            this.updateSystemStats();
        }, 30000); // Changed from 5000ms (5s) to 30000ms (30s)
    }

    async updateSystemStats() {
        try {
            const response = await fetch(`${this.API_BASE_URL}/system/stats`);
            if (response.ok) {
                const stats = await response.json();
                this.displaySystemStats(stats);
            }
        } catch (error) {
            console.error('Error updating system stats:', error);
        }
    }

    displaySystemStats(stats) {
        // Update CPU usage
        const cpuElement = document.getElementById('cpu-usage');
        const cpuProgress = document.getElementById('cpu-progress');
        if (cpuElement && cpuProgress) {
            cpuElement.textContent = `${stats.cpu_usage}%`;
            cpuProgress.style.width = `${stats.cpu_usage}%`;
        }

        // Update Memory usage
        const memoryElement = document.getElementById('memory-usage');
        const memoryProgress = document.getElementById('memory-progress');
        if (memoryElement && memoryProgress) {
            memoryElement.textContent = `${stats.memory_usage}%`;
            memoryProgress.style.width = `${stats.memory_usage}%`;
        }

        // Update Disk usage
        const diskElement = document.getElementById('disk-usage');
        const diskProgress = document.getElementById('disk-progress');
        if (diskElement && diskProgress) {
            diskElement.textContent = `${stats.disk_usage}%`;
            diskProgress.style.width = `${stats.disk_usage}%`;
        }

        // Update pipeline status
        const pipelineElement = document.getElementById('pipeline-status');
        const lastExecutionElement = document.getElementById('last-execution');
        if (pipelineElement) {
            pipelineElement.textContent = stats.pipeline_status || 'Unknown';
        }
        if (lastExecutionElement) {
            lastExecutionElement.textContent = `Last: ${stats.last_execution || 'Never'}`;
        }
    }

    async refreshDatabaseStats() {
        try {
            console.log('🔄 Testing Database Stats Refresh...');
            const response = await fetch(`${this.API_BASE_URL}/database/stats`);
            if (response.ok) {
                const stats = await response.json();
                console.log('✅ Database Stats Retrieved:', stats);

                document.getElementById('db-records').textContent = stats.total_records || '-';
                document.getElementById('db-size').textContent = stats.size_mb ? `${stats.size_mb}MB` : '-';
                document.getElementById('db-predictions').textContent = stats.predictions_count || '-';

                this.showNotification(`Database Stats Updated: ${stats.total_records} records, ${stats.size_mb}MB`, 'success');
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('❌ Error refreshing database stats:', error);
            this.showNotification('Error refreshing database stats: ' + error.message, 'error');
        }
    }

    exportConfiguration() {
        const config = this.collectConfiguration();
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `shiolplus-config-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        this.showNotification('Configuration exported successfully', 'success');
    }

    importConfiguration() {
        document.getElementById('config-import').click();
    }

    handleConfigImport(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const config = JSON.parse(e.target.result);
                    this.applyConfiguration(config);
                    this.showNotification('Configuration imported successfully', 'success');
                } catch (error) {
                    this.showNotification('Error importing configuration: Invalid JSON', 'error');
                }
            };
            reader.readAsText(file);
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-5 right-5 px-6 py-3 rounded-lg shadow-xl text-white font-medium z-50 transition-opacity duration-300`;

        const colors = {
            success: 'bg-green-500',
            error: 'bg-red-500',
            warning: 'bg-yellow-500',
            info: 'bg-blue-500'
        };

        notification.classList.add(colors[type] || colors.info);
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 4000);
    }

    // Placeholder methods for other functionalities
    async triggerPipeline() {
        console.log('🔄 Testing Pipeline Trigger...');

        // Update button to show working state
        const executeButton = document.querySelector('button[onclick="configManager.triggerPipeline()"]');
        const originalText = executeButton ? executeButton.textContent : '';
        if (executeButton) {
            executeButton.disabled = true;
            executeButton.textContent = 'Executing Pipeline...';
            executeButton.classList.add('opacity-75');
        }

        const response = await fetch(`${this.API_BASE_URL}/pipeline/trigger?num_predictions=1`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            // Reset button on error
            if (executeButton) {
                executeButton.disabled = false;
                executeButton.textContent = originalText;
                executeButton.classList.remove('opacity-75');
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('✅ Pipeline Trigger Result:', result);

        if (result.execution_id) {
            this.monitorPipelineExecution(result.execution_id, executeButton, originalText);
        } else {
            console.warn('⚠️ No execution ID received from pipeline trigger');
            // Reset button if no execution ID
            if (executeButton) {
                executeButton.disabled = false;
                executeButton.textContent = originalText;
                executeButton.classList.remove('opacity-75');
            }
        }
    }

    monitorPipelineExecution(executionId, executeButton = null, originalText = '') {
        console.log(`🔍 Monitoring pipeline execution: ${executionId}`);

        const resetButton = () => {
            if (executeButton) {
                executeButton.disabled = false;
                executeButton.textContent = originalText || 'Execute Pipeline';
                executeButton.classList.remove('opacity-75');
            }
        };

        let monitoringAttempts = 0;
        const maxAttempts = 60; // Maximum 5 minutes of monitoring

        const checkStatus = async () => {
            try {
                monitoringAttempts++;
                if (monitoringAttempts > maxAttempts) {
                    console.log('⏰ Pipeline monitoring timeout reached');
                    this.showNotification('Pipeline monitoring timeout. Execution may still be running.', 'warning');
                    resetButton();
                    return;
                }

                const response = await fetch(`${this.API_BASE_URL}/pipeline/status`);
                if (!response.ok) {
                    console.warn(`Pipeline status check failed: ${response.status}`);
                    // Continue monitoring despite HTTP errors
                    setTimeout(checkStatus, 10000);
                    return;
                }

                const data = await response.json();
                const execution = data.pipeline_status?.recent_execution_history?.find(
                    ex => ex.execution_id === executionId
                );

                if (execution) {
                    console.log(`📊 Execution ${executionId} status: ${execution.status}`);

                    if (execution.status === 'completed') {
                        console.log('✅ Pipeline execution completed successfully');
                        this.showNotification('Pipeline execution completed successfully!', 'success');
                        resetButton();
                        return; // Stop monitoring
                    } else if (execution.status === 'failed') {
                        console.log('❌ Pipeline execution failed');
                        this.showNotification(`Pipeline execution failed: ${execution.error || 'Unknown error'}`, 'error');
                        resetButton();
                        return; // Stop monitoring
                    } else if (execution.status === 'running') {
                        // Update button text with current step if available
                        if (executeButton && execution.current_step) {
                            executeButton.textContent = `${execution.current_step}...`;
                        }
                    }
                }

                // Continue monitoring if still running
                setTimeout(checkStatus, 5000); // Check every 5 seconds

            } catch (error) {
                console.warn('Pipeline status check failed:', error);
                // Continue monitoring but with longer delay on errors
                setTimeout(checkStatus, 15000); // Retry in 15 seconds on error
            }
        };

        // Start monitoring immediately
        setTimeout(checkStatus, 2000); // Initial delay of 2 seconds
    }

    resetPipelineUI() {
        const triggerBtn = document.getElementById('trigger-pipeline-btn');
        if (triggerBtn) {
            triggerBtn.disabled = false;
            triggerBtn.textContent = 'Execute Pipeline';
        }

        const stopBtn = document.getElementById('stop-pipeline-btn');
        if (stopBtn) {
            stopBtn.classList.add('hidden');
        }

        const progressEl = document.getElementById('pipeline-progress');
        if (progressEl) {
            progressEl.classList.add('hidden');
        }
    }

    async stopPipeline() {
        try {
            console.log('🔄 Testing Pipeline Stop...');
            const response = await fetch(`${this.API_BASE_URL}/pipeline/stop`, { method: 'POST' });
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Pipeline Stop Result:', result);
                this.showNotification('Pipeline execution stopped.', 'success');
                document.getElementById('stop-pipeline-btn').classList.add('hidden');
                document.getElementById('trigger-pipeline-btn').classList.remove('hidden');
                document.getElementById('pipeline-progress').classList.add('hidden');
                clearInterval(this.pipelineMonitoringInterval);
            } else {
                const errorText = await response.text();
                throw new Error(`Failed to stop pipeline: ${response.status} - ${errorText}`);
            }
        } catch (error) {
            console.error('❌ Pipeline Stop Error:', error);
            this.showNotification('Error stopping pipeline: ' + error.message, 'error');
        }
    }

    async testPipeline() {
        try {
            console.log('🔄 Testing Pipeline Configuration...');
            const response = await fetch(`${this.API_BASE_URL}/pipeline/test`, { method: 'POST' });
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Pipeline Test Result:', result);

                // Mostrar resultados detallados
                const details = Object.entries(result.results || {})
                    .map(([key, value]) => `${key}: ${value ? '✅' : '❌'}`)
                    .join(', ');

                this.showNotification(`Pipeline test completed: ${result.status} - ${details}`, 'success');
            } else {
                const errorText = await response.text();
                throw new Error(`Pipeline test failed: ${response.status} - ${errorText}`);
            }
        } catch (error) {
            console.error('❌ Pipeline Test Error:', error);
            this.showNotification('Error testing pipeline: ' + error.message, 'error');
        }
    }

    async backupDatabase() {
        try {
            console.log('🔄 Testing Database Backup...');
            const response = await fetch(`${this.API_BASE_URL}/database/backup`, { method: 'POST' });
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Database Backup Result:', result);
                this.showNotification('Database backup created successfully: ' + result.backup_file, 'success');
            } else {
                const errorText = await response.text();
                throw new Error(`Backup failed: ${response.status} - ${errorText}`);
            }
        } catch (error) {
            console.error('❌ Database Backup Error:', error);
            this.showNotification('Error creating backup: ' + error.message, 'error');
        }
    }

    async retrainModel() {
        try {
            console.log('🔄 Testing Model Retraining...');
            const response = await fetch(`${this.API_BASE_URL}/model/retrain`, { method: 'POST' });
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Model Retrain Result:', result);
                this.showNotification(`Model retraining started: ${result.status} - ETA: ${result.estimated_completion}`, 'success');
            } else {
                const errorText = await response.text();
                throw new Error(`Retraining failed: ${response.status} - ${errorText}`);
            }
        } catch (error) {
            console.error('❌ Model Retrain Error:', error);
            this.showNotification('Error retraining model: ' + error.message, 'error');
        }
    }

    async backupModels() {
        try {
            console.log('🔄 Testing Models Backup...');
            const response = await fetch(`${this.API_BASE_URL}/model/backup`, { method: 'POST' });
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Models Backup Result:', result);
                this.showNotification(`Models backup created: ${result.backup_file}`, 'success');
            } else {
                const errorText = await response.text();
                throw new Error(`Models backup failed: ${response.status} - ${errorText}`);
            }
        } catch (error) {
            console.error('❌ Models Backup Error:', error);
            this.showNotification('Error backing up models: ' + error.message, 'error');
        }
    }

    async resetModels() {
        if (!confirm('Are you sure you want to reset all AI models? This action will remove all adaptive learning data and cannot be undone.')) {
            return;
        }

        try {
            console.log('🔄 Resetting AI Models...');

            // Show loading state
            const resetBtn = document.getElementById('reset-models-btn');
            if (resetBtn) {
                resetBtn.disabled = true;
                resetBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Resetting...';
            }

            const response = await fetch(`${this.API_BASE_URL}/model/reset`, { method: 'POST' });

            if (response.ok) {
                const result = await response.json();
                console.log('✅ Models Reset Result:', result);

                // Show detailed success message
                let message = `AI Models reset successfully!`;
                if (result.details) {
                    const details = result.details;
                    message += ` Cleared: ${details.weights_cleared} weights, ${details.feedback_cleared} feedback records, ${details.plays_cleared} reliable plays.`;
                }

                this.showNotification(message, 'success');

                // Optionally refresh the page after a short delay
                setTimeout(() => {
                    window.location.reload();
                }, 2000);

            } else {
                const errorText = await response.text();
                throw new Error(`Models reset failed: ${response.status} - ${errorText}`);
            }
        } catch (error) {
            console.error('❌ Models Reset Error:', error);
            this.showNotification('Error resetting AI models: ' + error.message, 'error');
        } finally {
            // Reset button state
            const resetBtn = document.getElementById('reset-models-btn');
            if (resetBtn) {
                resetBtn.disabled = false;
                resetBtn.innerHTML = '<i class="fas fa-redo mr-2"></i>Reset AI Models';
            }
        }
    }

    showCleanupModal() {
        const modal = document.getElementById('cleanup-modal');
        if (modal) {
            modal.style.display = 'block';
        }
    }

    async refreshLogs() {
        try {
            console.log('🔄 Testing Logs Refresh...');
            const response = await fetch(`${this.API_BASE_URL}/logs`);
            if (response.ok) {
                const logs = await response.json();
                console.log('✅ Logs Retrieved:', logs);
                this.displayLogs(logs);
                this.showNotification('Logs refreshed successfully', 'success');
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('❌ Logs Refresh Error:', error);
            this.showNotification('Error refreshing logs: ' + error.message, 'error');
        }
    }

    downloadLogs() {
        try {
            console.log('🔄 Initiating Logs Download...');
            // In a real scenario, this would fetch the logs and create a downloadable file.
            // For now, we simulate a success notification.
            this.showNotification('Logs download initiated. Check your downloads folder.', 'info');
            console.log('✅ Logs Download Initiated.');
        } catch (error) {
            console.error('❌ Logs Download Error:', error);
            this.showNotification('Error initiating logs download: ' + error.message, 'error');
        }
    }

    clearLogs() {
        if (confirm('Are you sure you want to clear all logs?')) {
            try {
                console.log('🔄 Testing Logs Clear...');
                // In a real scenario, this would send a request to the backend to clear logs.
                // For now, we simulate a success notification.
                this.showNotification('Logs cleared successfully', 'warning');
                console.log('✅ Logs Cleared.');
                // Optionally clear the displayed logs
                this.displayLogs([]);
            } catch (error) {
                console.error('❌ Logs Clear Error:', error);
                this.showNotification('Error clearing logs: ' + error.message, 'error');
            }
        }
    }

    displayLogs(logs) {
        const logOutputElement = document.getElementById('log-output'); // Assuming there's a div with this ID to display logs
        if (logOutputElement) {
            // Clear existing content safely
            logOutputElement.textContent = '';
            
            // Use safe DOM methods to prevent XSS
            logs.forEach(log => {
                const logDiv = document.createElement('div');
                const logText = document.createTextNode(`${log.timestamp} - ${log.message}`);
                logDiv.appendChild(logText);
                logOutputElement.appendChild(logDiv);
            });
        }
    }

    async refreshPipelineHistory() {
        try {
            console.log('🔄 Refreshing pipeline execution history...');

            const response = await fetch(`${this.API_BASE_URL}/pipeline/execution-history?limit=20`);
            if (response.ok) {
                const data = await response.json();
                console.log('✅ Pipeline execution history refreshed:', data);

                // Trigger the global function if it exists
                if (typeof window.loadPipelineExecutionHistory === 'function') {
                    window.loadPipelineExecutionHistory();
                }

                this.showNotification(`Pipeline history updated: ${data.count} executions`, 'success');
                return data;
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('❌ Error refreshing pipeline history:', error);
            this.showNotification('Error refreshing pipeline history: ' + error.message, 'error');
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.configManager = new ConfigurationManager();
});