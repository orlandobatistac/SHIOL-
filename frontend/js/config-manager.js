
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
            const config = this.collectConfiguration();
            const response = await fetch(`${this.API_BASE_URL}/config/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                this.showNotification('Configuration saved successfully', 'success');
                this.currentConfig = config;
            } else {
                throw new Error('Failed to save configuration');
            }
        } catch (error) {
            console.error('Error saving configuration:', error);
            this.showNotification('Error saving configuration', 'error');
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
        }, 5000);
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
            const response = await fetch(`${this.API_BASE_URL}/database/stats`);
            if (response.ok) {
                const stats = await response.json();
                
                document.getElementById('db-records').textContent = stats.total_records || '-';
                document.getElementById('db-size').textContent = stats.size_mb ? `${stats.size_mb}MB` : '-';
                document.getElementById('db-predictions').textContent = stats.predictions_count || '-';
                
                this.showNotification('Database stats refreshed', 'success');
            }
        } catch (error) {
            console.error('Error refreshing database stats:', error);
            this.showNotification('Error refreshing database stats', 'error');
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
        this.showNotification('Pipeline execution triggered', 'info');
    }

    async testPipeline() {
        this.showNotification('Pipeline test initiated', 'info');
    }

    async backupDatabase() {
        this.showNotification('Database backup initiated', 'info');
    }

    async retrainModel() {
        this.showNotification('Model retraining started', 'info');
    }

    showCleanupModal() {
        const modal = document.getElementById('cleanup-modal');
        if (modal) {
            modal.style.display = 'block';
        }
    }

    async refreshLogs() {
        this.showNotification('Logs refreshed', 'info');
    }

    downloadLogs() {
        this.showNotification('Logs download started', 'info');
    }

    clearLogs() {
        if (confirm('Are you sure you want to clear all logs?')) {
            this.showNotification('Logs cleared', 'warning');
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.configManager = new ConfigurationManager();
});
