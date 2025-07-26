# SHIOL+ Phase 5 Linux Deployment Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration Setup](#configuration-setup)
4. [Execution Instructions](#execution-instructions)
5. [Service Setup (Optional)](#service-setup-optional)
6. [Verification Steps](#verification-steps)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting Section](#troubleshooting-section)

---

## System Requirements

### Minimum Linux Distribution Requirements
- **Ubuntu**: 20.04 LTS or later
- **CentOS/RHEL**: 8.0 or later
- **Debian**: 10 (Buster) or later
- **Fedora**: 32 or later
- **openSUSE**: Leap 15.2 or later

### Python Version Requirements
- **Python**: 3.8 or later (3.10+ recommended)
- **pip**: Latest version (automatically updated during installation)

### System Resources
- **RAM**: Minimum 4GB, Recommended 8GB+
- **CPU**: 2+ cores recommended for ML model training
- **Disk Space**: 
  - Minimum: 2GB free space
  - Recommended: 10GB+ for logs, data, and model storage
- **Swap**: 2GB recommended for memory-intensive operations

### Network Requirements
- **Internet Access**: Required for initial data download and updates
- **Ports**: 
  - Port 8000 (default API server)
  - Port 80/443 (if using reverse proxy)
- **Bandwidth**: Minimal ongoing requirements (~10MB/month for data updates)

---

## Installation Steps

### Step 1: System Preparation

#### Ubuntu/Debian Systems
```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3 python3-pip python3-venv git curl wget build-essential

# Install additional dependencies for ML libraries
sudo apt install -y python3-dev libffi-dev libssl-dev
```

#### CentOS/RHEL/Fedora Systems
```bash
# Update system packages
sudo dnf update -y  # For Fedora/RHEL 8+
# OR for CentOS 7: sudo yum update -y

# Install required packages
sudo dnf install -y python3 python3-pip python3-devel git curl wget gcc gcc-c++ make

# Install additional development tools
sudo dnf groupinstall -y "Development Tools"
```

### Step 2: Create System User (Recommended)
```bash
# Create dedicated user for SHIOL+
sudo useradd -m -s /bin/bash shiolplus
sudo usermod -aG sudo shiolplus  # Optional: add to sudo group

# Switch to the new user
sudo su - shiolplus
```

### Step 3: Download and Setup SHIOL+
```bash
# Create application directory
mkdir -p ~/shiolplus
cd ~/shiolplus

# Clone or extract SHIOL+ files (replace with your method)
# If using git:
# git clone <repository-url> .
# If using archive, extract files here

# Verify main files are present
ls -la
# Should see: main.py, requirements.txt, config/, src/, etc.
```

### Step 4: Python Virtual Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip to latest version
pip install --upgrade pip

# Install wheel for better package compilation
pip install wheel
```

### Step 5: Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list
```

### Step 6: Directory Structure Setup
```bash
# Create required directories with proper permissions
mkdir -p {db,logs,data,models,outputs,reports}
mkdir -p data/{predictions,validations}
mkdir -p frontend/{css,js}

# Set appropriate permissions
chmod 755 db logs data models outputs reports
chmod 644 config/config.ini
```

---

## Configuration Setup

### Step 1: Configuration File Customization
```bash
# Copy and customize configuration
cp config/config.ini config/config.ini.backup
nano config/config.ini
```

#### Key Configuration Sections to Review:

**Database and File Paths:**
```ini
[paths]
db_file = db/shiolplus.db
log_file = logs/shiolplus.log
model_file = models/shiolplus.pkl
```

**Pipeline Settings:**
```ini
[pipeline]
default_predictions_count = 5
execution_schedule = weekly
auto_execution_enabled = true
pipeline_timeout_seconds = 3600
```

**Logging Configuration:**
```ini
pipeline_log_level = INFO
log_retention_days = 90
max_log_file_size_mb = 100
max_log_files = 10
```

### Step 2: Database Initialization
```bash
# Initialize the database (first run)
python main.py --step data

# Verify database creation
ls -la db/
# Should see: shiolplus.db
```

### Step 3: Log Directory Setup and Permissions
```bash
# Ensure log directory exists with proper permissions
sudo mkdir -p /var/log/shiolplus
sudo chown shiolplus:shiolplus /var/log/shiolplus
sudo chmod 755 /var/log/shiolplus

# Update config to use system log directory (optional)
# Edit config/config.ini:
# log_file = /var/log/shiolplus/shiolplus.log
```

### Step 4: Environment Variables (if needed)
```bash
# Create environment file
cat > ~/.shiolplus_env << 'EOF'
export SHIOLPLUS_HOME=/home/shiolplus/shiolplus
export SHIOLPLUS_CONFIG=/home/shiolplus/shiolplus/config/config.ini
export PYTHONPATH=/home/shiolplus/shiolplus:$PYTHONPATH
EOF

# Load environment variables
source ~/.shiolplus_env

# Add to shell profile for persistence
echo "source ~/.shiolplus_env" >> ~/.bashrc
```

---

## Execution Instructions

### Basic Pipeline Execution

#### Run Full Pipeline
```bash
# Activate virtual environment
cd ~/shiolplus
source venv/bin/activate

# Run complete pipeline
python main.py

# Run with verbose output
python main.py --verbose
```

#### Run Individual Pipeline Steps
```bash
# Data update only
python main.py --step data

# Adaptive analysis
python main.py --step adaptive

# Weight optimization
python main.py --step weights

# Generate predictions
python main.py --step prediction

# Historical validation
python main.py --step validation

# Performance analysis
python main.py --step performance

# Generate reports
python main.py --step reports
```

#### Check Pipeline Status
```bash
# Get current pipeline status
python main.py --status

# Use custom configuration file
python main.py --config /path/to/custom/config.ini --status
```

### API Server for Web Interface

#### Start API Server
```bash
# Start development server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Start with auto-reload (development)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Start in background
nohup uvicorn src.api:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
```

#### Access Web Interface
```bash
# Local access
curl http://localhost:8000

# Remote access (replace with your server IP)
curl http://your-server-ip:8000
```

### Command Line Options Reference
```bash
# Available command line options
python main.py --help

# Common usage patterns:
python main.py                           # Full pipeline
python main.py --step prediction        # Single step
python main.py --status                 # Status check
python main.py --config custom.ini      # Custom config
python main.py --verbose               # Detailed output
```

---

## Service Setup (Optional)

### Systemd Service Configuration

#### Create Pipeline Service
```bash
# Create systemd service file
sudo tee /etc/systemd/system/shiolplus-pipeline.service > /dev/null << 'EOF'
[Unit]
Description=SHIOL+ Pipeline Service
After=network.target

[Service]
Type=oneshot
User=shiolplus
Group=shiolplus
WorkingDirectory=/home/shiolplus/shiolplus
Environment=PATH=/home/shiolplus/shiolplus/venv/bin
ExecStart=/home/shiolplus/shiolplus/venv/bin/python main.py
StandardOutput=journal
StandardError=journal
SyslogIdentifier=shiolplus-pipeline

[Install]
WantedBy=multi-user.target
EOF
```

#### Create API Service
```bash
# Create API service file
sudo tee /etc/systemd/system/shiolplus-api.service > /dev/null << 'EOF'
[Unit]
Description=SHIOL+ API Server
After=network.target

[Service]
Type=simple
User=shiolplus
Group=shiolplus
WorkingDirectory=/home/shiolplus/shiolplus
Environment=PATH=/home/shiolplus/shiolplus/venv/bin
ExecStart=/home/shiolplus/shiolplus/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=shiolplus-api

[Install]
WantedBy=multi-user.target
EOF
```

#### Enable and Start Services
```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable services for automatic startup
sudo systemctl enable shiolplus-api.service

# Start API service
sudo systemctl start shiolplus-api.service

# Check service status
sudo systemctl status shiolplus-api.service
```

### Cron Job Setup for Scheduled Execution

#### Create Cron Job
```bash
# Edit crontab for shiolplus user
crontab -e

# Add weekly execution (Mondays at 2:00 AM)
0 2 * * 1 cd /home/shiolplus/shiolplus && /home/shiolplus/shiolplus/venv/bin/python main.py >> logs/cron.log 2>&1

# Add daily data update (every day at 6:00 AM)
0 6 * * * cd /home/shiolplus/shiolplus && /home/shiolplus/shiolplus/venv/bin/python main.py --step data >> logs/cron.log 2>&1
```

#### Verify Cron Jobs
```bash
# List current cron jobs
crontab -l

# Check cron logs
tail -f logs/cron.log
```

### Process Monitoring with Supervisor (Alternative)

#### Install Supervisor
```bash
# Ubuntu/Debian
sudo apt install supervisor

# CentOS/RHEL/Fedora
sudo dnf install supervisor
```

#### Configure Supervisor
```bash
# Create supervisor configuration
sudo tee /etc/supervisor/conf.d/shiolplus.conf > /dev/null << 'EOF'
[program:shiolplus-api]
command=/home/shiolplus/shiolplus/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000
directory=/home/shiolplus/shiolplus
user=shiolplus
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/shiolplus-api.log
EOF

# Update supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update

# Start the service
sudo supervisorctl start shiolplus-api
```

### Log Rotation Setup
```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/shiolplus > /dev/null << 'EOF'
/home/shiolplus/shiolplus/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 shiolplus shiolplus
    postrotate
        systemctl reload shiolplus-api.service > /dev/null 2>&1 || true
    endscript
}
EOF

# Test logrotate configuration
sudo logrotate -d /etc/logrotate.d/shiolplus
```

---

## Verification Steps

### Step 1: Installation Verification
```bash
# Check Python environment
cd ~/shiolplus
source venv/bin/activate
python --version
pip list | grep -E "(pandas|numpy|scikit-learn|xgboost|fastapi)"
```

### Step 2: Configuration Verification
```bash
# Test configuration loading
python -c "
import configparser
config = configparser.ConfigParser()
config.read('config/config.ini')
print('Configuration sections:', list(config.sections()))
print('Database file:', config.get('paths', 'db_file'))
"
```

### Step 3: Database Verification
```bash
# Check database initialization
python main.py --status

# Expected output should show:
# - Database Initialized: True
# - Configuration Loaded: True
# - Database Records: > 0
```

### Step 4: Pipeline Execution Test
```bash
# Run a single step to test pipeline
python main.py --step data

# Check for successful completion
echo $?  # Should return 0 for success
```

### Step 5: API Server Test
```bash
# Start API server in background
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for startup
sleep 5

# Test API endpoints
curl -s http://localhost:8000/api/v1/predict | python -m json.tool
curl -s http://localhost:8000/api/v1/pipeline/status | python -m json.tool

# Stop test server
kill $API_PID
```

### Step 6: Web Interface Test
```bash
# Test web interface accessibility
curl -I http://localhost:8000/
# Should return: HTTP/1.1 200 OK

# Test static file serving
curl -s http://localhost:8000/ | grep -i "SHIOL"
```

### Expected Outputs and Results

#### Successful Pipeline Execution
```
STARTING SHIOL+ PHASE 5 FULL PIPELINE EXECUTION
STEP 1/7: Data Update
✓ data_update completed successfully
STEP 2/7: Adaptive Analysis
✓ adaptive_analysis completed successfully
...
SHIOL+ PHASE 5 PIPELINE EXECUTION COMPLETED SUCCESSFULLY
Total execution time: 0:02:15
```

#### Successful API Response
```json
{
  "prediction": [12, 25, 33, 41, 58, 19],
  "method": "deterministic",
  "score_total": 0.7234,
  "dataset_hash": "d2a34006806ab9c8"
}
```

---

## Monitoring and Maintenance

### Log File Locations and Monitoring

#### Primary Log Files
```bash
# Main application logs
tail -f logs/shiolplus.log

# API server logs (if using systemd)
sudo journalctl -u shiolplus-api.service -f

# Cron execution logs
tail -f logs/cron.log

# System logs for SHIOL+ services
sudo journalctl -t shiolplus-pipeline -f
```

#### Log Analysis Commands
```bash
# Check for errors in the last 24 hours
grep -i error logs/shiolplus.log | tail -20

# Monitor pipeline execution patterns
grep "PIPELINE EXECUTION" logs/shiolplus.log | tail -10

# Check API request patterns
grep "Received request" logs/shiolplus.log | tail -20
```

### Database Backup Procedures

#### Automated Backup Script
```bash
# Create backup script
cat > ~/backup_shiolplus.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/shiolplus/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_FILE="/home/shiolplus/shiolplus/db/shiolplus.db"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
cp $DB_FILE $BACKUP_DIR/shiolplus_backup_$DATE.db

# Compress backup
gzip $BACKUP_DIR/shiolplus_backup_$DATE.db

# Remove backups older than 30 days
find $BACKUP_DIR -name "shiolplus_backup_*.db.gz" -mtime +30 -delete

echo "Backup completed: shiolplus_backup_$DATE.db.gz"
EOF

chmod +x ~/backup_shiolplus.sh
```

#### Schedule Automated Backups
```bash
# Add to crontab (daily backup at 1:00 AM)
crontab -e
# Add line:
0 1 * * * /home/shiolplus/backup_shiolplus.sh >> /home/shiolplus/backup.log 2>&1
```

#### Manual Backup Commands
```bash
# Create immediate backup
cp db/shiolplus.db backups/shiolplus_manual_$(date +%Y%m%d_%H%M%S).db

# Restore from backup
cp backups/shiolplus_backup_YYYYMMDD_HHMMSS.db db/shiolplus.db
```

### System Health Checks

#### Health Check Script
```bash
# Create health check script
cat > ~/health_check.sh << 'EOF'
#!/bin/bash
echo "=== SHIOL+ Health Check ==="
echo "Date: $(date)"
echo

# Check disk space
echo "Disk Usage:"
df -h /home/shiolplus/shiolplus | tail -1

# Check memory usage
echo "Memory Usage:"
free -h | grep Mem

# Check if API service is running
echo "API Service Status:"
systemctl is-active shiolplus-api.service 2>/dev/null || echo "Not configured"

# Check database size
echo "Database Size:"
ls -lh db/shiolplus.db 2>/dev/null || echo "Database not found"

# Check recent log entries
echo "Recent Errors (last 24h):"
find logs/ -name "*.log" -mtime -1 -exec grep -i error {} \; | wc -l

# Check API endpoint
echo "API Health:"
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/pipeline/health 2>/dev/null || echo "API not accessible"

echo "=== Health Check Complete ==="
EOF

chmod +x ~/health_check.sh
```

#### Run Health Checks
```bash
# Manual health check
./health_check.sh

# Schedule regular health checks (every 6 hours)
crontab -e
# Add line:
0 */6 * * * /home/shiolplus/health_check.sh >> /home/shiolplus/health.log 2>&1
```

### Performance Monitoring

#### Monitor System Resources
```bash
# CPU and memory usage
top -p $(pgrep -f "python.*main.py\|uvicorn.*api")

# Disk I/O monitoring
iotop -p $(pgrep -f "python.*main.py")

# Network monitoring (if API is running)
netstat -tulpn | grep :8000
```

#### Application Performance Metrics
```bash
# Check pipeline execution times
grep "execution time" logs/shiolplus.log | tail -10

# Monitor prediction generation performance
grep "prediction generated" logs/shiolplus.log | tail -10

# Check database query performance
grep -i "database" logs/shiolplus.log | grep -i "slow\|timeout" | tail -10
```

---

## Security Considerations

### File Permissions and Ownership

#### Set Secure Permissions
```bash
# Set ownership
sudo chown -R shiolplus:shiolplus /home/shiolplus/shiolplus

# Set directory permissions
find /home/shiolplus/shiolplus -type d -exec chmod 755 {} \;

# Set file permissions
find /home/shiolplus/shiolplus -type f -exec chmod 644 {} \;

# Make scripts executable
chmod +x /home/shiolplus/shiolplus/main.py
chmod +x /home/shiolplus/*.sh

# Secure configuration files
chmod 600 config/config.ini
```

#### Protect Sensitive Files
```bash
# Secure database file
chmod 600 db/shiolplus.db

# Secure log files
chmod 640 logs/*.log

# Secure model files
chmod 600 models/*.pkl
```

### Network Security (Firewall Rules)

#### Configure UFW (Ubuntu/Debian)
```bash
# Enable UFW
sudo ufw enable

# Allow SSH (if needed)
sudo ufw allow ssh

# Allow API port (adjust as needed)
sudo ufw allow 8000/tcp

# Allow specific IP ranges only (recommended)
sudo ufw allow from 192.168.1.0/24 to any port 8000

# Check firewall status
sudo ufw status verbose
```

#### Configure firewalld (CentOS/RHEL/Fedora)
```bash
# Start and enable firewalld
sudo systemctl start firewalld
sudo systemctl enable firewalld

# Allow API port
sudo firewall-cmd --permanent --add-port=8000/tcp

# Allow specific IP range
sudo firewall-cmd --permanent --add-rich-rule="rule family='ipv4' source address='192.168.1.0/24' port protocol='tcp' port='8000' accept"

# Reload firewall
sudo firewall-cmd --reload

# Check configuration
sudo firewall-cmd --list-all
```

### User Account Setup

#### Create Restricted Service Account
```bash
# Create system user (no login shell)
sudo useradd -r -s /bin/false -d /home/shiolplus -m shiolplus-service

# Set up directory structure
sudo mkdir -p /home/shiolplus-service/{shiolplus,backups}
sudo chown -R shiolplus-service:shiolplus-service /home/shiolplus-service

# Copy application files
sudo cp -r /home/shiolplus/shiolplus/* /home/shiolplus-service/shiolplus/
sudo chown -R shiolplus-service:shiolplus-service /home/shiolplus-service/shiolplus
```

#### Configure sudo Access (if needed)
```bash
# Create sudoers file for limited access
sudo tee /etc/sudoers.d/shiolplus << 'EOF'
# Allow shiolplus user to restart its own services
shiolplus ALL=(root) NOPASSWD: /bin/systemctl restart shiolplus-api.service
shiolplus ALL=(root) NOPASSWD: /bin/systemctl status shiolplus-api.service
EOF
```

### Data Protection Measures

#### Encrypt Sensitive Data
```bash
# Install encryption tools
sudo apt install gnupg  # Ubuntu/Debian
sudo dnf install gnupg2  # CentOS/RHEL/Fedora

# Create GPG key for backups
gpg --gen-key

# Encrypt database backups
gpg --encrypt --recipient your-email@domain.com backups/shiolplus_backup_*.db
```

#### Secure Configuration Management
```bash
# Use environment variables for sensitive settings
cat > ~/.shiolplus_secrets << 'EOF'
export SHIOLPLUS_DB_PASSWORD="your-secure-password"
export SHIOLPLUS_API_SECRET="your-api-secret-key"
EOF

chmod 600 ~/.shiolplus_secrets
source ~/.shiolplus_secrets
```

#### Network Security Best Practices
```bash
# Use reverse proxy with SSL (nginx example)
sudo apt install nginx certbot python3-certbot-nginx

# Configure nginx proxy
sudo tee /etc/nginx/sites-available/shiolplus << 'EOF'
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable site and get SSL certificate
sudo ln -s /etc/nginx/sites-available/shiolplus /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com
```

---

## Troubleshooting Section

### Common Installation Issues and Solutions

#### Issue: Python Version Compatibility
**Symptoms:**
```
ERROR: Package requires Python '>=3.8' but the running Python is 3.7
```

**Solution:**
```bash
# Install Python 3.8+ from deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev

# Create virtual environment with specific Python version
python3.8 -m venv venv
source venv/bin/activate
```

#### Issue: Package Installation Failures
**Symptoms:**
```
ERROR: Failed building wheel for package-name
```

**Solution:**
```bash
# Install build dependencies
sudo apt install build-essential python3-dev libffi-dev libssl-dev

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install packages one by one to identify problematic package
pip install pandas numpy scikit-learn xgboost joblib loguru statsmodels requests fastapi uvicorn apscheduler
```

#### Issue: Permission Denied Errors
**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'db/shiolplus.db'
```

**Solution:**
```bash
# Fix ownership and permissions
sudo chown -R $USER:$USER ~/shiolplus
chmod -R 755 ~/shiolplus
chmod 644 ~/shiolplus/db/shiolplus.db
```

#### Issue: Database Initialization Fails
**Symptoms:**
```
ERROR: Failed to initialize database: no such table
```

**Solution:**
```bash
# Remove corrupted database and reinitialize
rm -f db/shiolplus.db
python main.py --step data

# If still failing, check disk space
df -h .
```

### Error Messages and Their Meanings

#### Database Errors
```bash
# Error: "database is locked"
# Cause: Another process is using the database
# Solution: 
ps aux | grep python | grep shiolplus
kill <process_id>

# Error: "no such table: draws"
# Cause: Database not properly initialized
# Solution:
python main.py --step data
```

#### Model Errors
```bash
# Error: "Model file not found"
# Cause: Model hasn't been trained yet
# Solution:
python -c "from src.predictor import Predictor; p = Predictor(); p.train_model()"

# Error: "Prediction failed: insufficient data"
# Cause: Not enough historical data
# Solution:
python main.py --step data  # Update data first
```

#### API Errors
```bash
# Error: "Address already in use"
# Cause: Port 8000 is already occupied
# Solution:
lsof -i :8000
kill <process_id>
# Or use different port:
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

#### Memory Errors
```bash
# Error: "MemoryError" during model training
# Cause: Insufficient RAM
# Solution:
# Add swap space:
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Optimization Tips

#### Database Optimization
```bash
# Vacuum database to reclaim space
sqlite3 db/shiolplus.db "VACUUM;"

# Analyze database for query optimization
sqlite3 db/shiolplus.db "ANALYZE;"

# Check database integrity
sqlite3 db/shiolplus.db "PRAGMA integrity_check;"
```

#### Memory Optimization
```bash
# Monitor memory usage during execution
python -c "
import psutil
import time
process = psutil.Process()
while True:
    print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
    time.sleep(5)
" &

# Run pipeline and monitor memory
python main.py
```

#### CPU Optimization
```bash
# Set CPU affinity for intensive operations
taskset -c 0,1 python main.py

# Use nice to lower priority
nice -n 10 python main.py
```

#### Disk I/O Optimization
```bash
# Move database to faster storage (SSD)
sudo mkdir -p /opt/shiolplus/db
sudo chown shiolplus:shiolplus /opt/shiolplus/db
mv db/shiolplus.db /opt/shiolplus/db/
ln -s /opt/shiolplus/db/shiolplus.db db/shiolplus.db

# Use tmpfs for temporary files
sudo mount -t tmpfs -o size=1G tmpfs /tmp/shiolplus
```

### Network and Connectivity Issues

#### API Server Not Accessible
```bash
# Check if service is running
systemctl status shiolplus-api.service

# Check port binding
netstat -tulpn | grep :8000

# Test local connectivity
curl -v http://localhost:8000/api/v1/pipeline/health

# Check firewall
sudo ufw status
sudo iptables -L | grep 8000
```

#### Data Update Failures
```bash
# Test internet connectivity
ping -c 4 google.com

# Check DNS resolution
nslookup powerball-data-source.com

# Test with verbose curl
curl -v https://data-source-url/

# Use proxy if needed
export https_proxy=http://proxy-server:port
python main.py --step data
```

### Log Analysis and Debugging

#### Enable Debug Logging
```bash
# Modify config/config.ini
[pipeline]
pipeline_log_level = DEBUG

# Or set environment variable
export SHIOLPLUS_LOG_LEVEL=DEBUG
python main.py
```

#### Common Log Patterns to Monitor
```bash
# Check for critical errors
grep -i "critical\|fatal\|error" logs/shiolplus.log | tail -20

# Monitor pipeline execution flow
grep "STEP [0-9]/7" logs/shiolplus.log | tail -10

# Check prediction generation
grep "prediction generated" logs/shiolplus.log | tail -10

# Monitor API requests
grep "Received request" logs/shiolplus.log | tail -20
```

#### Log Rotation Issues
```bash
# Check logrotate configuration
sudo logrotate -d /etc/logrotate.d/shiolplus

# Force log rotation
sudo logrotate -f /etc/logrotate.d/shiolplus

# Check log file permissions after rotation
ls -la logs/
```

### Contact Information for Support

#### Self-Help Resources
- **Documentation**: Check `docs/` directory for additional guides
- **Configuration**: Review `config/config.ini` for all available options
- **Logs**: Always check `logs/shiolplus.log` for detailed error information

#### System Information for Support Requests
When requesting support, please provide:

```bash
# Generate system information report
cat > system_info.txt << 'EOF'
=== SHIOL+ System Information ===
Date: $(date)
Hostname: $(hostname)
OS: $(cat /etc/os-release | grep PRETTY_NAME)
Python Version: $(python --version)
Disk Space: $(df -h .)
Memory: $(free -h)
SHIOL+ Version: $(grep -r "version" config/ 2>/dev/null || echo "Unknown")

=== Recent Errors ===
$(tail -20 logs/shiolplus.log | grep -i error || echo "No recent errors")

=== Configuration ===
$(cat config/config.ini)
EOF

# Send this file when requesting support
```

#### Emergency Recovery Procedures
```bash
# Complete system reset (preserves data)
cp db/shiolplus.db ~/shiolplus_backup.db
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/shiolplus_backup.db db