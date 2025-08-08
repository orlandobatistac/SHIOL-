
# SHIOL+ v6.0: Advanced AI-Powered Lottery Analysis System with Configuration Dashboard

An intelligent, enterprise-grade system designed to analyze historical lottery data and predict combinations using advanced Machine Learning techniques with a comprehensive configuration dashboard and automated pipeline orchestrator.

**🌐 Live Demo**: [https://shiolplus.replit.app](https://shiolplus.replit.app)

## Project Summary

**SHIOL+ (System for Hybrid Intelligence Optimization and Learning)** is a comprehensive software platform that analyzes historical Powerball lottery draw data to identify statistical patterns and generate optimized predictions. The system combines artificial intelligence models with adaptive learning mechanisms, providing a complete pipeline for data processing, prediction generation, validation, and performance analysis.

Version 6.0 introduces a revolutionary **Configuration Dashboard** that transforms SHIOL+ into a fully configurable, enterprise-ready system with real-time monitoring, advanced database management, comprehensive system controls, and **enterprise-grade security**.

> **Important**: This tool was created for educational, research, and entertainment purposes. The lottery is a game of chance, and SHIOL+ **does not guarantee prizes or winnings**. Always play responsibly.

## 🚀 What's New in v6.0

### **Advanced Configuration Dashboard**
- **Real-time System Monitoring**: CPU, memory, disk usage with live graphs
- **Pipeline Configuration**: Schedule execution, configure prediction methods, adjust scoring weights
- **Database Management**: Complete database control with cleanup, backup, and statistics
- **Model Management**: AI model retraining, backup, and performance tracking
- **Configuration Profiles**: Pre-built profiles (Conservative, Aggressive, Balanced, Custom)
- **Notification System**: Email and browser notifications for system events
- **Advanced Analytics**: Performance trends, win rate analysis, and method comparison

### **Enterprise Security Features**
- **XSS Protection**: Complete elimination of innerHTML vulnerabilities with safe DOM manipulation
- **SQL Injection Prevention**: Parameterized queries and safe table operations
- **Command Injection Security**: Secure subprocess execution with proper escaping
- **CORS Security**: Configurable origin restrictions for production environments
- **Session Management**: Secure HttpOnly cookie-based authentication
- **Security Headers**: Comprehensive HTTP security headers implementation
- **Input Sanitization**: Complete user input validation and sanitization

### **Enterprise Features**
- **Import/Export Settings**: Save and share configuration templates
- **Session Management**: Configurable timeouts and security controls
- **System Health Monitoring**: Comprehensive health checks and alerts
- **Audit Logging**: Complete system activity tracking
- **Backup & Recovery**: Automated backup systems with one-click restore

## Key Features

### 🎯 Phase 6 Configuration Management System

*   **Unified Dashboard**: Single interface for all system configuration and monitoring
*   **Profile-Based Configuration**: Quick setup with predefined configuration profiles
*   **Real-time Monitoring**: Live system health, resource usage, and pipeline status
*   **Advanced Database Controls**: Complete database management with cleanup and backup tools
*   **Model Management Suite**: AI model training, validation, and performance optimization
*   **Notification System**: Multi-channel alerts for system events and winning predictions
*   **Configuration Import/Export**: Share and backup system configurations

### 🔒 Enterprise Security System

*   **XSS Prevention**: Complete protection against Cross-Site Scripting attacks
*   **SQL Injection Protection**: Parameterized queries and safe database operations
*   **Command Injection Security**: Safe subprocess execution with proper validation
*   **CSRF Protection**: Cross-Site Request Forgery prevention mechanisms
*   **Secure Authentication**: HttpOnly cookies with secure session management
*   **Input Validation**: Comprehensive user input sanitization and validation
*   **Security Headers**: Full HTTP security headers implementation

### 🤖 Enhanced AI Pipeline System

*   **Smart Scheduling**: Configurable execution days and times with timezone support
*   **Multiple Prediction Methods**: Smart AI, Deterministic, Ensemble, and Adaptive Hybrid
*   **Dynamic Weight Adjustment**: Real-time scoring weight optimization based on performance
*   **Batch Processing**: Configure prediction quantities from 10 to 1000 per execution
*   **Adaptive Learning**: Continuous improvement based on historical performance
*   **Performance Analytics**: Comprehensive metrics and trend analysis

### 🌐 Advanced Web Interface

*   **Public Interface**: Clean, modern interface for viewing predictions and statistics
*   **Admin Dashboard**: Comprehensive configuration and monitoring interface
*   **Real-time Updates**: Live data refresh and status monitoring
*   **Mobile Responsive**: Optimized for desktop, tablet, and mobile devices
*   **RESTful API**: Complete API suite for integration and automation
*   **Security-First Design**: All interfaces protected against common web vulnerabilities

### 📊 Comprehensive Analytics & Monitoring

*   **Performance Dashboards**: Win rates, accuracy metrics, and trend analysis
*   **System Health Monitoring**: Resource usage, database health, and service status
*   **Historical Analysis**: Long-term performance tracking and pattern identification
*   **Comparative Analytics**: Method performance comparison and optimization recommendations
*   **Real-time Logging**: Advanced logging with filtering, search, and export capabilities

## Quick Start

### Simple Setup & Execution

The easiest way to get started with SHIOL+ v6.0:

```bash
# Install dependencies
pip install -r requirements.txt
pip install bcrypt==4.1.2 psutil==5.9.8

# Initialize the system
python src/database.py

# Run the complete pipeline
python main.py

# Start the web server
python main.py --server --host 0.0.0.0 --port 3000
```

### Access the System

After starting the server, access the different interfaces:

*   **Public Interface**: `http://localhost:3000/` - View predictions and statistics
*   **Admin Login**: `http://localhost:3000/login.html` - Administrator login
*   **Configuration Dashboard**: `http://localhost:3000/dashboard.html` - Complete system control
*   **Live Demo**: [https://shiolplus.replit.app](https://shiolplus.replit.app) - Public demonstration

**Default Admin Credentials**: `admin` / `shiol2024!` (Change immediately in production)

## Security Improvements

### 🛡️ Security Vulnerabilities Resolved

#### XSS (Cross-Site Scripting) Protection
- **Fixed**: All `innerHTML` usage replaced with safe DOM manipulation
- **Files Secured**: `app.js`, `config-manager.js`, `powerball-utils.js`, `public.js`
- **Method**: Using `textContent`, `createElement()`, and `appendChild()` for safe content insertion

#### SQL Injection Prevention
- **Fixed**: Dynamic SQL construction replaced with parameterized queries
- **Files Secured**: `api_database_endpoints.py`
- **Method**: Predefined safe queries and parameterized statements

#### Command Injection Security
- **Fixed**: Subprocess calls secured with proper escaping
- **Files Secured**: `main.py`
- **Method**: Using `shlex.escape()` for command parameter sanitization

#### Authentication Security
- **Enhanced**: HttpOnly cookie-based session management
- **Files Updated**: `auth.js`, `auth.py`
- **Features**: Secure token storage, CSRF protection, session validation

### 🔧 Security Best Practices Implemented

1. **Input Sanitization**: All user inputs validated and sanitized
2. **Output Encoding**: Safe content rendering without HTML injection
3. **Parameterized Queries**: Database operations use prepared statements
4. **Secure Sessions**: HttpOnly, Secure, SameSite cookie attributes
5. **CORS Configuration**: Configurable origin restrictions
6. **Error Handling**: Secure error messages without information disclosure

## Configuration Dashboard Features

### 🎯 Pipeline Configuration
- **Execution Scheduling**: Set days and times for automatic pipeline execution
- **Timezone Management**: Configure system timezone for accurate scheduling
- **Prediction Settings**: Adjust quantity, methods, and scoring weights
- **Auto-execution Control**: Enable/disable automated pipeline runs

### 🗄️ Database Management
- **Real-time Statistics**: View record counts, database size, and health metrics
- **Cleanup Tools**: Selective data cleanup with safety confirmations
- **Backup System**: Create and manage database backups
- **Performance Monitoring**: Track database performance and optimization

### 🤖 Model Management
- **Training Controls**: Retrain models with latest data
- **Performance Tracking**: Monitor model accuracy and effectiveness
- **Backup & Recovery**: Backup and restore model states
- **Version Management**: Track model versions and improvements

### 📊 System Monitoring
- **Resource Usage**: Real-time CPU, memory, and disk monitoring
- **Pipeline Status**: Track execution progress and history
- **Health Checks**: Comprehensive system health monitoring
- **Performance Metrics**: Detailed analytics and trend analysis

### ⚙️ Advanced Configuration
- **Configuration Profiles**: Quick setup with predefined profiles
  - **Conservative**: Lower risk, higher accuracy focus
  - **Aggressive**: Higher volume, diverse prediction strategies  
  - **Balanced**: Optimal balance of all factors
  - **Custom**: Full manual configuration control
- **Import/Export**: Save and share configuration templates
- **Notification Settings**: Configure email and browser alerts
- **Security Controls**: Session management and access controls

## System Architecture

### Core Components

*   **Main Pipeline (`main.py`)**: Complete pipeline orchestrator with web server integration
*   **Configuration Dashboard (`frontend/dashboard.html`)**: Advanced web-based configuration interface
*   **API Layer (`src/api.py`)**: RESTful API with v6.0 configuration endpoints
*   **Database Engine (`src/database.py`)**: Enhanced SQLite database with analytics
*   **AI Engine (`src/predictor.py`)**: Machine learning models with adaptive learning
*   **Intelligent Generator (`src/intelligent_generator.py`)**: Advanced prediction algorithms
*   **Adaptive Feedback (`src/adaptive_feedback.py`)**: Performance-based learning system

### New v6.0 Components

*   **System Monitor**: Real-time resource and performance monitoring
*   **Configuration Manager**: Centralized configuration management system
*   **Database Manager**: Advanced database operations and maintenance
*   **Model Manager**: AI model lifecycle management
*   **Notification Engine**: Multi-channel notification system
*   **Analytics Engine**: Advanced performance analytics and reporting
*   **Security Layer**: Comprehensive security controls and validation

### Security Components

*   **Authentication System (`src/auth.py`)**: Secure user authentication and session management
*   **Input Validator**: Comprehensive input validation and sanitization
*   **SQL Security**: Parameterized queries and injection prevention
*   **XSS Protection**: Safe DOM manipulation and content rendering
*   **CSRF Protection**: Cross-site request forgery prevention

## API Endpoints

### Core Prediction Endpoints
- `GET /api/v1/predict/smart` - Smart AI predictions
- `GET /api/v1/predict-deterministic` - Deterministic predictions
- `GET /api/v1/prediction-history` - Historical predictions
- `GET /api/v1/prediction-history-grouped` - Grouped prediction analytics

### v6.0 Configuration Endpoints
- `GET /api/v1/system/stats` - Real-time system statistics
- `GET /api/v1/database/stats` - Database health and statistics
- `POST /api/v1/database/cleanup` - Database maintenance operations
- `POST /api/v1/database/backup` - Create database backups
- `GET /api/v1/config/load` - Load system configuration
- `POST /api/v1/config/save` - Save system configuration
- `GET /api/v1/analytics/performance` - Performance analytics
- `POST /api/v1/pipeline/test` - Test pipeline execution
- `POST /api/v1/model/retrain` - Trigger model retraining
- `GET /api/v1/logs` - System logs and debugging

### Pipeline Control Endpoints
- `GET /api/v1/pipeline/status` - Pipeline execution status
- `POST /api/v1/pipeline/trigger` - Manual pipeline execution
- `GET /api/v1/pipeline/health` - System health check

### Authentication Endpoints
- `POST /api/v1/auth/login` - Secure user authentication
- `POST /api/v1/auth/logout` - Session termination
- `POST /api/v1/auth/verify` - Session validation

## Configuration Management

### Configuration Profiles

SHIOL+ v6.0 includes predefined configuration profiles for different use cases:

#### Conservative Profile
- **Predictions**: 50 per execution
- **Method**: Deterministic scoring
- **Focus**: Higher accuracy, lower risk
- **Weights**: Probability (50%), Historical (30%), Diversity (10%), Risk (10%)

#### Aggressive Profile  
- **Predictions**: 500 per execution
- **Method**: Ensemble method
- **Focus**: Maximum coverage, diverse strategies
- **Weights**: Diversity (35%), Probability (30%), Historical (20%), Risk (15%)

#### Balanced Profile
- **Predictions**: 100 per execution  
- **Method**: Smart AI Pipeline
- **Focus**: Optimal balance of all factors
- **Weights**: Probability (40%), Diversity (25%), Historical (20%), Risk (15%)

### Configuration File Structure

The system uses an enhanced `config/config.ini` file:

```ini
[pipeline]
execution_days = 0,2,5  # Monday, Wednesday, Saturday
execution_time = 02:00
timezone = America/New_York
auto_execution_enabled = true

[predictions]
default_count = 100
default_method = smart_ai

[scoring]
probability_weight = 40
diversity_weight = 25
historical_weight = 20
risk_weight = 15

[notifications]
admin_email = admin@shiolplus.com
email_results = true
email_errors = true
browser_notifications = true

[security]
session_timeout = 60
require_2fa = false
cors_origins = https://shiolplus.replit.app
```

## Database Management

### Enhanced Database Features
- **Automated Cleanup**: Selective data cleanup with confirmation dialogs
- **Backup System**: Automated and manual backup creation
- **Health Monitoring**: Real-time database health and performance metrics
- **Migration Support**: Automatic schema updates and data migration
- **Analytics Integration**: Built-in analytics for performance tracking
- **Security**: Protected against SQL injection with parameterized queries

### Database Tables
- `draws` - Historical lottery data
- `predictions_log` - All generated predictions with metadata
- `validation_results` - Prediction validation results
- `adaptive_weights` - Dynamic scoring weights
- `adaptive_performance` - Performance tracking data
- `system_logs` - System events and audit trail
- `user_sessions` - Authentication and session management

## Performance Analytics

### Key Metrics
- **Win Rate**: Percentage of predictions with prizes
- **Average Score**: Mean prediction quality score
- **Method Performance**: Comparative analysis of prediction methods
- **Trend Analysis**: Performance trends over time
- **ROI Analysis**: Return on investment calculations

### Monitoring Dashboards
- **Real-time Monitoring**: Live system resource usage
- **Pipeline Analytics**: Execution history and performance
- **Database Analytics**: Storage optimization and query performance
- **Model Analytics**: AI model accuracy and improvement tracking

## Security & Access Control

### Authentication System
- **Secure Login**: Encrypted password authentication with bcrypt
- **Session Management**: Configurable session timeouts with HttpOnly cookies
- **Role-based Access**: Different access levels for different users
- **Audit Logging**: Complete activity tracking and logging

### Security Features  
- **Password Encryption**: Bcrypt-based password hashing
- **Session Security**: Secure session token management with HttpOnly cookies
- **Access Control**: Protected endpoints and functionality
- **Data Protection**: Encrypted sensitive data storage
- **XSS Prevention**: Complete protection against script injection
- **SQL Injection Prevention**: Parameterized queries for all database operations
- **CSRF Protection**: Cross-site request forgery prevention
- **Command Injection Prevention**: Secure subprocess execution

## System Requirements

### Minimum Requirements
- Python 3.10+
- 4GB RAM (recommended: 8GB+)
- 2GB free disk space
- Modern web browser for dashboard access

### Recommended Setup
- Python 3.11+
- 8GB+ RAM for optimal performance  
- SSD storage for database operations
- High-speed internet for data updates

### Security Requirements
- HTTPS enabled for production deployment
- Environment variables for sensitive configuration
- Regular security updates and monitoring

## Installation & Setup

### Quick Installation

```bash
# Clone the repository
git clone <REPOSITORY_URL>
cd SHIOL-PLUS-V6

# Install dependencies
pip install -r requirements.txt
pip install bcrypt==4.1.2 psutil==5.9.8

# Initialize database
python src/database.py

# Configure system (edit config/config.ini)
# Run initial pipeline
python main.py

# Start web server
python main.py --server --host 0.0.0.0 --port 3000
```

### Production Deployment on Replit

For secure production deployment:

1. **Environment Setup**: Configure environment variables and secrets
2. **Database Configuration**: Set up persistent database storage  
3. **Security Configuration**: Change default credentials and enable security features
4. **Monitoring Setup**: Configure notification endpoints and monitoring
5. **Backup Configuration**: Set up automated backup schedules
6. **CORS Configuration**: Restrict origins to production domains
7. **HTTPS Setup**: Enable SSL/TLS for secure connections

**Live Demo**: [https://shiolplus.replit.app](https://shiolplus.replit.app)

## Usage Examples

### Command Line Interface

```bash
# Full pipeline execution
python main.py

# Specific pipeline steps
python main.py --step data          # Data update only
python main.py --step prediction    # Prediction generation only  
python main.py --step validation    # Validation only

# Start web server
python main.py --server --host 0.0.0.0 --port 3000

# Check system status
python main.py --status

# Get help
python main.py --help
```

### Dashboard Operations

1. **Access Dashboard**: [https://shiolplus.replit.app/dashboard.html](https://shiolplus.replit.app/dashboard.html)
2. **Configure Pipeline**: Set execution schedule and parameters
3. **Manage Database**: Monitor health and perform maintenance
4. **Monitor System**: View real-time resource usage and performance
5. **Export Configuration**: Save current settings for backup or sharing

### API Integration

```python
import requests

# Base URL for live demo
BASE_URL = 'https://shiolplus.replit.app/api/v1'

# Get system status
response = requests.get(f'{BASE_URL}/system/stats')
stats = response.json()

# Trigger pipeline
response = requests.post(f'{BASE_URL}/pipeline/trigger')

# Get predictions
response = requests.get(f'{BASE_URL}/predict/smart?limit=10')
predictions = response.json()
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**: Check database file permissions and path
2. **Model Loading Errors**: Ensure model files exist and are not corrupted
3. **Memory Issues**: Increase available RAM or reduce batch sizes
4. **Permission Errors**: Check file system permissions for data directories
5. **Web Interface Issues**: Clear browser cache and check console for errors
6. **Security Errors**: Verify CORS configuration and authentication settings

### Debug Commands

```bash
# Check system health
python main.py --status

# Validate configuration
python -c "import configparser; c=configparser.ConfigParser(); c.read('config/config.ini'); print('Config OK')"

# Test database connection
python -c "from src.database import get_db_connection; print('DB OK' if get_db_connection() else 'DB Error')"

# Check logs
tail -f logs/shiolplus.log

# Security scan (if available)
python src/security_analyzer.py
```

## Contributing

We welcome contributions to SHIOL+ v6.0! Please follow our contribution guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

### Code Standards
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for changes
- Follow security best practices
- Validate all user inputs
- Use parameterized queries for database operations

### Security Guidelines
- Never use `innerHTML` with user data
- Always validate and sanitize user inputs
- Use parameterized queries for all database operations
- Implement proper error handling without information disclosure
- Follow principle of least privilege
- Regular security testing and code review

## Version History

- **v6.0 (Phase 6)**: Advanced Configuration Dashboard with enterprise security features
- **v5.0 (Phase 5)**: Automated pipeline system with web dashboard
- **v4.0 (Phase 4)**: Adaptive feedback system with weight optimization
- **v3.0 (Phase 3)**: Advanced analytics and performance tracking
- **v2.0 (Phase 2)**: Deterministic prediction system with validation
- **v1.0**: Basic ML prediction system with SQLite database

## Security Audit

### Recent Security Improvements (Phase 6 Final)
- ✅ **XSS Vulnerabilities**: Eliminated all innerHTML usage with safe DOM manipulation
- ✅ **SQL Injection**: Implemented parameterized queries for all database operations
- ✅ **Command Injection**: Secured subprocess execution with proper escaping
- ✅ **Session Security**: Implemented HttpOnly cookie-based authentication
- ✅ **Input Validation**: Comprehensive user input sanitization
- ✅ **CORS Security**: Configurable origin restrictions
- ✅ **Error Handling**: Secure error messages without information disclosure

### Security Testing
The system has been thoroughly tested for common web vulnerabilities including:
- Cross-Site Scripting (XSS)
- SQL Injection
- Command Injection
- Cross-Site Request Forgery (CSRF)
- Session Management vulnerabilities
- Input validation bypass attempts

## Credits

- **Creator**: Orlando Batista
- **Version**: 6.0 (Phase 6 - Advanced Configuration Dashboard with Enterprise Security)
- **Last Updated**: August 2025
- **Live Demo**: [https://shiolplus.replit.app](https://shiolplus.replit.app)

## License

Private use – All rights reserved.

---

**SHIOL+ v6.0** - Transforming lottery analysis with enterprise-grade AI, comprehensive system management, and bulletproof security.

**🌐 Experience SHIOL+ Live**: [https://shiolplus.replit.app](https://shiolplus.replit.app)

For support, documentation, and updates, visit the project repository or contact the development team.
