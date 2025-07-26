# SHIOL+ v5.0: AI-Powered Lottery Analysis System with Automated Pipeline

An intelligent system designed to analyze historical data and predict lottery combinations using Machine Learning techniques with a fully automated pipeline orchestrator.

## Project Summary

**SHIOL+ (Heuristic and Inferential Optimized Lottery System)** is a comprehensive software tool that analyzes historical Powerball lottery draw data to identify statistical patterns. The system's main objective is to use an artificial intelligence model to predict the probability of each number appearing in future draws, thereby generating optimized plays through an automated pipeline system.

The system is now powered by an **SQLite database**, ensuring data integrity and performance. It features a complete **Phase 5 automated pipeline** that handles data updates, adaptive analysis, weight optimization, prediction generation, historical validation, performance analysis, and reporting - all orchestrated through a single command.

> **Important**: This tool was created for educational, research, and entertainment purposes. The lottery is a game of chance, and SHIOL+ **does not guarantee prizes or winnings**. Always play responsibly.

## Key Features

SHIOL+ v5.0 introduces a complete automated pipeline system with enhanced prediction capabilities, web dashboard, and comprehensive monitoring:

### 🚀 Phase 5 Automated Pipeline System

*   **Simple Execution**: Run the entire system with a single command: `python main.py`
*   **7-Step Pipeline**: Automated execution of data update, adaptive analysis, weight optimization, prediction generation, historical validation, performance analysis, and reporting
*   **Weekly Scheduling**: Automatic pipeline execution every week with configurable timing
*   **Manual Triggering**: On-demand pipeline execution via web dashboard or API
*   **Real-time Monitoring**: Live pipeline status tracking and execution logs
*   **Error Recovery**: Automatic retry logic with exponential backoff
*   **Health Monitoring**: System health checks and resource monitoring

### 🌐 Web Dashboard & API

*   **Interactive Web Interface**: Modern web dashboard for system monitoring and control
*   **RESTful API**: Comprehensive API endpoints for all system functions
*   **Real-time Status**: Live pipeline status, execution history, and system health
*   **Manual Controls**: Trigger pipeline execution, view logs, and monitor performance
*   **Generated Plays Visualization**: View recent predictions with scoring details

### 🧠 Enhanced Prediction System

*   **Deterministic Predictions**: Consistent, reproducible results using multi-criteria scoring
*   **Adaptive Learning**: System learns from historical performance to optimize weights
*   **Multi-Criteria Scoring**: Probability (40%), Diversity (25%), Historical (20%), Risk-Adjusted (15%)
*   **Performance Validation**: Automatic validation against historical draws
*   **Adaptive Feedback**: Continuous improvement based on validation results

### 📊 Monitoring & Analytics

*   **Performance Analytics**: Detailed metrics on prediction accuracy and win rates
*   **Execution History**: Complete log of pipeline runs and results
*   **System Health**: CPU, memory, and disk usage monitoring
*   **Validation Reports**: Automated comparison against actual lottery results
*   **Comprehensive Logging**: Detailed logs with filtering and search capabilities

## Quick Start

### Simple Execution (Recommended)

The easiest way to run SHIOL+ v5.0 is with the automated pipeline:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python main.py
```

This single command will:
1. Update the database with latest lottery data
2. Run adaptive analysis on recent performance
3. Optimize prediction weights based on historical results
4. Generate new predictions using the deterministic system
5. Validate predictions against historical data
6. Analyze system performance metrics
7. Generate comprehensive reports

### Web Dashboard Access

After running the pipeline, start the web dashboard:

```bash
# Start the web server
uvicorn src.api:app --reload
```

Then open your browser to `http://127.0.0.1:8000` to access:
- Real-time pipeline status monitoring
- Manual pipeline triggering
- Generated plays visualization
- System health metrics
- Execution logs and history

## Project Structure

The project is organized into the following directories:

-   `main.py`: **Phase 5 Pipeline Orchestrator** - Single entry point for complete system execution
-   `src/`: Contains all the application's Python source code
    -   `api.py`: FastAPI application serving the web interface and API endpoints
    -   `scheduler.py`: Weekly scheduling system with APScheduler integration
    -   `adaptive_feedback.py`: Adaptive learning and weight optimization system
    -   `pipeline_logger.py`: Comprehensive logging system for pipeline execution
    -   `predictor.py`: ML model training and prediction logic
    -   `intelligent_generator.py`: Deterministic prediction generation with scoring
    -   `database.py`: SQLite database management and analytics
    -   `loader.py`: Data loading and database update orchestration
    -   `evaluator.py`: Historical validation and performance analysis
    -   `basic_validator.py`: Basic prediction validation system
    -   `notifications.py`: Notification system for pipeline events
-   `frontend/`: Web dashboard static files
    -   `index.html`: Main dashboard interface
    -   `css/styles.css`: Dashboard styling
    -   `js/app.js`: Interactive dashboard functionality
-   `data/`: Data storage directories
    -   `predictions/`: JSON files with detailed prediction information
    -   `validations/`: CSV files with validation results
-   `reports/`: Pipeline execution reports and analytics
-   `models/`: Trained AI model artifacts (`shiolplus.pkl`)
-   `logs/`: Application execution logs with rotation
-   `config/`: Configuration files (`config.ini`)

## API Endpoints

SHIOL+ v5.0 provides comprehensive API endpoints for all system functions:

### Pipeline Control
- `GET /api/v1/pipeline/status` - Get current pipeline status and health
- `POST /api/v1/pipeline/trigger` - Manually trigger pipeline execution
- `GET /api/v1/pipeline/logs` - Retrieve pipeline logs with filtering
- `GET /api/v1/pipeline/health` - System health check

### Predictions
- `GET /api/v1/predict` - Generate single prediction (traditional or deterministic)
- `GET /api/v1/predict-deterministic` - Generate deterministic prediction with scoring
- `GET /api/v1/predict-detailed` - Get prediction with detailed component analysis
- `GET /api/v1/compare-methods` - Compare traditional vs deterministic methods
- `GET /api/v1/prediction-history` - View recent prediction history

### Adaptive System
- `GET /api/v1/adaptive/analysis` - Get adaptive performance analysis
- `GET /api/v1/adaptive/weights` - View current adaptive weights
- `POST /api/v1/adaptive/optimize-weights` - Trigger weight optimization
- `GET /api/v1/adaptive/performance` - Get detailed performance analytics
- `POST /api/v1/adaptive/validate` - Run adaptive validation with learning

## Usage Examples

### Command Line Options

```bash
# Run full pipeline (default)
python main.py

# Run specific pipeline step
python main.py --step data          # Data update only
python main.py --step prediction    # Prediction generation only
python main.py --step validation    # Historical validation only

# Check pipeline status
python main.py --status

# Get help
python main.py --help
```

### Available Pipeline Steps
- `data` - Update database from lottery data source
- `adaptive` - Run adaptive analysis on recent performance
- `weights` - Optimize prediction weights based on performance
- `prediction` - Generate new predictions using deterministic system
- `validation` - Validate predictions against historical data
- `performance` - Analyze system performance metrics
- `reports` - Generate comprehensive execution reports

### Web Dashboard Features

1. **Pipeline Status Monitor**: Real-time view of pipeline execution status
2. **Manual Trigger**: Start pipeline execution on-demand
3. **Generated Plays**: View recent predictions with detailed scoring
4. **System Health**: Monitor CPU, memory, and disk usage
5. **Execution History**: Browse past pipeline runs and results
6. **Log Viewer**: Search and filter system logs

## Scheduling & Automation

SHIOL+ v5.0 includes automatic weekly scheduling:

- **Default Schedule**: Every Monday at 2:00 AM (configurable)
- **Timezone Support**: Configurable timezone (default: America/New_York)
- **Retry Logic**: Automatic retries with exponential backoff on failure
- **Manual Override**: Force execution even when scheduled runs are active

Configure scheduling in `config/config.ini`:

```ini
[pipeline]
weekly_execution_day = 0        # 0=Monday, 6=Sunday
execution_time = 02:00          # HH:MM format
timezone = America/New_York     # Timezone string
auto_execution_enabled = true   # Enable/disable automatic execution
```

## Prediction Validation

The system includes comprehensive validation against historical lottery results:

### Basic Validation
- Compares all stored predictions against actual lottery draws
- Calculates match rates for white balls and Powerball
- Determines prize categories according to official Powerball rules
- Generates detailed CSV reports with validation results

### Adaptive Validation
- Learns from validation results to improve future predictions
- Adjusts scoring weights based on historical performance
- Provides feedback to the adaptive learning system
- Tracks performance trends over time

### Validation Reports
Results are automatically saved to:
```
data/validations/validation_results_TIMESTAMP.csv
```

## Configuration

The `config/config.ini` file allows comprehensive system customization:

-   **`[paths]`**: Database, model, and log file locations
-   **`[model_params]`**: ML model hyperparameters
-   **`[pipeline]`**: Pipeline scheduling and execution settings
-   **`[temporal_analysis]`**: Temporal feature engineering parameters
-   **`[database]`**: Database connection and optimization settings

## System Requirements

-   Python 3.10+
-   Dependencies listed in `requirements.txt`
-   Minimum 2GB RAM for optimal performance
-   1GB free disk space for data and logs

## Installation

1. **Clone the repository**:
   ```bash
   git clone <REPOSITORY_URL>
   cd SHIOL-PLUS-V5
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the system**:
   ```bash
   python main.py
   ```

## Monitoring & Troubleshooting

### Log Files
- Main log: `logs/shiolplus.log`
- Automatic rotation (10MB files, 30-day retention)
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)

### Health Checks
The system provides comprehensive health monitoring:
- Database connectivity
- Model availability
- Configuration validation
- System resources (CPU, memory, disk)
- Pipeline orchestrator status

### Common Issues
1. **Database locked**: Ensure no other instances are running
2. **Model not found**: Run data update and training first
3. **Permission errors**: Check file system permissions
4. **Memory issues**: Increase available RAM or reduce batch sizes

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Prediction Accuracy**: Percentage of predictions with at least one match
- **Win Rate**: Percentage of predictions that win prizes
- **Score Distribution**: Analysis of prediction scoring patterns
- **Execution Time**: Pipeline step and total execution timing
- **Resource Usage**: CPU, memory, and disk utilization

## Credits and Authorship

-   **Creator**: Orlando Batista
-   **Version**: 5.0 (Phase 5 - Automated Pipeline System)
-   **Last Updated**: 2025

## License

Private use – All rights reserved.

---

## Version History

- **v5.0 (Phase 5)**: Automated pipeline system with web dashboard and comprehensive monitoring
- **v4.0 (Phase 4)**: Adaptive feedback system with weight optimization
- **v3.0 (Phase 3)**: Advanced analytics and performance tracking
- **v2.0 (Phase 2)**: Deterministic prediction system with validation
- **v1.0**: Basic ML prediction system with SQLite database
