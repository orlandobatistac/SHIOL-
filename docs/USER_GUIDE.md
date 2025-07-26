# SHIOL+ v5.0 User Guide

Welcome to SHIOL+ Phase 5! This comprehensive guide will help you install, configure, and use the automated pipeline system to analyze lottery patterns and generate predictions.

## 1. What is SHIOL+ v5.0?

SHIOL+ (Heuristic and Inferential Optimized Lottery System) is a comprehensive software tool that uses artificial intelligence to analyze historical lottery draw data (specifically, Powerball) with a fully automated pipeline system.

**What does it do?**

*   **Automated Pipeline**: Runs a complete 7-step pipeline with a single command (`python main.py`)
*   **Analyzes patterns**: The system "learns" from thousands of past draws to identify statistical patterns and trends
*   **Generates predictions**: Uses advanced deterministic scoring to create optimized lottery plays
*   **Adaptive learning**: Continuously improves by learning from historical performance
*   **Web dashboard**: Provides a modern interface for monitoring and controlling the system
*   **Scheduling**: Automatically runs weekly with configurable timing
*   **Comprehensive monitoring**: Tracks performance, system health, and execution history

In short, SHIOL+ v5.0 is a complete automated lottery analysis system with enterprise-grade monitoring and control capabilities.

> **Important**: This tool is for educational and entertainment purposes. The lottery is a game of chance, and SHIOL+ **does not guarantee prizes or winnings**. Always play responsibly.

## 2. Installation

To use SHIOL+ v5.0, you need to have Python installed on your computer. Follow these steps:

### Step 1: Clone the Project

First, download the project code. If you have `git` installed, you can use this command in your terminal:

```bash
git clone <REPOSITORY_URL>
cd SHIOL-PLUS-V5
```

If you don't have `git`, you can download the project as a ZIP file and unzip it.

### Step 2: Create a Virtual Environment (Recommended)

A virtual environment is an isolated space to install the project's dependencies without affecting other programs on your computer.

From the project folder (`SHIOL-PLUS-V5`), run:

```bash
python -m venv venv
```

Then, activate the environment:

*   **On Windows**:
    ```bash
    .\venv\Scripts\activate
    ```
*   **On macOS or Linux**:
    ```bash
    source venv/bin/activate
    ```

You will see `(venv)` at the beginning of your terminal line, indicating that the environment is active.

### Step 3: Install Dependencies

With the environment activated, install all necessary libraries with a single command:

```bash
pip install -r requirements.txt
```

And that's it! The system is now ready to be used.

## 3. Quick Start - Simple Execution

The easiest way to use SHIOL+ v5.0 is with the automated pipeline system:

### Single Command Execution

```bash
python main.py
```

This single command will automatically:

1. **Data Update**: Download and update the lottery database with the latest draws
2. **Adaptive Analysis**: Analyze recent system performance and identify improvement opportunities
3. **Weight Optimization**: Optimize prediction scoring weights based on historical results
4. **Prediction Generation**: Generate new lottery predictions using the deterministic system
5. **Historical Validation**: Validate predictions against actual lottery results
6. **Performance Analysis**: Calculate detailed performance metrics and trends
7. **Reports & Notifications**: Generate comprehensive execution reports

### What You'll See

When you run the pipeline, you'll see detailed progress information:

```
============================================================
STARTING SHIOL+ PHASE 5 FULL PIPELINE EXECUTION
============================================================
STEP 1/7: Data Update
✓ Data update completed successfully in 0:00:15
STEP 2/7: Adaptive Analysis
✓ Adaptive analysis completed successfully in 0:00:08
STEP 3/7: Weight Optimization
✓ Weight optimization completed successfully in 0:00:12
STEP 4/7: Prediction Generation
✓ Prediction generation completed successfully in 0:00:05
STEP 5/7: Historical Validation
✓ Historical validation completed successfully in 0:00:20
STEP 6/7: Performance Analysis
✓ Performance analysis completed successfully in 0:00:03
STEP 7/7: Notifications & Reports
✓ Notifications and reports completed successfully in 0:00:02
============================================================
SHIOL+ PHASE 5 PIPELINE EXECUTION COMPLETED SUCCESSFULLY
Total execution time: 0:01:05
============================================================
```

## 4. Web Dashboard

SHIOL+ v5.0 includes a modern web dashboard for monitoring and controlling the system.

### Starting the Web Dashboard

After running the pipeline at least once, start the web server:

```bash
uvicorn src.api:app --reload
```

Then open your web browser and go to `http://127.0.0.1:8000`.

### Dashboard Features

The web dashboard provides:

#### 4.1 Pipeline Status Monitor
- **Real-time status**: See current pipeline execution status
- **Execution history**: Browse past pipeline runs and their results
- **Next scheduled run**: View when the next automatic execution is scheduled
- **System health**: Monitor CPU, memory, and disk usage

#### 4.2 Manual Pipeline Control
- **Trigger execution**: Start pipeline execution on-demand
- **Step selection**: Run specific pipeline steps only
- **Force execution**: Override running pipelines if needed
- **Execution tracking**: Monitor progress of background executions

#### 4.3 Generated Plays Visualization
- **Recent predictions**: View the latest generated lottery plays
- **Scoring details**: See detailed scoring breakdown for each prediction
- **Prediction history**: Browse historical predictions with timestamps
- **Method comparison**: Compare different prediction methods

#### 4.4 System Analytics
- **Performance metrics**: View prediction accuracy and win rates
- **Validation results**: See how predictions performed against actual draws
- **Trend analysis**: Track system performance over time
- **Resource monitoring**: Monitor system resource usage

#### 4.5 Log Viewer
- **Real-time logs**: View system logs with filtering options
- **Log levels**: Filter by DEBUG, INFO, WARNING, ERROR levels
- **Date filtering**: View logs for specific date ranges
- **Search functionality**: Find specific log entries

## 5. Command Line Interface (CLI)

For advanced users or automation, the CLI provides direct access to all system functions.

### Pipeline Commands

#### Run Full Pipeline
```bash
python main.py
```

#### Run Specific Steps
```bash
python main.py --step data          # Update database only
python main.py --step adaptive      # Run adaptive analysis only
python main.py --step weights       # Optimize weights only
python main.py --step prediction    # Generate predictions only
python main.py --step validation    # Run validation only
python main.py --step performance   # Analyze performance only
python main.py --step reports       # Generate reports only
```

#### Check System Status
```bash
python main.py --status
```

This will show:
- Database status and record count
- Configuration status
- Recent execution history
- System health metrics
- Next scheduled execution time

#### Get Help
```bash
python main.py --help
```

### Legacy CLI Commands

For compatibility, the system still supports individual CLI commands:

#### Update Database
```bash
python src/cli.py update
```

#### Train Model
```bash
python src/cli.py train
```

#### Generate Predictions
```bash
python src/cli.py predict --count 10
python src/cli.py predict-deterministic
```

#### Validate Predictions
```bash
python src/cli.py validate
```

#### Compare Methods
```bash
python src/cli.py compare-methods
```

## 6. API Endpoints

SHIOL+ v5.0 provides comprehensive REST API endpoints for integration and automation.

### Pipeline Control Endpoints

#### Get Pipeline Status
```http
GET /api/v1/pipeline/status
```
Returns current pipeline status, execution history, and system health.

#### Trigger Pipeline Execution
```http
POST /api/v1/pipeline/trigger?num_predictions=1&force=false
```
Manually triggers pipeline execution with optional parameters.

#### Get Pipeline Logs
```http
GET /api/v1/pipeline/logs?level=INFO&limit=100
```
Retrieves pipeline logs with filtering and pagination.

#### System Health Check
```http
GET /api/v1/pipeline/health
```
Comprehensive system health check validating all components.

### Prediction Endpoints

#### Generate Single Prediction
```http
GET /api/v1/predict?deterministic=true
```
Generates a single prediction using traditional or deterministic method.

#### Generate Deterministic Prediction
```http
GET /api/v1/predict-deterministic
```
Generates a deterministic prediction with detailed scoring information.

#### Get Detailed Prediction
```http
GET /api/v1/predict-detailed?deterministic=true
```
Returns prediction with detailed component scores and analysis.

#### Compare Prediction Methods
```http
GET /api/v1/compare-methods
```
Compares traditional vs deterministic prediction methods side-by-side.

#### Get Prediction History
```http
GET /api/v1/prediction-history?limit=10
```
Returns recent prediction history with scoring details.

### Adaptive System Endpoints

#### Get Adaptive Analysis
```http
GET /api/v1/adaptive/analysis?days_back=30
```
Provides comprehensive adaptive analysis of system performance.

#### Get Current Weights
```http
GET /api/v1/adaptive/weights
```
Returns current adaptive weights configuration.

#### Optimize Weights
```http
POST /api/v1/adaptive/optimize-weights?algorithm=differential_evolution
```
Triggers weight optimization using specified algorithm.

#### Get Performance Analytics
```http
GET /api/v1/adaptive/performance?days_back=30
```
Returns detailed performance analytics for the adaptive system.

#### Run Adaptive Validation
```http
POST /api/v1/adaptive/validate?enable_learning=true
```
Runs adaptive validation with learning feedback.

## 7. Scheduling & Automation

SHIOL+ v5.0 includes automatic weekly scheduling with comprehensive configuration options.

### Default Schedule

By default, the system runs automatically:
- **Day**: Every Monday (configurable)
- **Time**: 2:00 AM (configurable)
- **Timezone**: America/New_York (configurable)

### Configuration

Edit `config/config.ini` to customize scheduling:

```ini
[pipeline]
# Weekly execution settings
weekly_execution_day = 0        # 0=Monday, 1=Tuesday, ..., 6=Sunday
execution_time = 02:00          # HH:MM format (24-hour)
timezone = America/New_York     # Timezone string
auto_execution_enabled = true   # Enable/disable automatic execution

# Retry settings
max_retry_attempts = 3          # Number of retry attempts on failure
retry_delay_minutes = 30        # Initial retry delay in minutes
retry_backoff_multiplier = 2.0  # Backoff multiplier for retries

# Timeout settings
pipeline_timeout_seconds = 3600 # Maximum pipeline execution time
```

### Manual Override

You can force execution even when scheduled runs are active:

```bash
python main.py  # Will run immediately regardless of schedule
```

Or via API:
```http
POST /api/v1/pipeline/trigger?force=true
```

## 8. Prediction Validation System

SHIOL+ v5.0 includes comprehensive validation against historical lottery results.

### Automatic Validation

The pipeline automatically validates predictions during step 5:
- Compares all stored predictions against actual lottery draws
- Calculates match rates for white balls and Powerball
- Determines prize categories according to official Powerball rules
- Generates detailed performance metrics

### Validation Reports

Results are automatically saved to:
```
data/validations/validation_results_TIMESTAMP.csv
```

The CSV file contains:
- `prediction_id`: Unique prediction identifier
- `prediction_date`: When the prediction was generated
- `white_balls`: Predicted white ball numbers
- `powerball`: Predicted Powerball number
- `draw_date`: Date of the actual lottery draw
- `actual_white_balls`: Actual winning white balls
- `actual_powerball`: Actual winning Powerball
- `white_matches`: Number of white ball matches
- `powerball_match`: Whether Powerball matched (True/False)
- `prize_category`: Prize category won (if any)
- `is_winner`: Whether the prediction won a prize

### Validation Results Interpretation

Example validation output:
```
=== VALIDATION COMPLETED ===
Predictions processed: 25
Win rate: 12.0%
Prize distribution:
  - Match 2+PB: 2 predictions
  - Match 1+PB: 1 prediction
  - Match 0+PB: 0 predictions
Non-winning: 22 predictions
```

### Adaptive Learning

The system uses validation results to:
- Optimize prediction scoring weights
- Improve future prediction accuracy
- Identify successful prediction patterns
- Adjust system parameters automatically

## 9. System Monitoring & Health

SHIOL+ v5.0 provides comprehensive system monitoring capabilities.

### Health Checks

The system continuously monitors:

#### Database Health
- Connection status
- Record count and data integrity
- Latest draw date
- Query performance

#### Model Health
- Model availability and loading status
- Prediction functionality
- Training data freshness
- Performance metrics

#### System Resources
- CPU usage percentage
- Memory usage and availability
- Disk space and I/O
- Network connectivity

#### Configuration Health
- Configuration file validity
- Required sections and parameters
- Path accessibility
- Permission checks

### Performance Metrics

The system tracks:

#### Prediction Performance
- **Accuracy Rate**: Percentage of predictions with at least one match
- **Win Rate**: Percentage of predictions that win prizes
- **Average Score**: Mean prediction scoring values
- **Score Distribution**: Analysis of scoring patterns

#### System Performance
- **Execution Time**: Pipeline step and total execution timing
- **Resource Usage**: CPU, memory, and disk utilization
- **Error Rates**: Frequency and types of errors
- **Uptime**: System availability metrics

#### Trend Analysis
- Performance trends over time
- Seasonal patterns in accuracy
- Resource usage patterns
- Error frequency trends

### Alerts and Notifications

The system provides notifications for:
- Pipeline execution completion
- System errors and failures
- Resource usage warnings
- Performance degradation alerts

## 10. Configuration Guide

SHIOL+ v5.0 is highly configurable through the `config/config.ini` file.

### Configuration Sections

#### [paths]
```ini
[paths]
database_file = db/shiolplus.db
model_file = models/shiolplus.pkl
log_file = logs/shiolplus.log
```

#### [pipeline]
```ini
[pipeline]
weekly_execution_day = 0
execution_time = 02:00
timezone = America/New_York
auto_execution_enabled = true
max_retry_attempts = 3
retry_delay_minutes = 30
retry_backoff_multiplier = 2.0
pipeline_timeout_seconds = 3600
```

#### [model_params]
```ini
[model_params]
test_size = 0.2
random_state = 42
n_estimators = 100
max_depth = 10
```

#### [database]
```ini
[database]
connection_timeout = 30
query_timeout = 60
backup_enabled = true
backup_frequency_hours = 24
```

#### [temporal_analysis]
```ini
[temporal_analysis]
lookback_days = 365
seasonal_analysis = true
trend_analysis = true
```

### Environment Variables

You can also configure the system using environment variables:

```bash
export SHIOL_CONFIG_PATH=/path/to/config.ini
export SHIOL_LOG_LEVEL=INFO
export SHIOL_DATABASE_PATH=/path/to/database.db
```

## 11. Troubleshooting

### Common Issues and Solutions

#### Issue: "Database is locked"
**Cause**: Another instance of SHIOL+ is running or didn't shut down properly.
**Solution**: 
1. Check for running processes: `ps aux | grep python`
2. Kill any SHIOL+ processes: `kill <process_id>`
3. Restart the system: `python main.py`

#### Issue: "Model not found"
**Cause**: The ML model hasn't been trained yet.
**Solution**: 
1. Run the full pipeline: `python main.py`
2. Or train manually: `python src/cli.py train`

#### Issue: "Permission denied" errors
**Cause**: Insufficient file system permissions.
**Solution**: 
1. Check file permissions: `ls -la`
2. Fix permissions: `chmod 755 <directory>` or `chmod 644 <file>`
3. Ensure write access to logs/, data/, and reports/ directories

#### Issue: "Memory error" during execution
**Cause**: Insufficient RAM for large datasets.
**Solution**: 
1. Close other applications to free memory
2. Reduce batch sizes in configuration
3. Consider upgrading system RAM

#### Issue: Web dashboard not accessible
**Cause**: Port conflict or server not started.
**Solution**: 
1. Check if port 8000 is in use: `netstat -an | grep 8000`
2. Use different port: `uvicorn src.api:app --port 8001`
3. Check firewall settings

#### Issue: Pipeline execution fails
**Cause**: Various reasons - check logs for details.
**Solution**: 
1. Check logs: `tail -f logs/shiolplus.log`
2. Run with verbose output: `python main.py --verbose`
3. Check system status: `python main.py --status`

### Log Analysis

#### Log Levels
- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about system operation
- **WARNING**: Warning messages about potential issues
- **ERROR**: Error messages about failures

#### Log Locations
- Main log: `logs/shiolplus.log`
- Rotated logs: `logs/shiolplus.log.1`, `logs/shiolplus.log.2`, etc.

#### Log Filtering
```bash
# View only errors
grep "ERROR" logs/shiolplus.log

# View recent logs
tail -f logs/shiolplus.log

# View logs for specific date
grep "2025-01-26" logs/shiolplus.log
```

### Performance Optimization

#### Database Optimization
1. Regular database maintenance
2. Index optimization
3. Query performance tuning
4. Regular backups

#### Memory Optimization
1. Adjust batch sizes in configuration
2. Enable garbage collection
3. Monitor memory usage patterns
4. Consider system RAM upgrade

#### CPU Optimization
1. Adjust thread pool sizes
2. Enable parallel processing where appropriate
3. Monitor CPU usage patterns
4. Consider system CPU upgrade

## 12. Glossary of Terms

*   **Pipeline**: The complete 7-step automated process that runs the entire SHIOL+ system
*   **Orchestrator**: The main component that coordinates and manages pipeline execution
*   **Deterministic Prediction**: A prediction method that always produces the same result for the same input data
*   **Adaptive Learning**: The system's ability to learn from historical performance and improve over time
*   **Weight Optimization**: The process of adjusting scoring component weights based on performance data
*   **Validation**: The process of comparing predictions against actual lottery results
*   **Scoring Components**: The four criteria used to evaluate predictions (Probability, Diversity, Historical, Risk-Adjusted)
*   **API Endpoint**: A URL that provides access to specific system functions via HTTP requests
*   **Health Check**: A system diagnostic that verifies all components are functioning properly
*   **Backtest**: A simulation that tests a strategy against historical data
*   **ROI (Return on Investment)**: A measure of profitability (usually negative for lottery systems)

## 13. Frequently Asked Questions (FAQ)

**Q: Will this program make me win the lottery?**
**A:** No. The lottery is fundamentally a game of chance. This program is a data analysis tool for educational and entertainment purposes and cannot guarantee any prizes.

**Q: How often should I run the pipeline?**
**A:** The system automatically runs weekly by default. You can also run it manually whenever you want updated predictions or analysis.

**Q: What's the difference between traditional and deterministic predictions?**
**A:** Traditional predictions use random sampling based on ML probabilities and produce different results each time. Deterministic predictions use a multi-criteria scoring system and always produce the same result for the same dataset.

**Q: How does the adaptive learning work?**
**A:** The system analyzes the performance of past predictions against actual lottery results and adjusts its scoring weights to improve future predictions.

**Q: Can I run multiple instances of SHIOL+ simultaneously?**
**A:** No, the system uses a SQLite database that doesn't support concurrent access. Only run one instance at a time.

**Q: How much disk space does the system need?**
**A:** Approximately 1GB for the database, logs, and generated files. The system automatically manages log rotation to prevent excessive disk usage.

**Q: Can I customize the prediction scoring weights?**
**A:** Yes, the system can automatically optimize weights based on performance, or you can manually adjust them in the configuration file.

**Q: What happens if the pipeline fails during execution?**
**A:** The system has built-in retry logic with exponential backoff. It will attempt to retry failed steps up to 3 times by default.

**Q: How do I interpret the validation results?**
**A:** Validation results show how your predictions would have performed against actual lottery draws. A win rate above 5% is considered good for lottery prediction systems.

**Q: Can I access the system remotely?**
**A:** Yes, you can configure the web dashboard to accept connections from other machines by modifying the uvicorn startup parameters.

## 14. Advanced Usage

### Custom Scheduling

You can create custom schedules beyond the weekly default:

```python
from src.scheduler import WeeklyScheduler
from main import PipelineOrchestrator

# Create custom scheduler
orchestrator = PipelineOrchestrator()
scheduler = WeeklyScheduler(pipeline_orchestrator=orchestrator)

# Schedule for multiple days
scheduler.reschedule(weekly_execution_day=1, execution_time="14:30")  # Tuesday 2:30 PM
```

### API Integration

Integrate SHIOL+ with other systems using the REST API:

```python
import requests

# Trigger pipeline execution
response = requests.post("http://localhost:8000/api/v1/pipeline/trigger")
execution_id = response.json()["execution_id"]

# Check status
status_response = requests.get("http://localhost:8000/api/v1/pipeline/status")
print(status_response.json())

# Get predictions
predictions = requests.get("http://localhost:8000/api/v1/predict-deterministic")
print(predictions.json())
```

### Custom Validation

Create custom validation logic:

```python
from src.basic_validator import BasicValidator
from src.database import get_all_draws

# Initialize validator
validator = BasicValidator()
historical_data = get_all_draws()

# Run custom validation
results = validator.validate_predictions(
    predictions_df=your_predictions,
    historical_data=historical_data,
    save_results=True
)
```

## 15. Responsible Gaming Warning

*   **Educational Purpose**: This software was created as an educational project to explore data science and artificial intelligence applications.
*   **Chance Dominates**: No system can reliably predict lottery numbers. The results are fundamentally random.
*   **Play Responsibly**: Never spend more money than you can afford to lose. The lottery is a form of entertainment, not an investment strategy.
*   **Seek Help if Needed**: If you feel that gambling is becoming a problem, seek help from organizations dedicated to responsible gaming such as:
    - National Council on Problem Gambling: https://www.ncpgambling.org/
    - Gamblers Anonymous: https://www.gamblersanonymous.org/
*   **Understand the Odds**: Powerball odds are approximately 1 in 292 million for the jackpot. No prediction system can meaningfully improve these odds.

---

*This guide covers the complete SHIOL+ v5.0 system. For technical support or questions, refer to the system logs and health checks for diagnostic information.*