# SHIOL+ v2.0: AI-Powered Lottery Analysis System

An intelligent system designed to analyze historical data and predict lottery combinations using Machine Learning techniques.

## Project Summary

**SHIOL+ (Heuristic and Inferential Optimized Lottery System)** is a software tool that analyzes historical Powerball lottery draw data to identify statistical patterns. The system's main objective is to use an artificial intelligence model to predict the probability of each number appearing in future draws, thereby generating optimized plays.

The system is now powered by an **SQLite database**, ensuring data integrity and performance. It automatically downloads the latest draw data, keeping the model's knowledge base up-to-date for the most accurate predictions possible.

> **Important**: This tool was created for educational, research, and entertainment purposes. The lottery is a game of chance, and SHIOL+ **does not guarantee prizes or winnings**. Always play responsibly.

## Key Features

SHIOL+ v2.0 focuses on a clean and efficient pipeline with four key functionalities accessible from the command line:

*   **`update`**: Downloads the latest official Powerball data and intelligently updates the local SQLite database with only the new draws.
*   **`train`**: Trains the Machine Learning model (an `XGBoost` classifier) using the complete historical dataset stored in the SQLite database. The trained model is saved in the `models/` directory.
*   **`predict`**: Loads the trained model to predict the probabilities for each number (white balls and Powerball) for the next draw. Based on these probabilities, it generates a specified number of unique, weighted plays.
*   **`backtest`**: Performs a historical simulation to evaluate the performance of the model's strategy. It generates a set of plays and compares them against all past draws to calculate performance metrics like total cost, winnings, and ROI.

## Project Structure

The project is organized into the following directories:

-   `src/`: Contains all the application's Python source code.
    -   `__init__.py`: Makes the `src` directory a Python package.
    -   `api.py`: FastAPI application that serves the web interface and provides API endpoints.
    -   `cli.py`: The command-line interface (CLI) entry point.
    -   `database.py`: Manages all SQLite database interactions.
    -   `loader.py`: Handles loading data and orchestrates the database update process.
    -   `predictor.py`: Manages the training and prediction logic of the AI model.
    -   `intelligent_generator.py`: Includes feature engineering and weighted play generation.
    -   `evaluator.py`: Contains the logic for the `backtest` command.
-   `frontend/`: Contains the static files for the web interface.
    -   `index.html`: The main page of the web application.
    -   `css/styles.css`: Styles for the interface.
    -   `js/app.js`: JavaScript for interactive functionality.
-   `db/`: Stores the `shiolplus.db` SQLite database file (ignored by Git).
-   `data/`: Used for temporary storage of downloaded data before it's loaded into the database (ignored by Git).
-   `models/`: Saves the trained AI model artifacts (`shiolplus.pkl`).
-   `docs/`: Contains additional documentation for users and developers.
-   `config/`: Holds configuration files, like `config.ini`.
-   `logs/`: Stores application execution logs (ignored by Git).
-   `outputs/`: Default directory for generated files like prediction reports (ignored by Git).

## Requirements

-   Python 3.10+
-   Dependencies listed in `requirements.txt`.

To install all dependencies, it is recommended to first create a virtual environment and then run:

```bash
pip install -r requirements.txt
```

## Usage

SHIOL+ v2.0 offers two ways to interact with the system: a web interface and a command-line interface (CLI).

### 1. Web Interface (Recommended)

The easiest way to use SHIOL+ is through its integrated web interface.

**How to run it:**

1.  Make sure you have installed all the dependencies from `requirements.txt`.
2.  From the project's root directory, run the following command:
    
    ```bash
    uvicorn src.api:app --reload
    ```
3.  Open your web browser and go to `http://127.0.0.1:8000`.

The application will **automatically update the database** on the first run and then every 12 hours. It also trains the model if one doesn't exist, allowing you to generate plays interactively right away.

### 2. Command-Line Interface (CLI)

For more advanced users or for integration into automated workflows, the CLI provides direct access to the system's core functions.

All commands are executed through `src/cli.py`.

1.  **Update the database**:
    *   Downloads the latest data and updates your local database. Run this periodically.
    ```bash
    python src/cli.py update
    ```

2.  **Train the model**:
    *   This command must be run at least once before generating predictions. It uses the data in the local database.
    ```bash
    python src/cli.py train
    ```

3.  **Generate predictions (plays)**:
    *   Uses the trained model to generate a specific number of plays.
    ```bash
    python src/cli.py predict --count 10
    ```

4.  **Backtest the strategy**:
    *   Simulates the strategy's performance by generating plays and testing them against history.
    ```bash
    python src/cli.py backtest --count 20
    ```

---

## Configuration

The `config/config.ini` file allows for adjusting the system's behavior without modifying the code.

-   **`[paths]`**: Defines the paths to the database, model, and log files.
-   **`[model_params]`**: Hyperparameters for the model, such as `test_size`.
-   **`[temporal_analysis]`**: Parameters for temporal feature engineering.
-   **`[cli_defaults]`**: Default values for CLI arguments.

## Training and Evaluation

Training is performed with the `train` command, which processes historical data from the SQLite database and generates a `shiolplus.pkl` model file.

Evaluation is done with `backtest`, which provides a JSON report with key performance indicators.

## Credits and Authorship

-   **Creator**: Orlando Batista

## License

Private use – All rights reserved.
