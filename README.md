# SHIOLPlus v1.4: Intelligent Lottery Analysis System

## Project Description

SHIOLPlus (System of Heuristic and Inferential Optimized Lottery) is a Python-based application designed to analyze Powerball lottery data. Version 1.4 features a complete, end-to-end pipeline that integrates statistical analysis, machine learning, and evolutionary algorithms to generate and optimize lottery plays. The system is accessible through a user-friendly graphical interface (GUI) or a command-line interface (CLI).

### Methodology

The core methodology of SHIOLPlus is a multi-stage intelligence pipeline:
1.  **Feature Engineering:** The system ingests the entire history of Powerball draws and calculates a rich set of features for each draw, including number parity, sum, spread, and advanced recency (delay) metrics.
2.  **ML-Powered Ranking:** An XGBoost classification model is trained on these features to learn the statistical "shape" of high-quality winning combinations.
3.  **Weighted Candidate Generation:** Instead of pure random generation, the system creates a large pool of candidate plays weighted by the historical frequency of each number.
4.  **Evolutionary Optimization:** The top-ranked plays from the ML model are used as the initial population for a Genetic Algorithm. This algorithm "evolves" the plays over dozens of generations, using crossover and mutation to search for even more optimal combinations.
5.  **Monte Carlo Simulation:** The final, evolved plays are run through a Monte Carlo simulation of 100,000+ synthetic draws to estimate their long-term performance, calculating metrics like expected Return on Investment (ROI).

### Disclaimer

**This project is for educational, research, and entertainment purposes only. It is an exploration of data analysis and machine learning concepts. It does not and cannot guarantee any winnings. The lottery is a game of chance, and you should always play responsibly.**

---

## Table of Contents
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- **Graphical User Interface (GUI):** A simple, dark-themed Tkinter interface allows for easy operation without using the command line.
- **End-to-End Automated Pipeline:** A single click or command runs the entire workflow from data loading to final play generation.
- **Advanced ML and AI Integration:** Combines statistical analysis, XGBoost classification, a Genetic Algorithm optimizer, and Monte Carlo simulation.
- **Intelligent Candidate Generation:** Uses historical number frequencies to generate statistically relevant candidate plays.
- **Data-Driven Updates:** The system can ingest new draw results from a CSV file to automatically update its historical data and retrain its models.
- **Detailed Logging:** All operations are logged to `logs/shiolplus.log` for easy debugging and tracking.

---

## Tech Stack

- **Language:** Python 3.10+
- **GUI:** Tkinter
- **Core Libraries:**
    - `pandas` & `numpy`
    - `scikit-learn` & `xgboost`
    - `sqlalchemy`
    - `joblib`
    - `loguru`

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd SHIOLPlus
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run

The system can be operated via the GUI (recommended for most users) or the CLI.

### Running the GUI

To launch the graphical user interface, run:
```bash
python main.py gui
```
The GUI provides buttons to "Generate Plays", "Update from CSV", and a link to this "User Manual".

### Using the CLI

-   **Generate New Plays:**
    ```bash
    python main.py generate
    ```
-   **Update with New Results:**
    ```bash
    python main.py update --file <path/to/new_results.csv>
    ```

---

## Project Structure

-   `main.py`: The main CLI entry point.
-   `src/`:
    -   `gui.py`: Contains all code for the Tkinter graphical user interface.
    -   `pipeline.py`: Holds the core orchestration logic for the `generate` and `update` commands.
    -   `data_loader.py`: Handles loading and cleaning of historical data.
    -   `feature_engineer.py`: Extracts statistical and heuristic features.
    -   `model_trainer.py`: Manages training and versioning of the ML model.
    -   `play_generator.py`: Implements the candidate generation and ranking logic.
    -   `evolutionary_engine.py`: Contains the Genetic Algorithm for play optimization.
    -   `simulation_engine.py`: Runs the Monte Carlo simulation to evaluate plays.
    -   `evaluator.py`: Contains logic to evaluate plays against winning numbers.
-   `config/`: Contains the `config.ini` file for all settings.
-   `data/`: Contains the source `NCELPowerball.csv` file.
-   `models/`: Stores the serialized model artifact.
-   `outputs/`: Default location for generated play CSV files.
-   `logs/`: Contains execution logs.

---

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes.

---

## License

This project is distributed under the MIT License.