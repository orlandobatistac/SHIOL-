# SHIOLPlus - Strategic Lottery Optimization System

`SHIOLPlus` is a Python console application designed for analyzing historical Powerball-type lottery data and generating an optimized set of combinations to play.

## Objective

The system uses a data-driven approach to:
1.  Analyze number frequencies from historical draws.
2.  Select a set of **base numbers** based on strategic balance criteria.
3.  Generate thousands of combinations from these base numbers.
4.  Apply a series of **heuristic filters** (sum, parity, etc.) to discard improbable combinations.
5.  Perform **backtesting** to evaluate the historical performance of the remaining combinations.
6.  Select and export the **top-performing combinations** to a CSV file, ready to be played.

## Project Structure

```
/SHIOLPlus
├── /data
│   └── NCELPowerball.csv   # Historical data source
├── /outputs                # Directory for generated results
├── /src
│   ├── __init__.py
│   ├── config.py           # Configurable system parameters
│   ├── data_processor.py   # Data loading and analysis module
│   ├── base_number_selector.py # Base number selection module
│   ├── combination_generator.py # Generation and filtering module
│   ├── performance_evaluator.py # Evaluation and backtesting module
│   └── output_exporter.py  # Results export module
├── main.py                 # Main application entry point
└── README.md
```

## Requirements

- Python 3.10+
- pandas
- numpy
- openpyxl

You can install the dependencies with:
```bash
pip install pandas numpy openpyxl
```

## How to Run

To execute the full analysis and play generation workflow, simply run the `main.py` script from the project root:

```bash
python main.py
```

The entire process may take a few seconds, depending on the size of the data file. Upon completion, a summary will be printed to the console, and a `.csv` file with the recommended plays will be generated in the `/outputs` directory.
## ⚠️ Disclaimer

This project was created for **educational and experimental purposes** only. It is not a commercial tool and does not, in any way, guarantee winnings in lotteries or games of chance.

SHIOLPlus uses public historical data and statistical strategies to demonstrate concepts in data analysis, computational simulation, and software engineering.

**The use of this system for actual betting is the sole responsibility of the user.** The author is not liable for any financial losses, misuse of the system, or violation of lottery policies in any jurisdiction.

Always consult local laws regarding participation in games of chance.