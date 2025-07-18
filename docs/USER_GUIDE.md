# SHIOL+ v2.0 User Guide

Welcome to SHIOL+! This guide will help you install and use the system to analyze lottery patterns and generate plays.

## 1. What is SHIOL+ v2.0?

SHIOL+ (Heuristic and Inferential Optimized Lottery System) is a software tool that uses artificial intelligence to analyze historical lottery draw data (specifically, Powerball).

**What does it do?**

*   **Analyzes patterns**: The system "learns" from thousands of past draws to identify statistical patterns and trends that humans cannot see.
*   **Generates plays**: Based on its analysis, the system suggests new plays that align with the patterns it has identified as relevant.
*   **Tests strategies**: It allows you to simulate how well a playing strategy would have performed if you had used it in the past.

In short, SHIOL+ is an experimental tool for exploring lottery data in an intelligent way.

> **Important**: This tool is for educational and entertainment purposes. The lottery is a game of chance, and SHIOL+ **does not guarantee prizes or winnings**. Always play responsibly.

## 2. Installation

To use SHIOL+, you need to have Python installed on your computer. Follow these steps:

### Step 1: Clone the Project

First, download the project code. If you have `git` installed, you can use this command in your terminal:

```bash
git clone <REPOSITORY_URL>
cd SHIOL-PLUS-V2
```

If you don't have `git`, you can download the project as a ZIP file and unzip it.

### Step 2: Create a Virtual Environment (Recommended)

A virtual environment is an isolated space to install the project's dependencies without affecting other programs on your computer.

From the project folder (`SHIOL-PLUS-V2`), run:

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

## 3. How to Use the System

SHIOL+ is controlled through simple commands in the terminal from the project folder. There are three main commands.

### Command 1: `train` (Train the Model)

This is the first command you should run. It tells the system to analyze all historical draw data and "train" its artificial intelligence brain.

**When to use it?**
*   The first time you use the program.
*   Occasionally, after the historical data has been updated, so the model can learn from the most recent draws.

**Command:**
```bash
python src/cli.py train
```
You will see a log of the training process on the screen. Once it's finished, the intelligent model is ready and saved.

### Command 2: `predict` (Generate New Plays)

Once the model is trained, you can ask it to generate new plays.

**When to use it?**
*   When you want to get a new set of numbers to play, based on the system's analysis.

**Command:**
You can specify how many plays you want to generate with the `--count` option. If you don't use it, it will generate 5 by default.

```bash
# Generate 15 new plays
python src/cli.py predict --count 15
```
The system will show you a table with the generated plays.

### Command 3: `backtest` (Test a Strategy)

This command is for simulating how the system's strategy would have performed in the past. It generates a set of plays and compares them against all historical draws to see how many prizes they would have won.

**When to use it?**
*   When you want to evaluate the theoretical performance of the model. It's a way to "verify" if the system's logic has any historical merit.

**Command:**
```bash
# Generate 20 plays and test them against the history
python src/cli.py backtest --count 20
```
The result will be a detailed report that includes:
*   `total_cost`: How much it would have cost to play those combinations in all draws.
*   `total_winnings`: How much money would have been won.
*   `roi_percent`: The Return on Investment (it will usually be negative).
*   `win_distribution`: A breakdown of how many prizes of each category would have been won.

## 4. Glossary of Simple Terms

*   **AI (Artificial Intelligence)**: The "brain" of the system. A program that can learn from data.
*   **Model**: The file that contains all the knowledge the AI has learned after being trained.
*   **Train**: The process of "teaching" the AI, where it analyzes historical data to find patterns.
*   **Backtest**: A simulation that tests an investment or gaming strategy against past data to see how it would have performed.
*   **ROI (Return on Investment)**: A measure that tells you if you won or lost money compared to what you spent. A negative ROI means a loss.

## 5. Frequently Asked Questions (FAQ)

**Q: Will this program make me win the lottery?**
**A:** No. The lottery is fundamentally a game of chance. This program is a data analysis tool for educational and entertainment purposes and cannot guarantee any prizes.

**Q: What do I need to get started?**
**A:** You just need to have Python installed on your computer and follow the instructions in the "Installation" section.

**Q: Do I have to run `train` every time I want to generate plays?**
**A:** No. You only need to train the model once. After that, you can use `predict` and `backtest` as many times as you want. You should only retrain the model if the historical data is updated with new draws.

**Q: Why does the `backtest` almost always show a negative ROI?**
**A:** This is normal and expected. It reflects the reality of the lottery: it is very difficult to win, and in the long run, most players lose money. A negative ROI shows that even with intelligent analysis, the odds are still against the player.

## 6. Responsible Gaming Warning

*   **Educational Purpose**: This software was created as an educational project to explore data science and artificial intelligence.
*   **Chance Dominates**: No system can reliably predict lottery numbers. The results are random.
*   **Play Responsibly**: Never spend more money than you can afford to lose. The lottery is a form of entertainment, not a way to make money.
*   **Seek Help if Needed**: If you feel that gambling is becoming a problem, seek help from organizations dedicated to responsible gaming.