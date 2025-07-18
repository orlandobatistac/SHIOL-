from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger


class Evaluator:
    """
    Evaluator for lottery plays and syndicate strategies.

    This class provides methods to evaluate lottery plays against winning numbers
    and calculate various performance metrics including:

    1. Basic evaluation metrics:
       - Number of white ball hits
       - Powerball hits
       - Prize tier determination

    2. Syndicate-specific metrics:
       - Combination Coverage: Measures how well a set of syndicate plays covers
         the possible number combinations, including pair coverage, number distribution,
         and uniqueness of combinations.

    3. Statistical metrics:
       - ROI Variance: Calculates the statistical variance of Return on Investment
         across multiple simulations to measure consistency/volatility of returns.

    4. Multi-objective optimization metrics:
       - Number Pattern Diversity: Measures diversity in terms of number patterns
       - Expected Value: Calculates expected value based on prize structure
       - Risk-Adjusted Return: Balances expected return with risk
       - Budget Efficiency: Measures how efficiently plays use the available budget
    """

    def __init__(self):
        logger.info("Evaluator initialized.")
        # Constants for Powerball game
        self.WHITE_BALL_RANGE = range(1, 70)  # 1-69
        self.POWERBALL_RANGE = range(1, 27)  # 1-26
        self.TOTAL_WHITE_BALL_PAIRS = len(list(combinations(self.WHITE_BALL_RANGE, 2)))

    def evaluate_plays(self, plays_df, winning_numbers, winning_pb):
        """
        Evaluates a DataFrame of plays against a set of winning numbers.

        :param plays_df: DataFrame with columns ['n1', 'n2', 'n3', 'n4', 'n5', 'pb']
        :param winning_numbers: A set of the 5 winning white ball numbers.
        :param winning_pb: The winning Powerball number.
        :return: DataFrame with added evaluation columns.
        """
        logger.info("Evaluating plays...")

        if plays_df.empty:
            logger.warning("Plays DataFrame is empty, nothing to evaluate.")
            return plays_df

        eval_results = []
        for _, play in plays_df.iterrows():
            play_numbers = {play["n1"], play["n2"], play["n3"], play["n4"], play["n5"]}

            hits_white = len(play_numbers.intersection(winning_numbers))
            hits_powerball = 1 if play["pb"] == winning_pb else 0

            prize_tier = self._get_prize_tier(hits_white, hits_powerball)

            eval_results.append(
                {
                    "hits_white": hits_white,
                    "hits_powerball": hits_powerball,
                    "prize_tier": prize_tier,
                }
            )

        results_df = pd.DataFrame(eval_results)

        # Merge results back into the original DataFrame
        plays_df_evaluated = plays_df.reset_index(drop=True).join(results_df)

        logger.info("Evaluation complete.")
        return plays_df_evaluated

    def run_backtest(self, plays_df: pd.DataFrame, historical_data: pd.DataFrame, ticket_cost: float = 2.0):
        """
        Runs a backtest of the given plays against historical draw data.

        Args:
            plays_df (pd.DataFrame): A DataFrame of plays to test.
            historical_data (pd.DataFrame): A DataFrame of past winning numbers.
            ticket_cost (float): The cost per ticket.

        Returns:
            dict: A summary report of the backtest results.
        """
        logger.info(f"Running backtest for {len(plays_df)} plays against {len(historical_data)} historical draws...")

        prize_structure = {
            "Jackpot": 20000000,  # Using a conservative floor for jackpot
            "Match 5": 1000000,
            "Match 4 + PB": 50000,
            "Match 4": 100,
            "Match 3 + PB": 100,
            "Match 3": 7,
            "Match 2 + PB": 7,
            "Match 1 + PB": 4,
            "Match PB": 4,
            "Non-winning": 0,
        }

        total_cost = len(plays_df) * len(historical_data) * ticket_cost
        total_winnings = 0
        win_counts = {tier: 0 for tier in prize_structure}

        white_ball_cols = [f"n{i}" for i in range(1, 6)]

        for _, draw in historical_data.iterrows():
            winning_numbers = {draw[col] for col in white_ball_cols}
            winning_pb = draw["pb"]

            evaluated_plays = self.evaluate_plays(plays_df, winning_numbers, winning_pb)
            
            for _, play_result in evaluated_plays.iterrows():
                tier = play_result["prize_tier"]
                if tier != "Non-winning":
                    win_counts[tier] += 1
                    total_winnings += prize_structure[tier]
        
        roi = ((total_winnings - total_cost) / total_cost) * 100 if total_cost > 0 else 0

        report = {
            "total_plays_simulated": len(plays_df) * len(historical_data),
            "total_cost": f"${total_cost:,.2f}",
            "total_winnings": f"${total_winnings:,.2f}",
            "roi_percent": f"{roi:.2f}%",
            "win_distribution": win_counts,
        }

        logger.info("Backtest complete.")
        logger.info(f"ROI: {roi:.2f}% | Total Winnings: ${total_winnings:,.2f} | Total Cost: ${total_cost:,.2f}")
        
        return report

    def calculate_combination_coverage(self, syndicate_plays_df):
        """
        Calculates the combination coverage metric for a set of syndicate plays.
        This metric measures how well the syndicate plays cover the possible number
        combinations.

        The metric consists of three components:
        1. Pair Coverage: Percentage of all possible white ball pairs covered
        2. Number Distribution: How evenly the numbers 1-69 are distributed
        3. Uniqueness Score: How diverse the combinations are

        :param syndicate_plays_df: DataFrame containing syndicate plays
        :return: Dictionary with combination coverage metrics
        """
        if syndicate_plays_df.empty:
            logger.warning(
                "Syndicate plays DataFrame is empty, "
                "cannot calculate combination coverage."
            )
            return {
                "pair_coverage_percent": 0,
                "number_distribution_score": 0,
                "uniqueness_score": 0,
                "overall_coverage_score": 0,
            }

        logger.info("Calculating combination coverage metrics for syndicate plays...")

        # 1. Calculate Pair Coverage
        all_pairs = set()
        for _, play in syndicate_plays_df.iterrows():
            white_balls = [play["n1"], play["n2"], play["n3"], play["n4"], play["n5"]]
            pairs = list(combinations(white_balls, 2))
            all_pairs.update(pairs)

        pair_coverage = len(all_pairs) / self.TOTAL_WHITE_BALL_PAIRS
        pair_coverage_percent = pair_coverage * 100

        # 2. Calculate Number Distribution
        number_counts = {}
        for i in self.WHITE_BALL_RANGE:
            number_counts[i] = 0

        for _, play in syndicate_plays_df.iterrows():
            for i in range(1, 6):
                num = play[f"n{i}"]
                number_counts[num] += 1

        # Calculate coefficient of variation (lower is better - more even distribution)
        counts = np.array(list(number_counts.values()))
        distribution_cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 1

        # Convert to a 0-100 score (100 is perfectly even distribution)
        number_distribution_score = max(0, 100 * (1 - distribution_cv))

        # 3. Calculate Uniqueness Score
        # Compare each play with every other play and count shared numbers
        similarity_matrix = np.zeros((len(syndicate_plays_df), len(syndicate_plays_df)))

        for i, (_, play1) in enumerate(syndicate_plays_df.iterrows()):
            set1 = {play1["n1"], play1["n2"], play1["n3"], play1["n4"], play1["n5"]}
            for j, (_, play2) in enumerate(syndicate_plays_df.iterrows()):
                if i != j:
                    set2 = {
                        play2["n1"],
                        play2["n2"],
                        play2["n3"],
                        play2["n4"],
                        play2["n5"],
                    }
                    similarity = len(set1.intersection(set2))
                    similarity_matrix[i, j] = similarity

        # Average similarity (lower is better - more unique plays)
        avg_similarity = np.mean(similarity_matrix) if similarity_matrix.size > 0 else 0

        # Convert to a 0-100 score (100 is completely unique plays)
        # For 5-number sets, max similarity is 5, min is 0
        uniqueness_score = max(0, 100 * (1 - avg_similarity / 5))

        # Calculate overall coverage score (weighted combination)
        overall_coverage_score = (
            0.5 * pair_coverage_percent
            + 0.25 * number_distribution_score
            + 0.25 * uniqueness_score
        )

        logger.info(
            f"Combination coverage calculation complete. "
            f"Overall score: {overall_coverage_score:.2f}"
        )

        return {
            "pair_coverage_percent": round(pair_coverage_percent, 2),
            "number_distribution_score": round(number_distribution_score, 2),
            "uniqueness_score": round(uniqueness_score, 2),
            "overall_coverage_score": round(overall_coverage_score, 2),
        }

    def calculate_roi_variance(self, plays_df, simulation_results):
        """
        Calculates the variance of Return on Investment (ROI) across multiple
        simulations.
        This metric helps users understand how consistent or volatile their expected
        returns might be.

        :param plays_df: DataFrame containing plays
        :param simulation_results: List of simulation results, where each result
            contains
            ROI for each play
        :return: DataFrame with ROI variance metrics added
        """
        if plays_df.empty or not simulation_results:
            logger.warning(
                "Plays DataFrame or simulation results are empty, "
                "cannot calculate ROI variance."
            )
            return plays_df

        logger.info("Calculating ROI variance metrics...")

        # Initialize arrays to store ROI values for each play across simulations
        num_plays = len(plays_df)
        num_simulations = len(simulation_results)
        roi_values = np.zeros((num_plays, num_simulations))

        # Collect ROI values from each simulation
        for sim_idx, sim_result in enumerate(simulation_results):
            for play_idx in range(num_plays):
                if play_idx < len(sim_result):
                    roi_values[play_idx, sim_idx] = sim_result[play_idx].get(
                        "roi_percent", 0
                    )

        # Calculate variance and other statistics for each play
        roi_variance = np.var(roi_values, axis=1)
        roi_std_dev = np.std(roi_values, axis=1)
        roi_min = np.min(roi_values, axis=1)
        roi_max = np.max(roi_values, axis=1)
        roi_range = roi_max - roi_min

        # Create a DataFrame with the results
        variance_results = pd.DataFrame(
            {
                "roi_variance": roi_variance,
                "roi_std_dev": roi_std_dev,
                "roi_min": roi_min,
                "roi_max": roi_max,
                "roi_range": roi_range,
            },
            index=plays_df.index,
        )

        # Round the values for better readability
        variance_results = variance_results.round(4)

        logger.info("ROI variance calculation complete.")

        # Return the original DataFrame with variance metrics added
        return plays_df.join(variance_results)

    def calculate_number_pattern_diversity(self, plays_df):
        """
        Calculates the diversity of number patterns in a set of plays.
        This metric considers various patterns like:
        - Even/odd distribution
        - High/low number distribution
        - Number spacing
        - Consecutive numbers

        :param plays_df: DataFrame containing plays
        :return: Series with diversity scores for each play and an overall diversity
            score
        """
        logger.info("Calculating number pattern diversity...")

        if plays_df.empty:
            logger.warning("Plays DataFrame is empty, cannot calculate diversity.")
            return pd.Series({"overall_diversity_score": 0})

        # Initialize metrics
        diversity_scores = []

        for _, play in plays_df.iterrows():
            white_balls = [play["n1"], play["n2"], play["n3"], play["n4"], play["n5"]]

            # 1. Even/Odd Balance (ideal is 2/3 or 3/2)
            even_count = sum(1 for num in white_balls if num % 2 == 0)
            odd_count = 5 - even_count
            even_odd_balance = (
                1 - abs(even_count - odd_count) / 5
            )  # 1 is perfect balance

            # 2. High/Low Balance (numbers above/below 35, ideal is 2/3 or 3/2)
            high_count = sum(1 for num in white_balls if num > 35)
            low_count = 5 - high_count
            high_low_balance = (
                1 - abs(high_count - low_count) / 5
            )  # 1 is perfect balance

            # 3. Number Spacing (std deviation of differences between consecutive nums)
            sorted_nums = sorted(white_balls)
            diffs = [sorted_nums[i + 1] - sorted_nums[i] for i in range(4)]
            spacing_std = np.std(diffs)
            # Normalize: lower std dev means more evenly spaced (better)
            spacing_score = max(
                0, 1 - spacing_std / 20
            )  # Assuming max std dev around 20

            # 4. Consecutive Numbers (penalize too many consecutive numbers)
            consecutive_count = sum(
                1 for i in range(4) if sorted_nums[i + 1] - sorted_nums[i] == 1
            )
            consecutive_score = (
                1 - consecutive_count / 4
            )  # Fewer consecutive numbers is better

            # 5. Sum Range (ideally in middle range, not too high or low)
            num_sum = sum(white_balls)
            # Ideal sum range is around 115-175 (middle of possible range 5-345)
            sum_score = (
                1 - abs(num_sum - 145) / 140
            )  # 145 is middle, 140 is approx half-range
            sum_score = max(0, min(1, sum_score))  # Clamp to [0,1]

            # Calculate overall diversity score (weighted average)
            diversity_score = (
                0.2 * even_odd_balance
                + 0.2 * high_low_balance
                + 0.2 * spacing_score
                + 0.2 * consecutive_score
                + 0.2 * sum_score
            ) * 100  # Scale to 0-100

            diversity_scores.append(
                {
                    "even_odd_balance": round(even_odd_balance * 100, 2),
                    "high_low_balance": round(high_low_balance * 100, 2),
                    "spacing_score": round(spacing_score * 100, 2),
                    "consecutive_score": round(consecutive_score * 100, 2),
                    "sum_score": round(sum_score * 100, 2),
                    "diversity_score": round(diversity_score, 2),
                }
            )

        # Create DataFrame with individual play scores
        diversity_df = pd.DataFrame(diversity_scores, index=plays_df.index)

        # Calculate overall diversity score for the set of plays
        overall_diversity = diversity_df["diversity_score"].mean()

        logger.info(
            f"Number pattern diversity calculation complete. "
            f"Overall score: {overall_diversity:.2f}"
        )

        # Add overall score to the results
        result = diversity_df.copy()
        result["overall_diversity_score"] = round(overall_diversity, 2)

        return result

    def calculate_expected_value(self, plays_df, prize_structure, ticket_cost=2.0):
        """
        Calculates the expected value of each play based on the prize structure.

        :param plays_df: DataFrame containing plays
        :param prize_structure: Dictionary mapping prize tiers to prize amounts
        :param ticket_cost: Cost of a single ticket
        :return: Series with expected value for each play
        """
        logger.info("Calculating expected value based on prize structure...")

        if plays_df.empty:
            logger.warning("Plays DataFrame is empty, cannot calculate expected value.")
            return pd.Series()

        # Default prize structure if not provided (approximate Powerball values)
        default_prize_structure = {
            "Jackpot": 100000000,  # $100M (average jackpot)
            "Match 5": 1000000,  # $1M
            "Match 4 + PB": 50000,  # $50K
            "Match 4": 100,  # $100
            "Match 3 + PB": 100,  # $100
            "Match 3": 7,  # $7
            "Match 2 + PB": 7,  # $7
            "Match 1 + PB": 4,  # $4
            "Match PB": 4,  # $4
            "Non-winning": 0,  # $0
        }

        prize_structure = prize_structure or default_prize_structure

        # Probabilities of each prize tier
        probabilities = {
            "Jackpot": 1 / 292201338,
            "Match 5": 1 / 11688053.52,
            "Match 4 + PB": 1 / 913129.18,
            "Match 4": 1 / 36525.17,
            "Match 3 + PB": 1 / 14494.11,
            "Match 3": 1 / 579.76,
            "Match 2 + PB": 1 / 701.33,
            "Match 1 + PB": 1 / 91.98,
            "Match PB": 1 / 38.32,
            "Non-winning": 1
            - (
                1 / 292201338
                + 1 / 11688053.52
                + 1 / 913129.18
                + 1 / 36525.17
                + 1 / 14494.11
                + 1 / 579.76
                + 1 / 701.33
                + 1 / 91.98
                + 1 / 38.32
            ),
        }

        # Calculate expected value for each play
        expected_values = []

        for _, play in plays_df.iterrows():
            # For simplicity, we use the same probability for all plays
            # In a more sophisticated model, we could adjust probabilities based on
            # play characteristics
            ev = (
                sum(
                    prize_structure[tier] * probabilities[tier]
                    for tier in prize_structure
                )
                - ticket_cost
            )
            expected_values.append(ev)

        # Create Series with expected values
        ev_series = pd.Series(
            expected_values, index=plays_df.index, name="expected_value"
        )

        # Normalize to 0-100 scale for consistency with other metrics
        min_ev = min(expected_values) if expected_values else -ticket_cost
        max_ev = max(expected_values) if expected_values else 0
        range_ev = max_ev - min_ev if max_ev > min_ev else 1

        normalized_ev = ((ev_series - min_ev) / range_ev * 100).round(2)

        logger.info("Expected value calculation complete.")

        return normalized_ev

    def calculate_risk_adjusted_return(self, plays_df, expected_values, roi_variance):
        """
        Calculates the risk-adjusted return for each play, balancing expected return
        with risk.
        Uses a Sharpe ratio-like metric: (expected return) / (std deviation of return)

        :param plays_df: DataFrame containing plays
        :param expected_values: Series with expected values for each play
        :param roi_variance: Series with ROI variance for each play
        :return: Series with risk-adjusted return scores
        """
        logger.info("Calculating risk-adjusted return...")

        if plays_df.empty or expected_values.empty or roi_variance.empty:
            logger.warning(
                "Input data is empty, cannot calculate risk-adjusted return."
            )
            return pd.Series()

        # Calculate risk-adjusted return (similar to Sharpe ratio)
        # Higher expected value and lower variance is better
        risk_adjusted_return = expected_values / np.sqrt(roi_variance)

        # Handle infinite values (when variance is 0)
        risk_adjusted_return = risk_adjusted_return.replace([np.inf, -np.inf], np.nan)

        # Fill NaN values with the expected value (assuming no risk)
        risk_adjusted_return = risk_adjusted_return.fillna(expected_values)

        # Normalize to 0-100 scale
        min_rar = risk_adjusted_return.min()
        max_rar = risk_adjusted_return.max()
        range_rar = max_rar - min_rar if max_rar > min_rar else 1

        normalized_rar = ((risk_adjusted_return - min_rar) / range_rar * 100).round(2)

        logger.info("Risk-adjusted return calculation complete.")

        return normalized_rar

    def calculate_budget_efficiency(self, plays_df, budget, ticket_cost=2.0):
        """
        Calculates how efficiently the plays use the available budget.
        Considers factors like:
        - Coverage per dollar spent
        - Diversity per dollar spent
        - Expected value per dollar spent

        :param plays_df: DataFrame containing plays
        :param budget: Total budget available
        :param ticket_cost: Cost of a single ticket
        :return: Dictionary with budget efficiency metrics
        """
        logger.info("Calculating budget efficiency...")

        if plays_df.empty:
            logger.warning(
                "Plays DataFrame is empty, cannot calculate budget efficiency."
            )
            return {"budget_efficiency_score": 0}

        # Number of plays
        num_plays = len(plays_df)

        # Total cost
        total_cost = num_plays * ticket_cost

        # Budget utilization (percentage of budget used)
        budget_utilization = min(1.0, total_cost / budget) if budget > 0 else 0

        # Calculate combination coverage
        coverage_metrics = self.calculate_combination_coverage(plays_df)

        # Coverage per dollar spent
        coverage_per_dollar = (
            coverage_metrics["overall_coverage_score"] / total_cost
            if total_cost > 0
            else 0
        )

        # Calculate diversity
        diversity_metrics = self.calculate_number_pattern_diversity(plays_df)

        # Diversity per dollar spent
        diversity_per_dollar = (
            diversity_metrics["overall_diversity_score"].iloc[0] / total_cost
            if total_cost > 0
            else 0
        )

        # Calculate overall budget efficiency score (0-100)
        budget_efficiency_score = (
            0.4 * (budget_utilization * 100)  # 40% weight on budget utilization
            + 0.3
            * (
                coverage_per_dollar / 0.05 * 100
            )  # 30% weight on coverage per dollar (normalized)
            + 0.3
            * (
                diversity_per_dollar / 0.05 * 100
            )  # 30% weight on diversity per dollar (normalized)
        )

        # Clamp to 0-100 range
        budget_efficiency_score = max(0, min(100, budget_efficiency_score))

        logger.info(
            f"Budget efficiency calculation complete. "
            f"Score: {budget_efficiency_score:.2f}"
        )

        return {
            "budget_utilization": round(budget_utilization * 100, 2),
            "coverage_per_dollar": round(coverage_per_dollar, 4),
            "diversity_per_dollar": round(diversity_per_dollar, 4),
            "budget_efficiency_score": round(budget_efficiency_score, 2),
        }

    def get_objective_functions(self):
        """
        Returns a dictionary of objective functions that can be used for multi-objective
        optimization.
        Each function takes a DataFrame of plays and returns a Series of objective
        values.

        :return: Dictionary mapping objective names to functions
        """
        logger.info("Creating objective functions for multi-objective optimization...")

        objective_functions = {
            "likeliness_score": lambda df: pd.Series(
                [1.0] * len(df)
            ),  # Placeholder, will be calculated by ML model
            "diversity_score": lambda df: self.calculate_number_pattern_diversity(df)[
                "diversity_score"
            ],
            "coverage_score": lambda df: pd.Series(
                [self.calculate_combination_coverage(df)["overall_coverage_score"]]
                * len(df)
            ),
            "expected_value": lambda df: self.calculate_expected_value(df, None),
            "budget_efficiency": lambda df: pd.Series(
                [self.calculate_budget_efficiency(df, 100)["budget_efficiency_score"]]
                * len(df)
            ),
        }

        logger.info(f"Created {len(objective_functions)} objective functions.")

        return objective_functions

    def _get_prize_tier(self, hits_white, hits_powerball):
        """
        Determines the prize tier based on the number of hits.
        (Based on Powerball prize rules)
        """
        if hits_white == 5 and hits_powerball == 1:
            return "Jackpot"
        if hits_white == 5 and hits_powerball == 0:
            return "Match 5"
        if hits_white == 4 and hits_powerball == 1:
            return "Match 4 + PB"
        if hits_white == 4 and hits_powerball == 0:
            return "Match 4"
        if hits_white == 3 and hits_powerball == 1:
            return "Match 3 + PB"
        if hits_white == 3 and hits_powerball == 0:
            return "Match 3"
        if hits_white == 2 and hits_powerball == 1:
            return "Match 2 + PB"
        if hits_white == 1 and hits_powerball == 1:
            return "Match 1 + PB"
        if hits_white == 0 and hits_powerball == 1:
            return "Match PB"
        return "Non-winning"
