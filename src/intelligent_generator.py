import configparser
import os
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from scipy.spatial.distance import euclidean

class FeatureEngineer:
    def __init__(self, historical_data):
        self.data = historical_data.copy()
        self.row_count = len(self.data)
        logger.info(
            f"FeatureEngineer initialized with {self.row_count} rows of historical data."
        )
        self.expected_columns = ["draw_date", "n1", "n2", "n3", "n4", "n5", "pb"]
        self._validate_required_columns()
        self._load_temporal_config()

    def _load_temporal_config(self):
        """
        Loads temporal analysis configuration parameters from config.ini
        """
        try:
            config = configparser.ConfigParser()
            config.read(os.path.join("config", "config.ini"))
            self.time_decay_function = config.get(
                "temporal_analysis", "time_decay_function", fallback="exponential"
            )
            self.time_decay_rate = config.getfloat(
                "temporal_analysis", "time_decay_rate", fallback=0.05
            )
            self.min_weight_percent = (
                config.getfloat("temporal_analysis", "min_weight_percent", fallback=10)
                / 100
            )
            self.moving_window_size = config.getint(
                "temporal_analysis", "moving_window_size", fallback=20
            )
            self.num_windows = config.getint(
                "temporal_analysis", "num_windows", fallback=5
            )
            self.seasonality_period = config.getint(
                "temporal_analysis", "seasonality_period", fallback=30
            )
            self.seasonality_threshold = config.getfloat(
                "temporal_analysis", "seasonality_threshold", fallback=0.6
            )
            logger.info("Temporal analysis configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading temporal analysis configuration: {e}")
            self.time_decay_function = "exponential"
            self.time_decay_rate = 0.05
            self.min_weight_percent = 0.1
            self.moving_window_size = 20
            self.num_windows = 5
            self.seasonality_period = 30
            self.seasonality_threshold = 0.6
            logger.warning("Using default temporal analysis parameters")

    def _validate_required_columns(self):
        """
        Validates that the required columns exist in the dataset.
        """
        try:
            missing_columns = [
                col for col in self.expected_columns if col not in self.data.columns
            ]
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                for col in missing_columns:
                    if col in ["n1", "n2", "n3", "n4", "n5", "pb"]:
                        self.data[col] = np.nan
                        logger.info(f"Added missing column '{col}' with NaN values")
            number_cols = [
                col
                for col in ["n1", "n2", "n3", "n4", "n5", "pb"]
                if col in self.data.columns
            ]
            for col in number_cols:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    logger.warning(
                        f"Column '{col}' is not numeric. Attempting to convert."
                    )
                    try:
                        self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                    except Exception as e:
                        logger.error(
                            f"Failed to convert column '{col}' to numeric: {e}"
                        )
            logger.info("Column validation complete")
        except Exception as e:
            logger.error(f"Error during column validation: {e}")

    def _validate_date_column(self):
        """
        Centralized validation for the draw_date column.
        """
        try:
            if "draw_date" not in self.data.columns:
                date_column_alternatives = [
                    "date",
                    "draw date",
                    "drawdate",
                    "datetime",
                    "timestamp",
                ]
                found_column = None
                for alt_col in date_column_alternatives:
                    if alt_col in self.data.columns:
                        found_column = alt_col
                        logger.info(
                            f"Found alternative date column '{alt_col}'. "
                            f"Renaming to 'draw_date'."
                        )
                        self.data["draw_date"] = self.data[alt_col]
                        break
                if found_column is None:
                    logger.warning(
                        "'draw_date' column not found. Creating synthetic dates."
                    )
                    self.data["draw_date"] = pd.to_datetime("today") - pd.to_timedelta(
                        self.data.index, unit="D"
                    )
                    return False
            if not pd.api.types.is_datetime64_dtype(self.data["draw_date"]):
                logger.info("Converting draw_date column to datetime format.")
                try:
                    self.data["draw_date"] = pd.to_datetime(
                        self.data["draw_date"], errors="coerce"
                    )
                    nat_count = self.data["draw_date"].isna().sum()
                    if nat_count > 0:
                        logger.warning(
                            f"Found {nat_count} invalid date values that were "
                            f"converted to NaT."
                        )
                except Exception as e:
                    logger.error(f"Error converting draw_date to datetime: {e}")
                    self.data["draw_date"] = pd.to_datetime("today") - pd.to_timedelta(
                        self.data.index, unit="D"
                    )
                    return False
            today = pd.to_datetime("today")
            future_dates = self.data[self.data["draw_date"] > today]
            if len(future_dates) > 0:
                logger.warning(
                    f"Found {len(future_dates)} future dates. Setting to today's date."
                )
                self.data.loc[self.data["draw_date"] > today, "draw_date"] = today
            missing_dates = self.data["draw_date"].isna().sum()
            if missing_dates > 0:
                logger.warning(
                    f"Found {missing_dates} missing dates. "
                    f"Filling with synthetic dates."
                )
                missing_mask = self.data["draw_date"].isna()
                self.data.loc[missing_mask, "draw_date"] = pd.to_datetime(
                    "today"
                ) - pd.to_timedelta(self.data.index[missing_mask], unit="D")
            self.data.sort_values(by="draw_date", inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            return True
        except Exception as e:
            logger.error(f"Error during date validation: {e}")
            self.data["draw_date"] = pd.to_datetime("today") - pd.to_timedelta(
                self.data.index, unit="D"
            )
            return False

    def engineer_features(self, use_temporal_analysis=True):
        self._standardize_column_names()
        try:
            initial_row_count = len(self.data)
            logger.info(
                f"Starting feature engineering with {initial_row_count} rows"
                f"{' with' if use_temporal_analysis else ' without'} temporal analysis"
                f"..."
            )
            self._validate_required_columns()
            white_ball_cols = ["n1", "n2", "n3", "n4", "n5"]
            missing_cols = [
                col for col in white_ball_cols if col not in self.data.columns
            ]
            if missing_cols:
                logger.warning(
                    f"Missing white ball columns: {missing_cols}. "
                    f"Some features may not be calculated correctly."
                )
            self._calculate_basic_features(white_ball_cols)
            if use_temporal_analysis:
                self._calculate_temporal_features()
            self._add_prize_tier()
            self.calculate_euclidean_distance_features()
            final_row_count = len(self.data)
            logger.info(
                f"Feature engineering complete with {final_row_count} rows"
                f"{' with' if use_temporal_analysis else ' without'} temporal features"
                "."
            )
            if final_row_count != initial_row_count:
                logger.warning(
                    f"Row count changed during feature engineering: "
                    f"{initial_row_count} -> {final_row_count}"
                )
            return self.data
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            logger.info("Returning partially processed data due to errors.")
            return self.data

    def _calculate_basic_features(self, white_ball_cols):
        self._calculate_parity_features(white_ball_cols)
        self._calculate_sum_features(white_ball_cols)
        self._calculate_spread_features(white_ball_cols)
        self._calculate_consecutive_features(white_ball_cols)
        self._calculate_low_high_balance(white_ball_cols)

    def _calculate_parity_features(self, white_ball_cols):
        try:
            even_mask = (self.data[white_ball_cols] % 2 == 0) & self.data[
                white_ball_cols
            ].notna()
            self.data["even_count"] = even_mask.sum(axis=1)
            valid_count = self.data[white_ball_cols].notna().sum(axis=1)
            self.data["odd_count"] = valid_count - self.data["even_count"]
        except Exception as e:
            logger.error(f"Error calculating parity features: {e}")
            self.data["even_count"] = np.nan
            self.data["odd_count"] = np.nan

    def _calculate_sum_features(self, white_ball_cols):
        try:
            self.data["sum"] = self.data[white_ball_cols].sum(axis=1)
        except Exception as e:
            logger.error(f"Error calculating sum features: {e}")
            self.data["sum"] = np.nan

    def _calculate_spread_features(self, white_ball_cols):
        try:
            self.data["spread"] = self.data[white_ball_cols].max(axis=1) - self.data[
                white_ball_cols
            ].min(axis=1)
        except Exception as e:
            logger.error(f"Error calculating spread features: {e}")
            self.data["spread"] = np.nan

    def _calculate_consecutive_features(self, white_ball_cols):
        try:
            def count_consecutive(row):
                valid_numbers = [x for x in row if pd.notna(x)]
                if not valid_numbers:
                    return 0
                numbers = sorted(valid_numbers)
                consecutive_count = 0
                for i in range(len(numbers) - 1):
                    if numbers[i + 1] - numbers[i] == 1:
                        consecutive_count += 1
                return consecutive_count
            self.data["consecutive_count"] = self.data[white_ball_cols].apply(
                count_consecutive, axis=1
            )
        except Exception as e:
            logger.error(f"Error calculating consecutive number features: {e}")
            self.data["consecutive_count"] = np.nan

    def _calculate_low_high_balance(self, white_ball_cols):
        try:
            def low_high_balance(row):
                valid_numbers = [x for x in row if pd.notna(x)]
                if not valid_numbers:
                    return "0L-0H"
                low_count = sum(1 for x in valid_numbers if x <= 35)
                high_count = len(valid_numbers) - low_count
                return f"{low_count}L-{high_count}H"
            self.data["low_high_balance"] = self.data[white_ball_cols].apply(
                low_high_balance, axis=1
            )
        except Exception as e:
            logger.error(f"Error calculating low/high balance features: {e}")
            self.data["low_high_balance"] = "0L-0H"

    def _calculate_temporal_features(self):
        date_valid = self._validate_date_column()
        if not date_valid:
            logger.warning(
                "Using synthetic dates for temporal analysis. "
                "Results may be less accurate."
            )
        self._calculate_recency()
        self._calculate_time_weights()
        self._detect_trends()
        self._detect_seasonal_patterns()

    def _standardize_column_names(self):
        try:
            column_patterns = {
                "date": "draw_date", "draw date": "draw_date", "drawdate": "draw_date",
                "datetime": "draw_date", "timestamp": "draw_date", "white_ball_1": "n1",
                "white_ball_2": "n2", "white_ball_3": "n3", "white_ball_4": "n4",
                "white_ball_5": "n5", "number 1": "n1", "number 2": "n2",
                "number 3": "n3", "number 4": "n4", "number 5": "n5",
                "ball1": "n1", "ball2": "n2", "ball3": "n3", "ball4": "n4",
                "ball5": "n5", "powerball": "pb", "power_ball": "pb",
                "power ball": "pb", "ball6": "pb",
            }
            rename_map = {}
            for col in self.data.columns:
                col_lower = col.lower()
                if col_lower in column_patterns:
                    rename_map[col] = column_patterns[col_lower]
            if rename_map:
                logger.info(f"Standardizing column names: {rename_map}")
                self.data.rename(columns=rename_map, inplace=True)
        except Exception as e:
            logger.error(f"Error during column name standardization: {e}")

    def _add_prize_tier(self):
        try:
            required_cols = ["sum", "even_count", "consecutive_count"]
            missing_cols = [
                col for col in required_cols if col not in self.data.columns
            ]
            if missing_cols:
                logger.warning(
                    f"Missing columns for prize tier classification: {missing_cols}. "
                    f"Using default tier."
                )
                self.data["prize_tier"] = "LowTier"
                return
            for col in required_cols:
                if self.data[col].isna().any():
                    if col == "sum": self.data[col].fillna(0, inplace=True)
                    elif col == "even_count": self.data[col].fillna(0, inplace=True)
                    elif col == "consecutive_count": self.data[col].fillna(0, inplace=True)
            conditions = [
                (self.data["sum"].between(120, 240))
                & (self.data["even_count"].isin([2, 3]))
                & (self.data["consecutive_count"] <= 1),
                (self.data["sum"].between(100, 260))
                & (self.data["even_count"].isin([1, 2, 3, 4])),
            ]
            choices = ["TopTier", "MidTier"]
            self.data["prize_tier"] = np.select(conditions, choices, default="LowTier")
        except Exception as e:
            logger.error(f"Error during prize tier classification: {e}")
            self.data["prize_tier"] = "LowTier"

    def _calculate_recency(self):
        try:
            if len(self.data) < 10: # Corresponds to min_draws_for_recency
                self.data["avg_delay"] = 0
                self.data["max_delay"] = 0
                self.data["min_delay"] = 0
                return
            self.data.sort_values(by="draw_date", inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            last_seen = {}
            delays = []
            white_ball_cols = ["n1", "n2", "n3", "n4", "n5"]
            for index, row in self.data.iterrows():
                current_delays = []
                numbers_in_draw = []
                for col in white_ball_cols:
                    if col in row:
                        num = row.get(col)
                        if pd.notna(num):
                            numbers_in_draw.append(num)
                for num in numbers_in_draw:
                    delay = index - last_seen.get(num, index)
                    current_delays.append(delay)
                    last_seen[num] = index
                if not current_delays:
                    current_delays = [0] * 5
                while len(current_delays) < 5:
                    current_delays.append(0)
                if len(current_delays) > 5:
                    current_delays = current_delays[:5]
                delays.append(current_delays)
            delay_df = pd.DataFrame(delays, columns=[f"delay_n{i+1}" for i in range(5)])
            self.data["avg_delay"] = delay_df.mean(axis=1)
            self.data["max_delay"] = delay_df.max(axis=1)
            self.data["min_delay"] = delay_df.min(axis=1)
        except Exception as e:
            logger.error(f"Error calculating recency features: {e}")
            self.data["avg_delay"] = np.nan
            self.data["max_delay"] = np.nan
            self.data["min_delay"] = np.nan

    def calculate_euclidean_distance_features(self, top_n=5):
        white_ball_cols = ["n1", "n2", "n3", "n4", "n5"]
        if "draw_date" not in self.data.columns or len(self.data) == 0:
            self.data["draw_date"] = pd.to_datetime("today") - pd.to_timedelta(
                self.data.index, unit="D"
            )
        if len(self.data) < 5: # Corresponds to min_draws_for_distance
            self.data["dist_to_recent"] = 0.5
            self.data["avg_dist_to_top_n"] = 0.5
            self.data["dist_to_centroid"] = 0.5
            return
        self.data.sort_values(by="draw_date", inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self._calculate_distance_to_recent(white_ball_cols)
        self._calculate_distance_to_top_n(white_ball_cols, top_n)
        self._calculate_distance_to_centroid(white_ball_cols)
        self._normalize_distance_features()

    def _calculate_distance_to_recent(self, white_ball_cols):
        if len(self.data) == 0:
            self.data["dist_to_recent"] = np.nan
            return
        most_recent = self.data[white_ball_cols].iloc[-1].values
        self.data["dist_to_recent"] = self.data[white_ball_cols].apply(
            lambda row: self._safe_euclidean(row.values, most_recent), axis=1
        )

    def _calculate_distance_to_top_n(self, white_ball_cols, top_n):
        if len(self.data) <= top_n:
            self.data["avg_dist_to_top_n"] = np.nan
            return
        top_combinations = self.data[white_ball_cols].iloc[-top_n - 1 : -1].values
        top_combinations_float = []
        for combo in top_combinations:
            try:
                combo_array = np.array(combo, dtype=float)
                if not np.isnan(combo_array).any() and not np.isinf(combo_array).any():
                    top_combinations_float.append(combo_array)
            except Exception:
                continue
        distances = []
        for _, row in self.data.iterrows():
            current = row[white_ball_cols].values
            try:
                current = np.array(current, dtype=float)
                if np.isnan(current).any() or np.isinf(current).any():
                    distances.append(np.nan)
                    continue
                valid_distances = [
                    euclidean(current, combo) for combo in top_combinations_float
                ]
                if valid_distances:
                    avg_dist = np.mean(valid_distances)
                    distances.append(avg_dist)
                else:
                    distances.append(np.nan)
            except Exception:
                distances.append(np.nan)
        self.data["avg_dist_to_top_n"] = distances

    def _calculate_distance_to_centroid(self, white_ball_cols):
        if len(self.data) <= 1:
            self.data["dist_to_centroid"] = np.nan
            return
        centroid = self.data[white_ball_cols].mean().values
        self.data["dist_to_centroid"] = self.data[white_ball_cols].apply(
            lambda row: self._safe_euclidean(row.values, centroid), axis=1
        )

    def _normalize_distance_features(self):
        for col in ["dist_to_recent", "avg_dist_to_top_n", "dist_to_centroid"]:
            if col in self.data.columns and not self.data[col].isna().all():
                max_val = self.data[col].max()
                if max_val > 0:
                    self.data[f"{col}_norm"] = self.data[col] / max_val

    def _safe_euclidean(self, row_values, reference_values):
        try:
            row_values = np.array(row_values, dtype=float)
            reference_values = np.array(reference_values, dtype=float)
            if (
                np.isnan(row_values).any()
                or np.isinf(row_values).any()
                or np.isnan(reference_values).any()
                or np.isinf(reference_values).any()
            ):
                return np.nan
            return euclidean(row_values, reference_values)
        except Exception as e:
            logger.warning(f"Error calculating Euclidean distance: {e}")
            return np.nan

    def _calculate_time_weights(self):
        if "draw_date" not in self.data.columns or len(self.data) == 0:
            self.data["draw_date"] = pd.to_datetime("today") - pd.to_timedelta(
                self.data.index, unit="D"
            )
        if len(self.data) < 30: # Corresponds to min_draws_for_analysis
            return
        self.data.sort_values(by="draw_date", inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        if not pd.api.types.is_datetime64_dtype(self.data["draw_date"]):
            self.data["draw_date"] = pd.to_datetime(self.data["draw_date"])
        first_draw_date = self.data["draw_date"].min()
        self.data["days_since_first"] = (
            self.data["draw_date"] - first_draw_date
        ).dt.days
        max_days = self.data["days_since_first"].max()
        if max_days == 0:
            self.data["time_weight"] = 1.0
            return
        if self.time_decay_function == "linear":
            self.data["time_weight"] = 1 - (
                self.data["days_since_first"] / max_days
            ) * (1 - self.min_weight_percent)
        elif self.time_decay_function == "exponential":
            self.data["time_weight"] = (
                np.exp(-self.time_decay_rate * self.data["days_since_first"])
                * (1 - self.min_weight_percent)
                + self.min_weight_percent
            )
        elif self.time_decay_function == "inverse_square":
            self.data["time_weight"] = (
                1 / (1 + (self.data["days_since_first"] * self.time_decay_rate) ** 2)
            ) * (1 - self.min_weight_percent) + self.min_weight_percent
        else:
            self.data["time_weight"] = (
                np.exp(-self.time_decay_rate * self.data["days_since_first"])
                * (1 - self.min_weight_percent)
                + self.min_weight_percent
            )
        weight_sum = self.data["time_weight"].sum()
        if weight_sum > 0:
            self.data["time_weight_norm"] = self.data["time_weight"] / weight_sum
        else:
            self.data["time_weight_norm"] = 1.0 / len(self.data)

    def _detect_trends(self):
        if "draw_date" not in self.data.columns:
            self.data["draw_date"] = pd.to_datetime("today") - pd.to_timedelta(
                self.data.index, unit="D"
            )
        if len(self.data) < 20: # Corresponds to min_draws_for_trends
            return
        if len(self.data) < self.moving_window_size:
            return
        self.data.sort_values(by="draw_date", inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        number_trends = {i: [] for i in range(1, 70)}
        pb_trends = {i: [] for i in range(1, 27)}
        window_count = min(self.num_windows, len(self.data) // self.moving_window_size)
        if window_count < 2:
            return
        for w in range(window_count):
            end_idx = len(self.data) - w * self.moving_window_size
            start_idx = max(0, end_idx - self.moving_window_size)
            window_data = self.data.iloc[start_idx:end_idx]
            white_counts = {i: 0 for i in range(1, 70)}
            pb_counts = {i: 0 for i in range(1, 27)}
            for _, row in window_data.iterrows():
                for col in ["n1", "n2", "n3", "n4", "n5"]:
                    num = row[col]
                    if 1 <= num <= 69:
                        white_counts[num] += 1
                pb = row.get("pb")
                if pd.notna(pb) and 1 <= pb <= 26:
                    pb_counts[pb] += 1
            for num, count in white_counts.items():
                number_trends[num].append(count / len(window_data))
            for num, count in pb_counts.items():
                pb_trends[num].append(count / len(window_data))
        white_ball_trends = {}
        pb_ball_trends = {}
        for num, freqs in number_trends.items():
            if len(freqs) >= 2:
                x = np.arange(len(freqs))
                poly_coeffs = np.polyfit(x, freqs, 1)
                slope = poly_coeffs[0]
                if slope > 0.005: white_ball_trends[num] = "increasing"
                elif slope < -0.005: white_ball_trends[num] = "decreasing"
                else: white_ball_trends[num] = "stable"
        for num, freqs in pb_trends.items():
            if len(freqs) >= 2:
                x = np.arange(len(freqs))
                poly_coeffs = np.polyfit(x, freqs, 1)
                slope = poly_coeffs[0]
                if slope > 0.005: pb_ball_trends[num] = "increasing"
                elif slope < -0.005: pb_ball_trends[num] = "decreasing"
                else: pb_ball_trends[num] = "stable"
        def get_trends_for_draw(row):
            trends = []
            for col in ["n1", "n2", "n3", "n4", "n5"]:
                num = row[col]
                if num in white_ball_trends:
                    trends.append(white_ball_trends[num])
            pb = row.get("pb")
            if pd.notna(pb) and pb in pb_ball_trends:
                trends.append(pb_ball_trends[pb])
            increasing = trends.count("increasing")
            decreasing = trends.count("decreasing")
            stable = trends.count("stable")
            return pd.Series(
                {
                    "increasing_trend_count": increasing,
                    "decreasing_trend_count": decreasing,
                    "stable_trend_count": stable,
                    "dominant_trend": max(
                        ["increasing", "decreasing", "stable"],
                        key=lambda x: [increasing, decreasing, stable][
                            ["increasing", "decreasing", "stable"].index(x)
                        ],
                    ),
                }
            )
        trend_features = self.data.apply(get_trends_for_draw, axis=1)
        self.data = pd.concat([self.data, trend_features], axis=1)

    def _detect_seasonal_patterns(self):
        if "draw_date" not in self.data.columns:
            self.data["draw_date"] = pd.to_datetime("today") - pd.to_timedelta(
                self.data.index, unit="D"
            )
        if len(self.data) < 60: # Corresponds to min_draws_for_seasonality
            return
        if len(self.data) < 2 * self.seasonality_period:
            return
        self.data.sort_values(by="draw_date", inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        if not pd.api.types.is_datetime64_dtype(self.data["draw_date"]):
            self.data["draw_date"] = pd.to_datetime(self.data["draw_date"])
        self.data["day_of_week"] = self.data["draw_date"].dt.dayofweek
        self.data["day_of_month"] = self.data["draw_date"].dt.day
        self.data["month"] = self.data["draw_date"].dt.month
        self.data["quarter"] = self.data["draw_date"].dt.quarter
        seasonal_numbers = set()
        for num in range(1, 70):
            time_series = pd.Series(0, index=self.data.index)
            for col in ["n1", "n2", "n3", "n4", "n5"]:
                time_series = time_series | (self.data[col] == num)
            time_series = time_series.astype(int)
            if time_series.sum() == 0:
                continue
            try:
                acf = sm.tsa.acf(
                    time_series, nlags=self.seasonality_period * 2, fft=True
                )
                if (
                    len(acf) > self.seasonality_period
                    and abs(acf[self.seasonality_period]) > self.seasonality_threshold
                ):
                    seasonal_numbers.add(num)
            except Exception as e:
                logger.warning(
                    f"Error calculating autocorrelation for number {num}: {e}"
                )
        seasonal_pb_numbers = set()
        for num in range(1, 27):
            time_series = (self.data["pb"] == num).astype(int)
            if time_series.sum() == 0:
                continue
            try:
                acf = sm.tsa.acf(
                    time_series, nlags=self.seasonality_period * 2, fft=True
                )
                if (
                    len(acf) > self.seasonality_period
                    and abs(acf[self.seasonality_period]) > self.seasonality_threshold
                ):
                    seasonal_pb_numbers.add(num)
            except Exception as e:
                logger.warning(
                    f"Error calculating autocorrelation for Powerball {num}: {e}"
                )
        def count_seasonal_numbers(row):
            count = sum(
                1
                for col in ["n1", "n2", "n3", "n4", "n5"]
                if row[col] in seasonal_numbers
            )
            pb_seasonal = 1 if row.get("pb") in seasonal_pb_numbers else 0
            return pd.Series(
                {
                    "seasonal_number_count": count,
                    "pb_seasonal": pb_seasonal,
                    "has_seasonality": count > 0 or pb_seasonal > 0,
                }
            )
        seasonality_features = self.data.apply(count_seasonal_numbers, axis=1)
        self.data = pd.concat([self.data, seasonality_features], axis=1)
        day_counts = self.data.groupby("day_of_week").size()
        month_counts = self.data.groupby("month").size()
        day_freq = day_counts / day_counts.sum()
        month_freq = month_counts / month_counts.sum()
        high_freq_days = day_freq[day_freq > 1.0 / 7 + 0.05].index.tolist()
        high_freq_months = month_freq[month_freq > 1.0 / 12 + 0.05].index.tolist()
        self.data["high_freq_day"] = self.data["day_of_week"].isin(high_freq_days)
        self.data["high_freq_month"] = self.data["month"].isin(high_freq_months)

class IntelligentGenerator:
    """
    Generates lottery plays based on predicted probabilities from the model.
    """
    def __init__(self):
        logger.info("IntelligentGenerator initialized.")

    def generate_plays(self, wb_probs, pb_probs, num_plays: int):
        """
        Generates a specified number of lottery plays using weighted sampling.

        Args:
            wb_probs (dict): A dictionary mapping white ball numbers to their probabilities.
            pb_probs (dict): A dictionary mapping Powerball numbers to their probabilities.
            num_plays (int): The number of unique plays to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the generated plays.
        """
        logger.info(f"Generating {num_plays} intelligent plays...")

        if not wb_probs or not pb_probs:
            raise ValueError("Probability dictionaries cannot be empty.")

        # Unpack dictionaries into lists for np.random.choice
        wb_numbers = list(wb_probs.keys())
        wb_probabilities = np.array(list(wb_probs.values()))
        wb_probabilities /= wb_probabilities.sum()  # Normalize to ensure they sum to 1

        pb_numbers = list(pb_probs.keys())
        pb_probabilities = np.array(list(pb_probs.values()))
        pb_probabilities /= pb_probabilities.sum()  # Normalize

        generated_plays = set()
        plays_list = []
        
        # We'll try a max number of times to avoid an infinite loop
        max_attempts = num_plays * 10
        attempts = 0
        while len(generated_plays) < num_plays and attempts < max_attempts:
            # Generate 5 unique white balls
            white_balls = np.random.choice(
                wb_numbers, 
                size=5, 
                replace=False, 
                p=wb_probabilities
            )
            white_balls.sort()

            # Generate 1 Powerball
            powerball = np.random.choice(
                pb_numbers,
                p=pb_probabilities
            )

            play = tuple(white_balls) + (powerball,)
            
            if play not in generated_plays:
                generated_plays.add(play)
                plays_list.append(list(play))
            
            attempts += 1
            
        if len(generated_plays) < num_plays:
            logger.warning(f"Could only generate {len(generated_plays)} unique plays after {max_attempts} attempts.")

        plays_df = pd.DataFrame(plays_list, columns=["n1", "n2", "n3", "n4", "n5", "pb"])
        logger.info(f"Successfully generated {len(plays_df)} plays.")
        
        return plays_df