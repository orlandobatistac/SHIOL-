# SHIOL+ Phase 2A MVP - Basic Validation System

## Overview

The Basic Validation System validates SHIOL+ predictions against actual Powerball draw results. This is the Phase 2A MVP implementation that provides essential validation functionality for tracking prediction accuracy.

## Features

- **Automatic Validation**: Compares all predictions in `predictions_log` table with corresponding draws in `powerball_draws` table
- **Prize Tier Calculation**: Uses existing evaluator logic to determine prize categories (Jackpot, Match 5, Match 4 + PB, etc.)
- **CSV Export**: Saves validation results in structured CSV format
- **CLI Integration**: Easy-to-use command line interface
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## Usage

### Command Line Interface

```bash
# Run validation from command line
python src/cli.py validate
```

### Direct Python Usage

```python
from src.basic_validator import basic_validate_predictions

# Run validation and get CSV path
csv_path = basic_validate_predictions()
print(f"Results saved to: {csv_path}")
```

## Output Format

The validation results are saved as CSV files in `data/validations/` with the following format:

```csv
prediction_date,numbers,powerball,draw_numbers,draw_powerball,match_main,match_pb,prize_category,result_label
2025-07-23,19-35-2-25-18,25,19-35-2-25-18,25,5,1,Jackpot,Winner: Jackpot
2025-07-23,19-35-1-3-7,25,19-35-2-25-18,25,2,1,Match 2 + PB,Winner: Match 2 + PB
```

### CSV Fields

- **prediction_date**: Date of the prediction (YYYY-MM-DD)
- **numbers**: Predicted main numbers (format: N1-N2-N3-N4-N5)
- **powerball**: Predicted Powerball number
- **draw_numbers**: Actual draw main numbers (format: N1-N2-N3-N4-N5)
- **draw_powerball**: Actual draw Powerball number
- **match_main**: Number of main number matches (0-5)
- **match_pb**: Powerball match (0 or 1)
- **prize_category**: Prize tier (Jackpot, Match 5, Match 4 + PB, etc.)
- **result_label**: Human-readable result description

## Prize Categories

The system uses the standard Powerball prize structure:

- **Jackpot**: 5 main numbers + Powerball
- **Match 5**: 5 main numbers, no Powerball
- **Match 4 + PB**: 4 main numbers + Powerball
- **Match 4**: 4 main numbers, no Powerball
- **Match 3 + PB**: 3 main numbers + Powerball
- **Match 3**: 3 main numbers, no Powerball
- **Match 2 + PB**: 2 main numbers + Powerball
- **Match 1 + PB**: 1 main number + Powerball
- **Match PB**: Powerball only
- **Non-winning**: No significant matches

## Technical Details

### Database Requirements

The validation system requires two database tables:

1. **predictions_log**: Contains prediction data with timestamps
2. **powerball_draws**: Contains historical draw results

### Matching Logic

- Predictions are matched to draws by date (extracted from prediction timestamp)
- Only predictions with corresponding draw dates are validated
- Predictions without matching draws are logged but not included in results

### Error Handling

- Comprehensive error handling with detailed logging
- Graceful handling of missing data
- Automatic directory creation for output files
- Database connection error recovery

## Files Generated

- **CSV Results**: `data/validations/validation_results_YYYYMMDD_HHMMSS.csv`
- **Log Files**: Standard SHIOL+ logging system

## Integration

This validation system integrates seamlessly with existing SHIOL+ components:

- Uses existing database connection methods from `src/database.py`
- Leverages prize tier logic from `src/evaluator.py`
- Follows existing logging patterns
- Integrates with CLI system in `src/cli.py`

## Future Enhancements

This Phase 2A MVP provides the foundation for future validation enhancements:

- Real-time validation when new draws are available
- Batch processing for large historical datasets
- Advanced accuracy metrics and reporting
- API endpoints for validation results
- Performance optimization for large-scale validation

## Example Output

```
--- Validation Complete ---
Results saved to: data/validations/validation_results_20250725_214331.csv
Total predictions validated: 2
Winning predictions: 2
Win rate: 100.0%

Winning predictions breakdown:
  Jackpot: 1
  Match 2 + PB: 1
---------------------------
```

## Support

For issues or questions about the validation system, check the log files in the `logs/` directory for detailed error information.