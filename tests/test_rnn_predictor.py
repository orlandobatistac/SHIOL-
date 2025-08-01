import pandas as pd
from src.rnn_predictor import RNNPredictor

def test_rnn_predictor():
    """Unit test for the RNN-based model."""
    # Simulates historical data
    historical_data = pd.DataFrame({
        "n1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "n2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "n3": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "n4": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        "n5": [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        "pb": [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    })

    # Test the RNN predictor
    rnn_predictor = RNNPredictor(historical_data)
    rnn_predictor.train()
    predictions = rnn_predictor.predict(n_samples=5)

    # Verify that 5 predictions are generated
    assert len(predictions) == 5
    print("Predictions generated by the RNN model:", predictions)

if __name__ == "__main__":
    test_rnn_predictor()
