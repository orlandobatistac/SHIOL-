import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import List
from loguru import logger


class RNNPredictor:
    """
    Predictor basado en una red neuronal recurrente (RNN) para capturar patrones temporales.
    """
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(10, 6)))
        model.add(Dense(6, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def train(self, epochs: int = 50, batch_size: int = 32):
        logger.info("Entrenando el modelo RNN...")
        X, y = self._prepare_data(self.historical_data)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, n_samples: int) -> List[List[int]]:
        logger.info(f"Generando {n_samples} predicciones con el modelo RNN...")
        predictions = self.model.predict(np.random.rand(n_samples, 10, 6))
        return predictions.argmax(axis=-1).tolist()

    def _prepare_data(self, data: pd.DataFrame) -> tuple:
        # Preprocesar datos históricos para la RNN
        X, y = [], []
        for i in range(len(data) - 10):
            X.append(data.iloc[i:i+10].to_numpy())
            y.append(data.iloc[i+10].to_numpy())
        return np.array(X), np.array(y)
