import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from loguru import logger


class VariationalAutoencoder:
    """
    Generative model based on a Variational Autoencoder (VAE) to generate number combinations.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model, self.encoder, self.decoder = self._build_model()

    def _build_model(self) -> Tuple[Model, Model, Model]:
        # Encoder
        inputs = Input(shape=(self.input_dim,))
        h = Dense(64, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        # Decoder
        decoder_h = Dense(64, activation='relu')
        decoder_mean = Dense(self.input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # Models
        encoder = Model(inputs, z_mean)
        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        decoder = Model(decoder_input, _x_decoded_mean)

        vae = Model(inputs, x_decoded_mean)
        vae.add_loss(self._vae_loss(inputs, x_decoded_mean, z_mean, z_log_var))
        vae.compile(optimizer='adam')

        return vae, encoder, decoder

    def _vae_loss(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = K.sum(K.binary_crossentropy(inputs, outputs), axis=-1)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 32):
        logger.info("Training the VAE model...")
        self.model.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=1)

    def generate(self, n_samples: int) -> np.ndarray:
        logger.info(f"Generating {n_samples} combinations with the VAE model...")
        latent_samples = np.random.normal(size=(n_samples, self.latent_dim))
        return self.decoder.predict(latent_samples)


class GenerativePredictor:
    """
    Hybrid predictor that combines a generative model with a scoring system.
    """
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.vae = VariationalAutoencoder(input_dim=6, latent_dim=2)

    def train(self):
        # Preprocess historical data
        data = self._preprocess_data(self.historical_data)
        self.vae.train(data)

    def predict(self, n_samples: int) -> List[List[int]]:
        generated_data = self.vae.generate(n_samples)
        return self._postprocess_data(generated_data)

    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        # Normalize historical data
        return data.to_numpy() / 69  # Assuming a range of 1-69 for white numbers

    def _postprocess_data(self, data: np.ndarray) -> List[List[int]]:
        # Denormalize and round
        return (data * 69).astype(int).tolist()
