from keras.models import Model
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping

from pathlib import Path
from utils import pathlib_utils


class GenericModel:
    def __init__(self, name, version=1, auto_save=False):
        self.name = name
        self.version = version
        self.auto_save = auto_save

        self.model = None
        self.trained_model = None

        self.model_dir = Path(f'models/{self.name}')
        if not self.model_dir.exists(): self.model_dir.mkdir()
        self.model_weights_dir = self.model_dir / Path('weights')
        if not self.model_weights_dir.exists(): self.model_weights_dir.mkdir()
        self.model_tensorboard_dir = self.model_dir / Path('tensorboard')
        if not self.model_tensorboard_dir.exists(): self.model_tensorboard_dir.mkdir()

    def build_model(self):
        raise NotImplementedError

    def train_model(self, X, y, batch_size=32, epochs=1, early_stopping=False, validation_split=0.0,
                    optimizer=None, optimizer_params=dict(), loss=None, metrics=None):

        callbacks = [TensorBoard(log_dir=self.model_tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)]
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=5))

        optimizer = optimizer(**optimizer_params)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model.fit(X, y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=validation_split,
                       callbacks=callbacks,
                       verbose=1)

        self.trained_model = self.model
        if self.auto_save:
            self.save_model(update_version=True)

    def save_model(self, update_version=False):
        if update_version:
            self.version += 1
        self.model.save_weights(
            self.model_weights_dir / Path(f'v{self.version}.weights')
        )

    def load_model(self, version):
        assert self.model is not None, 'Build a model first using MyModel.build_model()'
        weights_file = self.model_weights_dir / Path(f'v{version}.weights')
        assert weights_file.exists(), f'Weights file does not exist for version {version} in directory {self.model_weights_dir}'
        self.model.load_weights(weights_file)

    def delete_model(self):
        pathlib_utils.delete_folder(self.model_dir)


# Used for sequence-to-sequence learning
class EncoderDecoder(GenericModel):
    def build_model(self):
        # RNN input shape per-batch is (max_seq_char_len, len(chars))
        encoder_inputs = Input(shape=(None, len(chars)), name="encoder_input")
        encoder = LSTM(latent_dim, return_state=True, name="encoder_lstm")
        _, enc_state_h, enc_state_c = encoder(encoder_inputs)
        encoder_states = [enc_state_h, enc_state_c]

        decoder_inputs = Input(shape=(None, len(chars)), name="decoder_input")
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
        decoder_lstm_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(len(chars), activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_lstm_outputs)

        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        self.model = model


# Used mainly for testing
class FeedForward(GenericModel):
    def build_model(self):
        inputs = Input(shape=(2,))

        dense1 = Dense(32, activation='relu')
        dense2 = Dense(32, activation='relu')
        final = Dense(1, activation='softmax')

        outputs = dense1(inputs)
        outputs = dense2(outputs)
        outputs = final(outputs)

        model = Model(inputs=inputs, outputs=outputs)

        self.model = model
