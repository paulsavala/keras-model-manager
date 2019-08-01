from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import TensorBoard, EarlyStopping

from pathlib import Path
from utils import pathlib_utils
from datetime import datetime


class GenericModel:
    def __init__(self, name, version=1):
        self.name = name
        self.version = version

        self.model = None

        self.model_dir = Path(f'models/{self.name}')
        if not self.model_dir.exists(): self.model_dir.mkdir()
        self.weights_dir = self.model_dir / Path('weights')
        if not self.weights_dir.exists(): self.weights_dir.mkdir()
        self.tensorboard_dir = self.model_dir / Path('tensorboard')
        if not self.tensorboard_dir.exists(): self.tensorboard_dir.mkdir()
        self.notes_dir = self.model_dir / Path('notes')
        if not self.notes_dir.exists(): self.notes_dir.mkdir()

    def build_model(self):
        raise NotImplementedError

    def train_model(self, X, y, batch_size=32, epochs=1, early_stopping=False, validation_split=0.0,
                    optimizer=None, optimizer_params=dict(), loss=None, metrics=None):

        callbacks = [TensorBoard(log_dir=self.tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)]
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

    def save_model(self, notes=None, update_version=False):
        if update_version:
            self.version += 1

        try:
            self.model.save_weights(
                self.weights_dir / Path(f'v{self.version}.weights')
            )
        except Exception as e:
            print('Error saving weights')
            print(e)
            raise

        if notes:
            version_notes_file = self.notes_dir / Path(f'v{self.version}.txt')
            global_notes_file = self.notes_dir / Path(f'version_notes.txt')

            formatted_notes = f'{"="*5} Version {self.version} ({datetime.now().strftime("%B %d, %Y - %H:%M:%S")}) {"="*5}'
            formatted_notes += '\n'
            formatted_notes += notes
            formatted_notes += '\n\n'

            pathlib_utils.append_or_write(version_notes_file, formatted_notes)
            pathlib_utils.append_or_write(global_notes_file, formatted_notes)

    def load_model(self, version):
        assert self.model is not None, 'Build a model first using MyModel.build_model()'
        weights_file = self.weights_dir / Path(f'v{version}.weights')
        assert weights_file.exists(), f'Weights file does not exist for version {version} in directory {self.weights_dir}'
        self.model.load_weights(weights_file)

    def delete_model(self):
        pathlib_utils.delete_folder(self.model_dir)


# Used mainly for testing and examples
class FeedForward(GenericModel):
    def build_model(self):
        inputs = Input(shape=(2,))

        dense1 = Dense(32, activation='relu')
        dense2 = Dense(32, activation='relu')
        final = Dense(1, activation='softmax')

        outputs = dense1(inputs)
        outputs = dense2(outputs)
        outputs = final(outputs)

        self.model = Model(inputs=inputs, outputs=outputs)
