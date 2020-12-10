from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from xgboost import XGBRegressor, XGBClassifier

from pathlib import Path
from utils import file_io
from datetime import datetime
import json

from utils.file_io import smart_load, list_to_csv


class GenericModel:
    def __init__(self, name, version=1):
        super()
        self.name = name
        self.version = version

        self.model = None

        self.base_model_dir = Path(f'models/{self.name}')
        if not self.base_model_dir.exists(): self.base_model_dir.mkdir()
        self.model_dir = self.base_model_dir / Path(f'saved_models')
        if not self.model_dir.exists(): self.model_dir.mkdir()
        self.weights_dir = self.base_model_dir / Path('weights')
        if not self.weights_dir.exists(): self.weights_dir.mkdir()
        self.attr_dir = self.base_model_dir / Path('attr')
        if not self.attr_dir.exists(): self.attr_dir.mkdir()
        self.notes_dir = self.base_model_dir / Path('notes')
        if not self.notes_dir.exists(): self.notes_dir.mkdir()
        self.config_dir = self.base_model_dir / Path('config')
        if not self.config_dir.exists(): self.config_dir.mkdir()
        self.preds_dir = self.base_model_dir / Path('predictions')
        if not self.preds_dir.exists(): self.preds_dir.mkdir()

        self.dir = {
            'base_model_dir': self.base_model_dir.as_posix(),
            'model_dir': self.model_dir.as_posix(),
            'weights_dir': self.weights_dir.as_posix(),
            'attr_dir': self.attr_dir.as_posix(),
            'notes_dir': self.notes_dir.as_posix(),
            'config_dir': self.config_dir.as_posix(),
            'preds_dir': self.preds_dir.as_posix()
        }

    def train(self):
        raise NotImplementedError

    def save_model(self, notes=None, update_version=False, config=None, save_attributes=True):
        raise NotImplementedError

    def load_model(self, name, version=1):
        raise NotImplementedError

    def _save_config(self, config):
        config_file = self.config_dir / Path(f'v{self.version}.txt')
        config_attrs = vars(config)
        file_io.append_or_write(config_file, config_attrs)

    def _save_notes(self, notes):
        version_notes_file = self.notes_dir / Path(f'v{self.version}.txt')
        global_notes_file = self.notes_dir / Path(f'version_notes.txt')

        formatted_notes = f'{"=" * 5} Version {self.version} ({datetime.now().strftime("%B %d, %Y - %H:%M:%S")}) {"=" * 5}'
        formatted_notes += '\n'
        formatted_notes += notes
        formatted_notes += '\n\n'

        file_io.append_or_write(version_notes_file, formatted_notes)
        file_io.append_or_write(global_notes_file, formatted_notes)

    def _save_attributes(self):
        version_attr_file = self.attr_dir / Path(f'v{self.version}.txt')

        # Find which attributes are actually serializable and only save those
        all_attrs = vars(self)
        for a in all_attrs.keys():
            try:
                json.dumps(all_attrs[a])
                file_io.append_or_write(version_attr_file, f'{a}:{all_attrs[a]}', newline=True)
            except:
                pass

    def _load_attributes(self, attr_dir=None):
        if attr_dir is None:
            # Allow the model to load attributes from a different pretrained saved model
            attr_dir = self.attr_dir
        attr_file = attr_dir / Path(f'v{self.version}.txt')
        with open(attr_file, 'r') as f:
            row = f.readline()
            while row:
                try:
                    k, v = row.split(':')
                except ValueError:
                    break
                v = smart_load(v, cleanup=True)
                setattr(self, k, v)
                row = f.readline()

    def delete_model(self):
        file_io.delete_folder(self.base_model_dir)


class XGBModel(GenericModel):
    def __init__(self, name, version=1, classifier=True, xgb_kwargs=None):
        super().__init__(name, version)
        self.xgb_kwargs = xgb_kwargs
        if classifier:
            self.model = XGBClassifier(**xgb_kwargs)
        else:
            self.model = XGBRegressor(**xgb_kwargs)

    def train(self):
        print('No custom train method implemented. Instead call self.model.fit(...)')

    def save_model(self, notes=None, update_version=False, config=None, save_attributes=True):
        if update_version:
            self.version += 1

        try:
            model_path = self.model_dir / Path(f'v{self.version}.json')
            self.model.save_model(
                model_path.as_posix()
            )
        except Exception as e:
            print('Error saving model')
            print(e)
            raise

        if save_attributes:
            self._save_attributes()

        if notes is not None:
            self._save_notes(notes)

        if config is not None:
            self._save_config(config)

    def load_model(self, version, load_attributes=True):
        # First load the xgb_kwargs so that we can create a new instance of XGB
        self._load_attributes(self.attr_dir)
        if hasattr(self, 'xgb_kwargs'):
            self.model = self.model(self.xgb_kwargs)

        # Next load the model
        model_path = self.model_dir / Path(f'v{self.version}.json')
        assert model_path.exists(), f'No model exists at {model_path.as_posix()}'
        self.model.load_model(model_path)


class TensorflowModel(GenericModel):
    '''
    A generic class for all models. Models which inherit from this gain the ability to keep notes, easily be saved
    and deleted, implement early stopping, etc.
    When defining a new model inheriting from this class be sure to specify the name. This name will be used when
    creating directories for saving.
    '''
    def __init__(self, name, version=1):
        super().__init__(name, version)

        self.tensorboard_dir = self.base_model_dir / Path('tensorboard')
        if not self.tensorboard_dir.exists(): self.tensorboard_dir.mkdir()

        self.dir['tensorboard_dir'] = self.tensorboard_dir.as_posix()

        self.model = None

    def set_model(self, tf_model):
        self.model = tf_model

    def train(self, X, y, epochs=1, early_stopping=False, validation_split=0.0,
                    optimizer=None, optimizer_params=dict(), loss=None, metrics=None):

        callbacks = [TensorBoard(log_dir=self.tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)]
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=5))

        optimizer = optimizer(**optimizer_params)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model.fit(X, y,
                       epochs=epochs,
                       validation_split=validation_split,
                       callbacks=callbacks,
                       verbose=1)

    def save_model(self, notes=None, update_version=False, config=None, save_attributes=True):
        directory_path = self.base_model_dir / Path('directory.json')

        # self.save_directory(directory_path)

        if update_version:
            self.version += 1

        try:
            weights_path = self.weights_dir / Path(f'v{self.version}.weights')
            self.model.save_weights(
                weights_path.as_posix()
            )
        except Exception as e:
            print('Error saving weights')
            print(e)
            raise
        
        try:
            model_path = self.model_dir / Path(f'v{self.version}.h5')
            self.model.save(
                model_path.as_posix()
            )
        except Exception as e:
            print('Error saving model')
            print(e)
            raise

        if save_attributes:
            self._save_attributes()

        if notes is not None:
            self._save_notes(notes)

        if config is not None:
            self._save_config(config)

    def save_directory(self):
        # Save a file pointing to all other folders. This will make loading easier later.
        existing_dir = self.base_model_dir / Path('directory.json')
        current_dir = json.loads(existing_dir)
        dir_str = json.dumps(self.dir)

        if current_dir != dir_str:
            existing_dir.unlink(existing_dir)
            file_io.append_or_write(existing_dir, dir_str)

    def get_attributes(self):
        # Find which attributes are actually serializable and only return those
        all_attrs = vars(self)
        attrs = dict()
        for a in all_attrs.keys():
            try:
                json.dumps(all_attrs[a])
                attrs[a] = all_attrs[a]
            except:
                pass
        return attrs

    def load_model(self, version, load_attributes=False):
        # Used to load a saved model. Note that some complex models cannot easily be saved. In that case it is easier
        # to build the model from scratch and then load the weights separately using the load_weights method
        model_file = self.model_dir / Path(f'v{version}.h5')
        assert model_file.exists(), f'Model file does not exist for version {version} in directory {self.model_dir}'
        self.model = load_model(model_file.as_posix())

        if load_attributes:
            self._load_attributes()

    def load_weights(self, target_model=None, weights_file=None, load_attributes=False):
        # Takes an already built model and loads weights from another (pretrained) model with the same architecture
        assert target_model is not None or weights_file is not None, 'You must specify either a target model to load weights from or a weights file'
        assert target_model is None or weights_file is None, 'Specify target_model or weights_file, not both'

        if target_model is not None:
            self.model.set_weights(target_model.model.get_weights())
        else:
            self.model.load_weights(weights_file)

        if load_attributes:
            self._load_attributes(attr_dir=target_model.attr_dir)
