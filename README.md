# Model manager
A Python library to manage Tensorflow 2 and XGBoost models.

## Installation
I _have not_ set this up as an installable package (e.g. using `pip`). To "install it", all you do is download the files (clone them) and place the folder inside whatever folder you're working in. Then you can simply follow what's shown below. Note that for the code below I have the downloaded folder named as `model_manager`. Make sure yours is named the same.

## Usage
### XGBoost
Instantiate a new XGBoost model directly through this package. To do so, just pass a `name` (used for saving the model), a bool `classifier` stating whether this is a classifier or regressor, and a dictionary of parameters to pass to XGBoost, called `xgb_kwargs`.

The XGBoost model will be accessible directly at `XGBModel.model`.
```
from model_manager.models import XGBModel

xgb_params = {'n_estimators': 100, 'max_depth': 30, 'learning_rate': 0.005}
xgb = XGBModel(name='my_model', classifier=True, xgb_kwargs=xgb_params)

from xgboost import XGBClassifier
assert isinstance(xgb.model, XGBClassifier)
```

You can then fit the model and run any other functions as usual by calling them directly on `XGBModel.model`.
```
xgb.model.fit(X, y, ...)
```

You can now easily persist the model by calling `XGBModel.save_model`. You can optionally pass the following parameters:
- `notes` - A string containing notes
- `update_version` (default `False`) - Models have versions (starting at version 1). Setting `update_version=False` causes the previous version to be overwritten. Setting `update_version=True` causes a new subfolder to be created for this version, thus keeping the previous version.
- `config` (default `dict()`) - A `dict` containing optional configuration information. This is more often used in Tensorflow models where things like epochs, validation sets, etc can be saved. However, you are free to save anything here you like.
- `save_attributes` (default `True`) - Whether or not to save all `XGBModel` attributes. This is useful when loading a saved model, as it allows it to fully load a previous model, including any parameters passed to `XGBClassifier` (or regressor).
```
v1_notes = 'My first XGBoost model!'
xgb.save_model(v1_notes)

# Fit the model on some new data
xgb.model.fit(X2, y2, ...)
v2_notes = 'Trained on some new data'
xgb.save_model(notes=v2_notes, update_version=True)
```
The model is then saved in a folder titled `name` (the model name supplied when first instanting `XGBModel`). Inside are various subfolders containing the notes, saved model, etc.

To load a previously saved model, simply pass the name of the model. Due to the rigid naming convention, the model folder will be located automatically. You may optionally pass a version if you want to load anything other than version 1. Note that even if you load a later version, all previous versions are still accessible.
```
new_xgb = XGBModel(name='my_model')
new_xgb.load_model(version=2)

assert new_xgb.model == xgb.model
```

### Tensorflow 2
In progress...