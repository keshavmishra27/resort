from keras.api.applications import MobileNetV2
import numpy as np
import pandas as pd
from scikeras.wrappers import KerasClassifier
import seaborn as sb
import matplotlib.pyplot as plot
import tensorflow as tf
import keras

waste_arr=["battery","biological","brown-glass","cardboard","clothes","green-glass","metal","paper","plastic","shoes","trash","white-glass"]

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1/255)

train_dir = r"c:\Users\User\Downloads\waste data\garbage_classification\garbage_split\train"
val_dir   = r"c:\Users\User\Downloads\waste data\garbage_classification\garbage_split\val"
test_dir  = r"c:\Users\User\Downloads\waste data\garbage_classification\garbage_split\test"

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224,224),
    batch_size=16, class_mode='sparse'
)

valid_gen = test_datagen.flow_from_directory(
    val_dir,   target_size=(224,224),
    batch_size=16, class_mode='sparse'
)


from keras import models, layers
from sklearn.model_selection import GridSearchCV

from keras.applications import EfficientNetB0
from keras import layers, models, optimizers

def build_model(optimizer='adam', learning_rate=1e-4, dropout_rate=0.3, dense_units=128, unfreeze_layers=50):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Freeze all layers except the last few
    base_model.trainable = True
    # Freeze all layers except the last `unfreeze_layers`
    for layer in base_model.layers[:-100]:  # Freeze more than before
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(12, activation='softmax')  # 12 waste classes
    ])

    # Select optimizers
    if optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Get multiple batches to have enough data for cross-validation
X_list = []
y_list = []

# Reset the generator
train_gen.reset()

# Extract several batches (adjust number based on your dataset size)
num_batches = min(10, len(train_gen))  # Get 10 batches or all available
for i in range(len(train_gen)):
    X_batch, y_batch = next(train_gen)
    X_list.append(X_batch)
    y_list.append(y_batch)

X_data = np.concatenate(X_list, axis=0)
y_data = np.concatenate(y_list, axis=0)
print(np.unique(y_data))  # Output should be like: [0 1 2 ... 11]


print(f"Total samples for grid search: {X_data.shape[0]}")

# Fix 4: Use stratified splits for small datasets
from sklearn.model_selection import StratifiedKFold

# Convert categorical (one-hot) labels to class indices for stratification
y_classes = y_data


# Extract the actual Keras model from the KerasClassifier wrapper
final_model = build_model()

# Recompile for categorical data (since GridSearch used sparse categorical)
final_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='rmsprop', 
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

earlystop_cb = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)

# Reset generators
train_gen.reset()
valid_gen.reset()

history = final_model.fit(
    train_gen,
    steps_per_epoch = len(train_gen), 
    epochs=1,  # Use best epochs from grid search
    batch_size=16,  # Note: batch_size here won't override generator's batch_size
    validation_data=valid_gen,
    validation_steps=len(valid_gen) // 4,
    callbacks=[checkpoint_cb, earlystop_cb,reduce_lr]
)

# Plotting results
import matplotlib.pyplot as plt
df = pd.DataFrame(history.history)
df[['loss','val_loss']].plot()
plt.title('Training and Validation Loss')
plt.show()

df[['accuracy','val_accuracy']].plot()
plt.title('Training and Validation Accuracy')
plt.show()

# Evaluation
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=(224,224),
    batch_size=16, class_mode='sparse'
)

print("Evaluating on test set:")
test_results = final_model.evaluate(test_gen)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")