import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tensorflow.keras.applications import MobileNetV2
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

waste_arr=["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]

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

valid_datagen = ImageDataGenerator(rescale=1/255)

train_dir = os.getenv('TRAIN_DIR', r'dataset\splitted\train')
val_dir   = os.getenv('VAL_DIR', r'dataset\splitted\valid')
test_dir  = os.getenv('TEST_DIR', r'dataset\splitted\test')

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224,224),
    batch_size=32, class_mode='sparse'
)

print(train_gen.class_indices)

valid_gen = valid_datagen.flow_from_directory(
    val_dir,   target_size=(224,224),
    batch_size=32, class_mode='sparse'
)


def build_model(optimizer='adam', learning_rate=1e-4, dropout_rate=0.3, dense_units=128, unfreeze_layers=50):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    base_model.trainable = True
   
    # Unfreezing 100 layers immediately is too aggressive and causes catastrophic forgetting. 
    # Let's unfreeze only the top 20 layers for gentle fine-tuning.
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(len(waste_arr), activation='softmax')  
    ])

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



train_gen.reset()

final_model = build_model(learning_rate=1e-4)


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

earlystop_cb = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)


train_gen.reset()
valid_gen.reset()

history = final_model.fit(
    train_gen,
    steps_per_epoch = len(train_gen), 
    epochs=5,
    batch_size=32,  
    validation_data=valid_gen,
    callbacks=[checkpoint_cb, earlystop_cb,reduce_lr]
)


df = pd.DataFrame(history.history)
df[['loss','val_loss']].plot(marker='o')
plt.title('Training and Validation Loss')
plt.savefig('loss_plot.png')
plt.close()

df[['accuracy','val_accuracy']].plot(marker='o')
plt.title('Training and Validation Accuracy')
plt.savefig('accuracy_plot.png')
plt.close()
model_save_path = os.getenv('MODEL_PATH', r'backend\models\garbage_tf_model.h5')
final_model.save(model_save_path)

#Final Evaluation
print("\n Final Model Evaluation ")

test_gen = valid_datagen.flow_from_directory(
    test_dir, target_size=(224,224),
    batch_size=32, class_mode='sparse', shuffle=False
)

test_loss, test_accuracy = final_model.evaluate(test_gen)
print(f"Final Testing Loss: {test_loss:.4f}")
print(f"Final Testing Accuracy: {test_accuracy:.4f}")

# Get predictions
print("\nGenerating predictions ")
Y_pred = final_model.predict(test_gen, steps=len(test_gen))
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

#test printing
print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=class_labels))
print('Generating Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()