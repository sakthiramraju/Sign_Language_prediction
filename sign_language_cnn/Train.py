# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping

# # SETTINGS
# img_size = 64
# batch_size = 64
# epochs = 70

# # DATA AUGMENTATION
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=10,
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=False,
#     validation_split=0.2
# )

# # TRAIN DATA
# train_data = train_datagen.flow_from_directory(
#     'dataset/train',
#     target_size=(img_size, img_size),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# # VALIDATION DATA
# val_data = train_datagen.flow_from_directory(
#     'dataset/train',
#     target_size=(img_size, img_size),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'
# )

# num_classes = train_data.num_classes

# # CNN MODEL
# model = Sequential([

#     Input(shape=(img_size, img_size, 3)),

#     Conv2D(32,(3,3),activation='relu'),
#     MaxPooling2D(2,2),

#     Conv2D(64,(3,3),activation='relu'),
#     MaxPooling2D(2,2),

#     Conv2D(128,(3,3),activation='relu'),
#     MaxPooling2D(2,2),

#     Conv2D(256,(3,3),activation='relu'),
#     MaxPooling2D(2,2),

#     Flatten(),

#     Dense(256,activation='relu'),
#     Dropout(0.5),

#     Dense(num_classes,activation='softmax')

# ])

# # COMPILE MODEL
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# early_stop = EarlyStopping(
#     patience=5,
#     restore_best_weights=True
# )

# # TRAIN MODEL
# model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=epochs,
#     callbacks=[early_stop]
# )

# # SAVE MODEL
# model.save("model/sign_model.h5")

# print("Model training complete and saved!")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                   Dropout, Input, BatchNormalization, 
                                   GlobalAveragePooling2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                      ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os

# SETTINGS
img_size = 128  # Increased from 64 for better feature extraction
batch_size = 32  # Adjusted for better gradient estimation
epochs = 100
num_classes = 29  # Your actual number of classes

# ENHANCED DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,  # Keep False for sign language (some signs are not symmetric)
    validation_split=0.2,
    fill_mode='nearest'
)

# Validation data should only be rescaled
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# TRAIN DATA
train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# VALIDATION DATA
val_data = val_datagen.flow_from_directory(
    'dataset/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# OPTION 1: Custom CNN with Batch Normalization (Improved from your original)
def create_custom_cnn():
    model = Sequential([
        Input(shape=(img_size, img_size, 3)),
        
        # Block 1
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Block 4
        Conv2D(256, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Classifier
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# OPTION 2: Transfer Learning with MobileNetV2 (Recommended for 99% accuracy)
def create_transfer_learning_model():
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classifier
    model = Sequential([
        Input(shape=(img_size, img_size, 3)),
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

# Choose which model to use
use_transfer_learning = True  # Set to True for better accuracy

if use_transfer_learning:
    model, base_model = create_transfer_learning_model()
else:
    model = create_custom_cnn()

# COMPILE MODEL with custom learning rate
initial_learning_rate = 0.001
optimizer = Adam(learning_rate=initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'model/best_sign_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# Calculate steps per epoch
steps_per_epoch = train_data.samples // batch_size
validation_steps = val_data.samples // batch_size

# TRAIN MODEL
print("Starting training...")
history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# If using transfer learning, fine-tune the model
if use_transfer_learning:
    print("\nFine-tuning the model...")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    fine_tune_epochs = 30
    total_epochs = epochs + fine_tune_epochs
    
    history_fine = model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1,
        callbacks=callbacks,
        verbose=1
    )

# Save final model
model.save("model/sign_model_final.h5")
print("Model training complete and saved!")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Evaluate on validation set
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_data)
print(f"\nFinal Validation Metrics:")
print(f"Accuracy: {val_accuracy*100:.2f}%")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")