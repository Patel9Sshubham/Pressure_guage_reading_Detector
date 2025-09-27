# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pandas as pd
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def train_gauge_model():
    DATASET_DIR = "uploads"
    CSV_FILE = "readings.csv"
    IMG_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 100

    labels_df = pd.read_csv(CSV_FILE)

    X, y = [], []
    for _, row in labels_df.iterrows():
        filename = str(row["filename"]).strip()
        reading = row["reading"]

        img_path = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping missing image: {img_path}")
            continue

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            X.append(img)
            y.append(float(reading))
        except Exception as e:
            print(f"⚠️ Error reading {img_path}: {e}")
            continue

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")

    if len(X) == 0:
        raise ValueError("No valid images found! Check readings.csv vs images/")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    datagen.fit(X_train)

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="linear")
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint("best_gauge_model.h5", monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    model.save("gauge_reader_model.h5")
    return "✅ Training completed, model saved as gauge_reader_model.h5"
