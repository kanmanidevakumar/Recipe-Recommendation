import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import numpy as np
import os

# =======================
# 1️⃣ Replace this with your dataset path
# Make sure dataset is structured as:
# dataset/
#    potato/
#    tomato/
#    carrot/
#    etc.
DATASET_PATH = r"D:\project_root\dataset\archive (1)\Fruits_Vegetables_Dataset(12000)"
# =======================

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

# ✅ Data Augmentation & Train/Validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# =======================
# 2️⃣ Check dataset balance
print("\n📊 Dataset Balance:")
for cls, index in train_generator.class_indices.items():
    count = np.sum(train_generator.labels == index)
    print(f"{cls}: {count} images")
# =======================

# =======================
# 3️⃣ Save class labels
class_labels = list(train_generator.class_indices.keys())
with open("class_labels.pkl", "wb") as f:
    pickle.dump(class_labels, f)
print("\n✅ Saved class labels:", class_labels)
# =======================

# =======================
# 4️⃣ Transfer Learning: MobileNetV2
base_model = MobileNetV2(input_shape=(150,150,3), include_top=False, weights="imagenet")
base_model.trainable = False  # freeze pretrained layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(class_labels), activation="softmax")
])
# =======================

# ✅ Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# =======================
# 5️⃣ Train model (50 epochs)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stop]
)
# =======================

# ✅ Save trained model
model.save("trained_model.h5")
print("\n✅ Model saved as trained_model.h5")

# =======================
# 6️⃣ Debug function: test top-3 predictions
def debug_predict(image_array):
    prediction = model.predict(image_array)
    top_3 = np.argsort(prediction[0])[-3:][::-1]
    print("\n📌 Top-3 Predictions:")
    for idx in top_3:
        print(class_labels[idx], ":", prediction[0][idx])
# =======================
