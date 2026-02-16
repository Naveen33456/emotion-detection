import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("Preparing dataset...")

train_dir = os.path.join("Dataset", "archive", "train")
test_dir = os.path.join("Dataset", "archive", "test")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

print("Building CNN model...")

model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(48,48,1)),
    MaxPooling2D(),
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(),
    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128,activation="relu"),
    Dropout(0.5),
    Dense(7,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training started...")

model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=10
)

model.save("models/emotion_model.h5")

print("Model saved successfully 🎉")