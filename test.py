from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

print("Loading model...")
model = load_model("models/emotion_model.h5")

test_dir = "dataset/archive/test"

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

print("Evaluating on test dataset...")

loss, acc = model.evaluate(test_gen)

print("\nTest Accuracy:", round(acc*100,2), "%")
print("Test Loss:", round(loss,4))