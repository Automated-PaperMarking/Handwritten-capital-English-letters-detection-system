import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Create folders
os.makedirs("graphs", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Load dataset
print("🔄 Loading dataset...")
data = pd.read_csv("A_Z Handwritten Data.csv")
label_map = {i: chr(i + 65) for i in range(26)}

# Visualize some examples
def plot_samples(letter_idx, name):
    samples = data[data['0'] == letter_idx].iloc[:5, 1:].values.reshape(-1, 28, 28)
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(samples[i], cmap='gray')
        axs[i].axis('off')
    plt.suptitle(f"Samples for letter '{name}'")
    plt.savefig(f"graphs/sample_letters_{name}.png")
    plt.close()

plot_samples(1, 'B')
plot_samples(12, 'M')

# Check class distribution (imbalanced)
counts = data['0'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.barplot(x=[label_map[i] for i in counts.index], y=counts.values)
plt.title("Class Distribution Before Balancing")
plt.xlabel("Letter")
plt.ylabel("Sample Count")
plt.savefig("graphs/class_distribution_before.png")
plt.close()

# Balance the data (under-sampling to smallest class count)
min_count = counts.min()
balanced_data = data.groupby('0').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# Visualize balanced distribution
balanced_counts = balanced_data['0'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.barplot(x=[label_map[i] for i in balanced_counts.index], y=balanced_counts.values)
plt.title("Class Distribution After Balancing")
plt.xlabel("Letter")
plt.ylabel("Sample Count")
plt.savefig("graphs/class_distribution_after.png")
plt.close()

# Prepare data
X = balanced_data.iloc[:, 1:].values / 255.0
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(balanced_data.iloc[:, 0].values, num_classes=26)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN
print("🧠 Building CNN...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(26, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
print("🚀 Training model...")
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Save accuracy graph
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("graphs/training_accuracy.png")
plt.close()

# Save loss graph
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("graphs/training_loss.png")
plt.close()

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# Save model
model.save("model/handwritten_capital_model.h5")
print("✅ Model saved to model/handwritten_capital_model_2.h5")
