import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 1: Load Dataset
print("Loading dataset...")
data = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')
print(f"Dataset shape: {data.shape}")  # Should be (372450, 785)

# Step 2: Split into features and labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Step 3: Normalize pixel values
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

# Step 4: One-hot encode labels
y = to_categorical(y, num_classes=26)

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 6: Build CNN model
print("Building model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 7: Train the model
print("Training model...")
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test))

# Step 8: Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# Step 9: Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Make predictions on test samples
print("\nSample Predictions:")
pred = model.predict(X_test[:5])
for i in range(5):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    pred_letter = chr(np.argmax(pred[i]) + 65)
    plt.title(f"Predicted: {pred_letter}")
    plt.axis('off')
    plt.show()

# Step 11: Save the trained model
model.save("handwritten_capital_model.h5")
print("✅ Model saved as 'handwritten_capital_model.h5'")