import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

HEIGHT = 28
WIDTH = 28

train = pd.read_csv("c:/Users/flof_/Documents/COM6033_Project_Assignment/emnist-balanced-train.csv")
test = pd.read_csv("c:/Users/flof_/Documents/COM6033_Project_Assignment/emnist-balanced-test.csv")

x_train = train.iloc[:,1:]
y_train = train.iloc[:,0]

x_test = test.iloc[:,1:]
y_test = test.iloc[:,0]

def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

train_x = np.asarray(x_train)
train_x = np.apply_along_axis(rotate, 1, train_x)

test_x = np.asarray(x_test)
test_x = np.apply_along_axis(rotate, 1, test_x)

train_x = train_x.astype('float32')
train_x /= 255
test_x = test_x.astype('float32')
test_x /= 255

print ("test_x:",test_x.shape)
print ("train_x:",train_x.shape)

y_train = to_categorical(y_train, num_classes=47)
y_test = to_categorical(y_test, num_classes=47)

model = Sequential([
    Input(shape = (28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(47, activation='softmax')  # 47 classes Includes uppercase and lowercase letters and digits.
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_x, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_x, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save('c:/Users/flof_/Documents/COM6033_Project_Assignment/CNN_Model.keras')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()