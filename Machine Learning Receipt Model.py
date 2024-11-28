import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained CNN model (ensure the model is trained on characters/digits)
model_path = "c:/Users/flof_/Documents/COM6033_Project_Assignment/CNN_Model_ByClass.keras"
cnn_model = load_model(model_path)


# Path to the input image
image_path = "c:/Users/flof_/Documents/COM6033_Project_Assignment/TextTest.jpg"

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply preprocessing to highlight text regions
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours to detect regions
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Filter and extract bounding boxes
bounding_boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if 40 < w < 200 and 40 < h < 200:  # Ignore small regions
        bounding_boxes.append((x, y, x+w, y+h))

# Draw bounding boxes on the original image
for bbox in bounding_boxes:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = gray[y1:y2, x1:x2]  # Extract ROI
    roi = cv2.resize(roi, (28, 28))  # Resize
    roi = roi.astype('float32') / 255.0  # Normalize to 0-1
    roi = np.fliplr(roi)  # Flip left-to-right, if done during training
    roi = np.rot90(roi)   # Rotate, if done during training
    roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
    roi = np.expand_dims(roi, axis=0)   # Add batch dimension

    # Predict the character using the CNN model
    prediction = cnn_model.predict(roi)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

    label = chr(predicted_class)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    


# Display the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()