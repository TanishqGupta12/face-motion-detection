import cv2 as cv
from keras.models import model_from_json
import numpy as np 

# Load model architecture
json_file = open("emotiondetectore.json", "r")
modle_json = json_file.read()
json_file.close()

# Load the model
model = model_from_json(modle_json)
model.load_weights("emotiondetectore.h5")

# Load Haar cascade
haar_file = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(haar_file)

# Feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Add batch and channel dimensions
    return feature / 255.0  # Normalize the image

# Initialize webcam
webcam = cv.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image from webcam")
        break

    # Convert to grayscale
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            # Extract the face region
            image = gray[q: q+s, p: p+r]
            cv.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            
            # Resize and preprocess the image
            image = cv.resize(image, (48, 48))
            img = extract_features(image)
            
            # Predict emotion
            prod = model.predict(img)
            pro_label = labels[prod.argmax()]

            # Display label on the image
            cv.putText(im, '%s' % (pro_label), (p-10, q-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Show output
        cv.imshow("OUTPUT", im)
        
        # Break loop on 'Esc' key press
        if cv.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for 'Esc'
            break

    except cv.error as e:
        print(f"OpenCV error: {e}")

# Release resources
webcam.release()
cv.destroyAllWindows()
