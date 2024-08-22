import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from tkinter import Tk
from tkinter.filedialog import askopenfilename

face_classifier = cv2.CascadeClassifier(r'C:\Users\arjun\OneDrive\Desktop\RESUME\SDE Projects\ML\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\arjun\OneDrive\Desktop\RESUME\SDE Projects\ML\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion_in_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to load image {image_path}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.namedWindow('Emotion Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Emotion Detector', 800, 600)
    
    cv2.imshow('Emotion Detector', frame)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

def detect_emotion_live():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Select Mode:")
    print("1. Real-time emotion detection using webcam")
    print("2. Emotion detection in an image file")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        detect_emotion_live()
    elif choice == '2':
        root = Tk()
        root.withdraw()
        image_path = askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not image_path:
            print("No image selected.")
            return
        detect_emotion_in_image(image_path)
    else:
        print("Invalid choice. Please select either 1 or 2.")

if __name__ == "__main__":
    main()
