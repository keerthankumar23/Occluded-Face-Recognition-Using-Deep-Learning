import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# Ensure the dataset directories exist
os.makedirs('C:\\Users\\Keerthan\\Desktop\\tempdataset\\train\\sam', exist_ok=True)
os.makedirs('C:\\Users\\Keerthan\\Desktop\\tempdataset\\test\\sam', exist_ok=True)

# Initialize face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to extract face from an image
def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
        return cropped_face

# GUI Class
class FaceApp:
    def _init_(self, root):  # Corrected constructor name
        self.root = root
        self.root.title("Face Detection and Recognition")
        self.root.geometry("800x600")

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.capture_button = tk.Button(root, text="Capture Faces", command=self.capture_faces)
        self.capture_button.pack()

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack()

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.cap = cv2.VideoCapture(0)
        self.count = 0
        self.model = None
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def capture_faces(self):
        while True:
            ret, frame = self.cap.read()
            if face_extractor(frame) is not None:
                self.count += 1
                face = cv2.resize(face_extractor(frame), (224, 224))
                file_name_path = f'C:\\Users\\Keerthan\\Desktop\\tempdataset\\train\\sam\\{self.count}.jpg'
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(self.count), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('face cropper', face)
            else:
                print("Face not found")
                pass
            if cv2.waitKey(1) == 13 or self.count == 20:
                break
        self.cap.release()
        cv2.destroyAllWindows()
        print("Sample Collection Completed")

    def train_model(self):
        img_rows, img_cols = 224, 224
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))
        base_model.trainable = False

        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',  # Use binary crossentropy for binary classification
                           metrics=['accuracy'])

        train_data_dir = 'C:\\Users\\Keerthan\\Desktop\\tempdataset\\train\\'
        validation_data_dir = 'C:\\Users\\Keerthan\\Desktop\\tempdataset\\test\\'

        # Check directory contents
        print("Training directory:", train_data_dir)
        print("Validation directory:", validation_data_dir)

        print("Training classes:", os.listdir(train_data_dir))
        print("Validation classes:", os.listdir(validation_data_dir))

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=45,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        batch_size = 32

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary')  # Use 'binary' for binary classification

        validation_generator = validation_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary')  # Use 'binary' for binary classification

        # Check the number of samples
        print(f"Number of training samples: {train_generator.samples}")
        print(f"Number of validation samples: {validation_generator.samples}")

        self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=8)

        print("Model Training Completed")

    def predict(self):
        def get_random_image(path):
            folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
            random_directory = np.random.randint(0, len(folders))
            path_class = folders[random_directory]
            file_path = os.path.join(path, path_class)
            file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
            random_file_index = np.random.randint(0, len(file_names))
            image_name = file_names[random_file_index]
            return cv2.imread(os.path.join(file_path, image_name)), path_class

        # Define the dictionary based on class indices
        class_indices = {'sam': 0}  # Update based on actual classes
        monkey_breeds_dict = {v: k for k, v in class_indices.items()}

        for i in range(0, 10):
            input_im, path_class = get_random_image("C:\\Users\\Keerthan\\Desktop\\tempdataset\\test\\")
            input_original = input_im.copy()
            input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            input_im = cv2.resize(input_im, (224, 224), interpolation=cv2.INTER_LINEAR)
            input_im = input_im / 255.
            input_im = input_im.reshape(1, 224, 224, 3)

            res = np.argmax(self.model.predict(input_im, 1, verbose=0), axis=1)
            predicted_class = monkey_breeds_dict.get(res[0], "Unknown")

            print(f"Predicted: {predicted_class}, Actual: {path_class}")  # Add this line for debugging

            BLACK = [0, 0, 0]
            expanded_image = cv2.copyMakeBorder(input_original, 80, 0, 0, 100, cv2.BORDER_CONSTANT, value=BLACK)
            cv2.putText(expanded_image, predicted_class, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Prediction', expanded_image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

if _name_ == "_main_":  # Fixed name check
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()