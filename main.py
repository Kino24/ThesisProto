import sys
import cv2
import os
import getPrediction
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from picamera2 import Picamera2

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("mainWindow.ui", self)  # Load the UI file
        
        # Camera label (from UI)
        self.camera_label = self.findChild(QtWidgets.QLabel, "cameraLabel")
        # Prediction label (from UI)
        self.prediction_label = self.findChild(QtWidgets.QLabel, "predictionLabel")
        # Capture button (from UI)
        self.capture_button = self.findChild(QtWidgets.QPushButton, "captureButton")
        
        # Connect button to capture function
        self.capture_button.clicked.connect(self.capture_image)
        
        # Initialize Picamera2
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration())
        self.picam2.start()
        
        # Timer to update camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)  # Refresh every 30ms

    def update_camera(self):
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimage = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qimage))

    def capture_image(self):
        # Define save path
        save_path = "backend/Data/input/captured_image.jpg"
        
        # Capture frame and save it
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, frame)
        print(f"Image saved: {save_path}")
        
        # Process the saved image
        self.process_image()

    def process_image(self):
        labels = getPrediction.process_images_in_folder("backend/Data/input/", "backend/Data/output/")
        if labels:
            self.prediction_label.setText(f"Prediction: {labels[0]}")
        else:
            self.prediction_label.setText("No prediction available")

    def closeEvent(self, event):
        self.picam2.close()  # Release the camera properly
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
