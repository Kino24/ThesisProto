import sys
import cv2
import os
import getPrediction
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("mainWindow.ui", self)  # Load the UI file
        
        # Camera label (from UI)
        self.camera_label = self.findChild(QtWidgets.QLabel, "label")
        print(f'Camera Label Found: {self.camera_label}')  # Debugging print statement
        # Prediction label (from UI)
        self.prediction_label = self.findChild(QtWidgets.QLabel, "predictionLabel")
        # Capture button (from UI)
        self.capture_button = self.findChild(QtWidgets.QPushButton, "capPhoto")
        
        # Connect button to capture function
        self.capture_button.clicked.connect(self.capture_image)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)  # Open default camera (0)
        
        # Timer to update camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)  # Refresh every 30ms

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimage = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qimage))

    def capture_image(self):
        # Define save path
        save_path = "./Data/input/captured_image.jpg"
        
        # Capture frame and save it
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(save_path, frame)
            print(f"Image saved: {save_path}")
            
            # Process the saved image
            self.process_image()

    def process_image(self):
        labels = getPrediction.process_images_in_folder("./Data/input/", "./Data/output/")
        if labels:
            self.prediction_label.setText(f"{labels[0]}")
        else:
            self.prediction_label.setText("N/A")

    def closeEvent(self, event):
        self.cap.release()  # Release the camera properly
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
