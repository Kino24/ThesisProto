import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2  # OpenCV for HSV processing

# Load YOLO models
segment_model = YOLO("./segmentModel.pt")
classify_model = YOLO("./classifyModel.pt")

# Input and output folders
input_folder = "./Data/input"  # Folder containing images to process
output_folder = "./Data/output"  # Folder to save segmented images
os.makedirs(output_folder, exist_ok=True)

# Output text file to store classification results
results_file = "classification_results.txt"

# Define class labels (assuming 0 = Fresh, 1 = Not Fresh)
class_labels = {0: "Fresh", 1: "Not Fresh"}

# Function to segment images
def segment_image(image_path, filename):
    results = segment_model.predict(source=image_path, save=False)  # Run segmentation
    
    image = Image.open(image_path).convert("RGBA")
    image_np = np.array(image)  # Convert to NumPy array
    image_hsv = cv2.cvtColor(image_np[:, :, :3], cv2.COLOR_RGB2HSV)  # Convert to HSV
    
    original_width, original_height = image.size
    segmented_images = []
    
    for i, mask in enumerate(results[0].masks.data):
        mask_array = (mask.cpu().numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_array, mode="L").resize((original_width, original_height), Image.NEAREST)
        mask_np = np.array(mask_image)
        if len(mask_np.shape) == 2:
            mask_np = cv2.merge([mask_np, mask_np, mask_np])
        masked_hsv = cv2.bitwise_and(image_hsv, mask_np)
        filtered_hsv = cv2.inRange(masked_hsv, np.array([0, 10, 100]), np.array([160, 255, 255]))
        filtered_hsv_colored = cv2.cvtColor(filtered_hsv, cv2.COLOR_GRAY2BGR)
        segmented_image = cv2.bitwise_and(image_np[:, :, :3], filtered_hsv_colored)
        
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_seg{i}.png")
        Image.fromarray(segmented_image).save(output_path)
        segmented_images.append(output_path)
        print(f"Segmented image saved: {output_path}")
    return segmented_images

# Function to classify images
def classify_image(image_path):
    results = classify_model.predict(source=image_path, save=False)
    
    # Check if there are any detections
    if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
        prediction_idx = int(results[0].boxes.cls[0].item())  # Get the index of the detected class
        label = class_labels.get(prediction_idx, "Unknown")  # Map to Fresh/Not Fresh
        return label
    
    return "No detection"

# Run segmentation and classification on all images in the input folder
with open(results_file, "w") as file:
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            segmented_images = segment_image(image_path, filename)
            
            for seg_img in segmented_images:
                label = classify_image(seg_img)
                result = f"Image: {seg_img} -> Prediction: {label}\n"
                print(result.strip())
                file.write(result)