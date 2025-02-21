import os
import SegmentAndClassify  # Import the segmentation and classification module

def process_images_in_folder(input_folder, output_folder):
    labels = []
    
    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            
            # Run segmentation and classification
            segmented_images = SegmentAndClassify.segment_image(image_path, filename)
            
            # Get classification labels for each segmented image
            labels.extend([SegmentAndClassify.classify_image(seg_img) for seg_img in segmented_images])
    
    # Delete all images in the input folder after processing
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Delete all images in the output folder after processing
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    return labels