import os
import numpy as np
from PIL import Image, ImageDraw
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov7'))
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
import sys


# Load YOLOv7 model
device = 'cuda'
weights = './best.pt'
model = attempt_load(weights, map_location=device)

# Class names
class_names = ['box', 'wm']

def detect_watermark(image_path):
    image = Image.open(image_path)
    image = image.resize((640, 640))
    image_tensor = torch.from_numpy(np.array(image)).to(device).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        pred = model(image_tensor)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    detected_classes = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(image_tensor.shape[2:], det[:, :4], image.size).round()
            detected_classes.extend([class_names[int(cls)] for *_, cls in det])
    # get the bbox of the photometre
    bboxes = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(image_tensor.shape[2:], det[:, :4], image.size).round()
            for *xyxy, conf, cls in det:
                xyxy = [int(round(coord.item())) for coord in xyxy]
                bboxes.append(xyxy)
    # Check if no bounding boxes were found
    if not bboxes:
        print(f"Photometre not detected in {image_path}")
        return False, []  # Return an empty list of bounding boxes
    return 'watermark' in detected_classes, bboxes

def save_image_without_bbox(original_image, bboxes, output_path):
    # Load original image dimensions
    w_orig, h_orig = original_image.size



    canvas = Image.new("RGBA", (w_orig, h_orig), (0, 0, 0, 255))
    draw = ImageDraw.Draw(canvas)
    # Scale bounding box coordinates from resized (640x640) to original image dimensions
    for bbox in bboxes:
        scaled_bbox = [
            int(round(bbox[0] * w_orig / 640)),
            int(round(bbox[1] * h_orig / 640)),
            int(round(bbox[2] * w_orig / 640)),
            int(round(bbox[3] * h_orig / 640))
        ]
        draw.rectangle(scaled_bbox, fill=(255, 0, 255, 255))

    canvas_array = np.array(canvas)
    background = np.where(
        (canvas_array[:, :, 0] == 0) &
        (canvas_array[:, :, 1] == 0) &
        (canvas_array[:, :, 2] == 0)
    )
    drawing = np.where(
        (canvas_array[:, :, 0] == 255) &
        (canvas_array[:, :, 1] == 0) &
        (canvas_array[:, :, 2] == 255)
    )
    # Modify the pixels according to the mask
    canvas_array[background] = [0, 0, 0, 255]  # Black background, fully opaque
    canvas_array[drawing] = [0, 0, 0, 0]       # Transparent drawing (masked area)

    # Convert the NumPy array back to an Image object
    result_image = Image.fromarray(canvas_array)
    output_path = output_path.replace("jpg", "png")
    # Save or display the result
    result_image.save(output_path)

    print(f"Image saved to {output_path}")





def process_images(source_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            image_path = os.path.join(source_folder, filename)
            _, bboxes = detect_watermark(image_path)

            if not bboxes:
                # Handle case when no bounding boxes are found
                # For example, log the filename and skip to the next image
                print(f"Skipping {filename} as no watermark detected.")
                continue  # Skip this image and move to the next one

            original_image = Image.open(image_path).convert('RGB')
            base_name, ext = os.path.splitext(filename)
            output_filename = filename
            output_path = os.path.join(dest_folder, output_filename)

            save_image_without_bbox(original_image, bboxes, output_path)

# Example usage
source_folder = './Images'
dest_folder = './masks'
process_images(source_folder, dest_folder)
