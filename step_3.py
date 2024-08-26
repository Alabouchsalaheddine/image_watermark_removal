import os
import glob
import numpy as np
from PIL import Image
import sys
from src.core import process_inpaint
input_folder = '../Images'
def process_images(input_folder, mask_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all JPG files in the input folder
    for file_path in glob.glob(os.path.join(input_folder, '*.jpg')):
        try :
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Construct the paths for mask and output
            mask_path = os.path.join(mask_folder, f"{base_name}.png")
            output_path = os.path.join(output_folder, f"{base_name}.jpg")

            # Check if mask file exists
            if not os.path.exists(mask_path):
                print(f"Mask file not found: {mask_path}")
                continue

            # Open the images
            img_input = Image.open(file_path).convert("RGBA")
            result_image = Image.open(mask_path).convert("RGBA")

            # Process the images
            output = process_inpaint(np.array(img_input), np.array(result_image))

            # Save the result
            img_output = Image.fromarray(output).convert("RGB")
            img_output.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e :
            print("error")

# Define folders

mask_folder = '../masks'
output_folder = '../Processed_Images'

# Process all images
process_images(input_folder, mask_folder, output_folder)
print("Execution finished")
