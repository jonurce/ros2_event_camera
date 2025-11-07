
#%% [0 - Capture Images]
import cv2
import os
import datetime

# Create output directory if it doesn't exist
output_dir = "Datasets/Example/Images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the camera (try different indices if needed)
# index = 2 for depth, 4 for RGB
camera_index = 4  # Start with 0, may need to try 1, 2, etc.
cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2) # cv2.CAP_DSHOW / cv2.CAP_V4L2

# Set desired resolution and frame rate (may not always work due to driver limitations)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if the camera opened successfully
if not cap.isOpened():
    print(f"Error: Could not open camera at index {camera_index}")
    exit()

# Get actual resolution and frame rate
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera opened at {width}x{height} @ {fps} FPS")

try:
    print("Camera started. Press SPACE to capture an image, 'q' to quit.")
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Display the frame
        cv2.imshow('D435i RGB Feed', frame)

        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spacebar to capture
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            filename = os.path.join(output_dir, f'image_{timestamp}.jpg')

            # Save the image
            cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"Image saved: {filename}")

        elif key == ord('q'):  # 'q' to quit
            print("Exiting...")
            break

finally:
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


#%% [1 - Check Labels]
import cv2
import os
import numpy as np

# Paths
image_dir = "Datasets/Example/Images"
annotation_dir = "Datasets/Example/Annotations"  # class_id, x_center, y_center, box_width, box_height
classes = ["Boat"]  # First coordinate of annotations: 0 for boat

# Get list of images
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Check all images
for image_file in image_files: 
    # Load image
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Load annotation
    annotation_path = os.path.join(annotation_dir, image_file.replace(".jpg", ".txt"))
    if not os.path.exists(annotation_path):
        print(f"No annotation for {image_file}")
        continue

    with open(annotation_path, "r") as f:
        lines = f.readlines()

    # Draw bounding boxes
    for line in lines:
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())
        class_id = int(class_id)
        class_name = classes[class_id]

        # Convert normalized coordinates to pixel values
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        # Calculate top-left and bottom-right corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display image
    cv2.imshow("Annotation Check", image)
    print(f"Showing {image_file}. Press any key to continue, 'q' to quit.")
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

#%% [2 - Split Dataset]
import os
import shutil
import random

# Paths
image_dir = "Datasets/Example/Images"
annotation_dir = "Datasets/Example/Annotations" # class_id, x_center, y_center, box_width, box_height
dataset_dir = "Datasets/Example/Split"
splits = {"train": 0.8, "val": 0.1, "test": 0.1}  # 80/10/10 split

# Create dataset directories
for split in splits:
    os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
random.shuffle(image_files)  # Randomize for unbiased split

# Calculate split sizes
total_images = len(image_files)
train_size = int(splits["train"] * total_images)
val_size = int(splits["val"] * total_images)
test_size = total_images - train_size - val_size

# Assign images to splits
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

# Copy files to respective directories
for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
    for image_file in files:
        # Copy image
        src_image_path = os.path.join(image_dir, image_file)
        dst_image_path = os.path.join(dataset_dir, split, "images", image_file)
        shutil.copy(src_image_path, dst_image_path)

        # Copy annotation
        annotation_file = image_file.replace(".jpg", ".txt")
        src_annotation_path = os.path.join(annotation_dir, annotation_file)
        dst_annotation_path = os.path.join(dataset_dir, split, "labels", annotation_file)
        if os.path.exists(src_annotation_path):
            shutil.copy(src_annotation_path, dst_annotation_path)
        else:
            print(f"Warning: No annotation for {image_file}")

print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

#%% [3 - Validate Dataset]
import os

def validate_split(split, dataset_dir):
    image_dir = os.path.join(dataset_dir, split, "images")
    label_dir = os.path.join(dataset_dir, split, "labels")

    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    labels = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    print(f"{split.capitalize()} split: {len(images)} images, {len(labels)} labels")

    # Check for missing labels
    for image in images:
        label = image.replace(".jpg", ".txt")
        if label not in labels:
            print(f"Missing label for {image}")

    # Check for orphaned labels
    for label in labels:
        image = label.replace(".txt", ".jpg")
        if image not in images:
            print(f"Missing image for {label}")


# Validate all splits
dataset_dir = "Datasets/Example/Split"
for split in ["train", "val", "test"]:
    validate_split(split, dataset_dir)
    
#%% [4 - Train Model]

#%% [5 - Test Model]

#%% [6 - Real Time Inference]
