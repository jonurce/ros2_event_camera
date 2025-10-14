#%% [Capture Images]
import cv2
import os
import datetime

# Create output directory if it doesn't exist
output_dir = "Datasets/Original"
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

#%% [Split Dataset]
print("Splitting dataset...")

# %%
