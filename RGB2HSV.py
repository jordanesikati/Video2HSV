import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the range of colors to consider for the heatmap (in HSV color space)
lower_color = np.array([0, 100, 100])
upper_color = np.array([179, 255, 255])

# Loop through the frames
while True:
    # Capture the frame
    ret, frame = cap.read()

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask based on the color range
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Calculate the saturation channel of the image
    saturation = hsv[:,:,0]

    # Apply the mask to the saturation channel
    saturation_masked = cv2.bitwise_and(saturation, saturation, mask=mask)

    # Normalize the saturation channel to 0-255 range
    saturation_norm = cv2.normalize(saturation_masked, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to heatmap using the "hot" colormap
    heatmap = cv2.applyColorMap(saturation_norm, cv2.COLORMAP_HOT)

    # Display the result
    cv2.imshow('Saturation Heatmap', heatmap)

    # Exit if the user presses the "q" key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
