import cv2
# Open default camera (usually 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Read frame-by-frame
    ret, frame = cap.read()

    # If frame is not read correctly, exit
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize frame 
    frame = cv2.resize(frame, (640, 480))

    # Create a red overlay (same size as frame)
    red_overlay = frame.copy()
    red_overlay[:] = (0,0,255) # BGR for Red

    # Blend the overlay with the original frame
    blended = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)

    #Show the result
    cv2.imshow('Red Hue Webcam', blended)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
