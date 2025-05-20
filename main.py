import cv2 as cv 
# We are importing cv2 and giving it a shorthand called cv
import mediapipe as mp
#We are importing mediapipe and giving it a shorthand called mp
mp_drawing = mp.solutions.drawing_utils
#Contains tools to draw landmarks on the body
mp_pose = mp.solutions.pose
#The module for detecting human body pose(shoulders,spine,etc.)
cap = cv.VideoCapture(0)
#Opens your default webcam
#0 Means the first camera device
with mp_pose.Pose(min_detection_confidence = 0.5,
                  min_tracking_confidence = 0.5) as pose:
#Creates a pose estimation object using MediaPipe
#min_detection_confidence = 0.5: How confident it must be to detect a person
#Explanation for the selection of 0.5:
#0.0 Accepts everything (Even very uncertain detections)
#1.0 only accepts detections that are 100% confident. 
#min_tracking_confidence = 0.5: How confident it must be to keep tracking landmarks from frame
#frame
    while cap.isOpened():
#Starts a loop to keep reading from the webcam, as long as it is open.
        success,frame = cap.read()
        #.read() is a method that grabs the next video frame from the camera and returns two values:
        #success: a boolean that will be true if a frame was successfully read, or false otherwise
        #frame: the actual image captured from the webcame as a NumPy array 
        if not success:
            print("Ignoring empty frame")
            continue
#Cap.read() grabs a frame from the webcame
#If it fails, we will skip that frame. 
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
#In a variable called image, we are storing the converted color version of the frame.
#cvt.Color() is an OpenCV function that stands for "convert color" | It is used to change the color space of an image (e.g. BGR -> RGB)
#frame | The input image to convert. The image just grabbed from the webcam using line 23.
#By default, OpenCV will give you the image in BGR format (Blue-Green-Red) not the usual RGB
#cv.COLOR_BGR2RGB is a constant defined by OpenCV, that tells cvt to convert from BGR color format to RGB.
#.flags is a special attribute of NumPY arrays. It gives you access to various internal memory settings and properties of the array such
# as: whether it is writeable, whether it is aligned in memory, whether it is contiguous 
        results = pose.process(image) #Sends the image to the pose model. results contains all of the landmarks(shoulders, knees, etc.)
        # .landmark gives you a list of 33 landmark objects, each with x, y, z, and visibility
        
        image.flags.writeable = True  #Sets the image to writeable again so that we can draw on it. 
        image = cv.cvtColor(image,cv.COLOR_RGB2BGR) #Convert back to BGR because OpenCV needs that format to display.
        if results.pose_landmarks: #Conditional Statement. we are checking if the pose landmarks exist in the results object. 
            landmarks = results.pose_landmarks.landmark #Get the full list of landmark points and store it in a variable called landmarks.

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] #Get the landmark object for the left shoulder by its list index in landmarks, and store
            #it in the variable called left_shoulder. 
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] #Get the landmark object for the right shoulder by its list index in landmarks, and store 
            #it in the variable called right_shoulder. 
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP] #Get the landmark object for the left hip by its list index in landmarks, and store it in the variable
            #called left_hip.
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP] #Get the landmark object for the right hip by its list index in landmarks, and store it in the variable
            #called right_hip.

            #print the x,y,and z values of the shoulders and hips with a label. 

            #f"text": formats your floating point value to 2 decimal using .2f (can change if wanted)
            #(30, 30)(30, 50), etc: is the x and y coordinate of the text on the screen (can be changed)
            #cv.FONT_HERSHEY_SIMPLEX is just a font
            #0.5 is the font scale (can make this bigger or smaller)
            #(0,255,0) green color (can be changed)
            # 1 is the line thickness
            cv.putText(image, f"Left Shoulder: {left_shoulder.x:.2f}, {left_shoulder.y:.2f}, {left_shoulder.z:.2f}",(30,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
            cv.putText(image, f"Right Shoulder: {right_shoulder.x:.2f}, {right_shoulder.y:.2f}, {right_shoulder.z:2f}",(30,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
            cv.putText(image, f"Left Hip: {left_hip.x:.2f}, {left_hip.y:.2f}, {left_hip.z:.2f}",(30,70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
            cv.putText(image, f"Right Hip: {right_hip.x:.2f}, {right_hip.y:.2f}, {right_hip.z:.2f}",(30,90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

            print("Left shoulder:", left_shoulder.x, left_shoulder.y, left_shoulder.z)
            print("Right shoulder:", right_shoulder.x, right_shoulder.y, right_shoulder.z)
            print("Left hip:", left_hip.x, left_hip.y, left_hip.z)
            print("Right hip:", right_hip.x, right_hip.y, right_hip.z)


            mp_drawing.draw_landmarks(
                image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Display the image 
        cv.imshow('Posture Detection',image)

        if cv.waitKey(5) & 0xFF == 27: #Press escape to exit 
            break
cap.release()
cv.destroyAllWindows()
            



