from ultralytics import YOLO
import cv2

import os



# Helper functions

def manual_label_change(original_label, names_dict):
    new_label = "Undefined" 

    if original_label == 'person':
        new_label = 'player'  # Example change
    else:
        new_label = original_label

    return new_label  # Return None or an appropriate default if the new label doesn't exist



#load the model

model = YOLO('yolov8n.pt')

# load the video
video_path = os.path.join(os.path.dirname(__file__), 'videos', 'shohei_pitch.mp4')

cap = cv2.VideoCapture(video_path)


# read frames


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    # detect and track objects at the same time
    # persist parameter allows the model to track the objects found in the video
    results = model.track(frame, persist = True)

    for result in results: # iterate through each results object in the list
        for det in result.boxes: # now accessing the boxes for each detection
            x1, y1, x2, y2 = det.xyxy[0].tolist() # get the coordinates of the bounding box
            original_class_id = int(det.cls) # Assuming det.cls to be the original class ID
            original_label = result.names[original_class_id] # Get original label
            # Manually determine the new label (custom label logic goes here)
            new_label = manual_label_change(original_label, result.names) #Function to get new label


            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Calculate the position for the custom label to be at the top of the box
            label_position = (int(x1), int(y1) - 10)

            # Add the custom label above the bounding box
            cv2.putText(frame, new_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)




    # Display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
