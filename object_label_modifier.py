from ultralytics import YOLO
import cv2
import os



# Helper functions

def get_custom_label(original_label):
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

processed_people = {}

cap = cv2.VideoCapture(video_path)


# read frames


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    # detect and track objects at the same time
    # persist parameter allows the model to track the objects found in the video
    results = model.track(frame, persist = True)

    for result in results:
        for det in result.boxes:
            object_id = int(det.id.item())
            
            if object_id not in processed_people:
                custom_label = get_custom_label(result.names[int(det.cls)])
                processed_people[object_id] = custom_label
            else:
                custom_label = processed_people[object_id]

            x1, y1, x2, y2 = det.xyxy[0].tolist() # get the coordinates of the bounding box
            
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Calculate the position for the custom label to be at the top of the box
            label_position = (int(x1), int(y1) - 10)

            # Add the custom label above the bounding box
            cv2.putText(frame, custom_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    
    # Display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


for object_id, label in processed_people.items():
    print(f"object_id: {object_id}, label: {label}")

