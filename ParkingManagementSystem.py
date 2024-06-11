import cv2
from ultralytics import YOLO
import numpy as np

# Define parking spots (adjust these values as needed)
parking_spots = [
    [(90, 324), (0, 410), (80, 434), (170, 326)],
    [(181, 326), (91, 434), (194, 432), (270, 326)],
    [(279, 326), (205, 434), (308, 433), (363, 325)],
    [(369, 326), (317, 431), (415, 432), (453, 324)],
    [(459, 326), (427, 431), (523, 430), (540, 326)],
    [(549, 324), (531, 430), (631, 429), (629, 325)],
    [(640, 325), (639, 429), (740, 428), (722, 324)],
    [(729, 324), (746, 428), (844, 426), (810, 325)],
    [(817, 322), (855, 427), (951, 426), (900, 321)],
    [(904, 322), (961, 428), (1059, 427), (988, 322)],
    [(995, 320), (1075, 325), (1166, 426), (1070, 426)],
    [(1083, 323), (1177, 426), (1270, 428), (1164, 322)],
    [(1175, 324), (1279, 426), (1279, 426), (1278, 322)],
]

# Colors and line thickness
COLOR_FREE = (0, 255, 0)
COLOR_OCCUPIED = (0, 0, 255)
THICKNESS = 2

# Vehicle classes in COCO 
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']

#This function loads the model 
def load_yolo_model(model_path='yolov8s.pt'):
    return YOLO(model_path)

#This function ensures that the video is open correctly
def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    return cap

#This function Create a video writer, where we want to show the results
def create_video_writer(output_path, frame_width, frame_height, fps):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

#In this function is where we do all the procces of vehicle detection and parking slots ocupation
def process_frame(frame, model, parking_spots):
    results = model(frame)
    vehicle_centers = []

    for result in results:
        for box in result.boxes:     #Here we obtain the bounding boxes from all the vehicles detected
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if model.names[cls] in VEHICLE_CLASSES:   #We are only interested in detect Vehicles and not other clases that Yolo detect
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                center_x = (x1 + x2) // 2   #We calculate the center of the bounding box
                center_y = (y1 + y2) // 2
                vehicle_centers.append((center_x, center_y))

    vehicles_in_parking = 0
    available_parking = 0

    for spot in parking_spots:
        plaza_occupied = False

        for center in vehicle_centers:  #This function analizes if one bounding box is inside the parking slot and update the counter
            if cv2.pointPolygonTest(np.array(spot), center, False) >= 0:
                plaza_occupied = True
                vehicles_in_parking += 1
                break

        color = COLOR_OCCUPIED if plaza_occupied else COLOR_FREE #Here we draw the parking slots in the video with different colour depending on the status
        available_parking += 1 if not plaza_occupied else 0
        for i in range(4):
            cv2.line(frame, spot[i], spot[(i + 1) % 4], color, THICKNESS)

    total_parking = len(parking_spots)  #Display in the video the number of free and ocupated slots
    info_text = f'Occupied: {vehicles_in_parking}, Free: {total_parking - vehicles_in_parking}'
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    return frame, vehicles_in_parking

def main():
    model = load_yolo_model()
    video_path = 'DLIP_parking_test_video.avi'
    cap = open_video(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = create_video_writer('output_video.mp4', frame_width, frame_height, fps)
    frame_number = 0

    with open('counting_result.txt', 'w') as f:  #We create countin_result.txt where we are gonna store the information of every frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, vehicles_count = process_frame(frame, model, parking_spots) #We store the frame number and the number of vehicles in parking slot
            f.write(f'{frame_number}, {vehicles_count}\n')
            
            out.write(frame)
            cv2.imshow('Frame', frame)
            
            frame_number += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()