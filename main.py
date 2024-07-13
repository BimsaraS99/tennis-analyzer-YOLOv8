import cv2
from yolo_interference import TennisTracker, HumanTracker, CourtTracker
from mapping import CameraViewToPlan
from classification import PeoplePlayerClassifier
from drawings import Drawing

# Create instances from classes
tennis_detector = TennisTracker('trained_models/tennis_ball_model/best.pt')
human_detector = HumanTracker('trained_models/yolo_model/yolov8x.pt')
keypoint_detector = CourtTracker('trained_models/court_model/last.pt') # detect key points on the tennis court

mapping_to_plan = CameraViewToPlan(
    ref_points=[(163, 657), (376, 658), (163, 195), (376, 195)] # mapping image key points
)

pep_ply_classifier = PeoplePlayerClassifier(
    map_image_keypoints=((163, 195), (376, 658)) # mapping image key points (bounding box)
)

draw = Drawing()

video_path = 'inputs/videos/B_video.mov'  
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('runs/detect/processed_tennis_match.mp4', fourcc, 24.0, (1500, 640))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        frame = cv2.resize(frame, (1280, 720))

        people = human_detector.track(frame)
        tennis_ball = tennis_detector.track(frame)
        key_points = keypoint_detector.track(frame)

        tennis_pos, people_pos = mapping_to_plan.map(key_points, people, tennis_ball)
        player_index, people_index = pep_ply_classifier.classify(people_pos)
        print("player ", player_index)

        players = [people[i] for i in player_index if i < len(people)]
        people_filtered = [people[i] for i in people_index if i < len(people)]
        player_map = [people_pos[i] for i in player_index if i < len(people_pos)]
        people_map = [people_pos[i] for i in people_index if i < len(people_pos)]

        object_detected_image = draw.draw_all_obj_image(frame, players, people_filtered, tennis_ball, key_points)
        mapped_image = draw.draw_map_image(player_map, people_map, tennis_pos, image_path='inputs/tennis_court_image/img.png')
        mapped_image = cv2.resize(mapped_image, (int(mapped_image.shape[1] * (720 / mapped_image.shape[0])), 720))
        
        combined_frame = cv2.hconcat([object_detected_image, mapped_image])
        combined_frame = cv2.resize(combined_frame, (1500, 640))

        out.write(combined_frame)

        cv2.imshow("Processed Frame", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

cap.release()
out.release()
cv2.destroyAllWindows()
