import ultralytics
import numpy as np
import cv2

class HumanTracker:
    def __init__(self, model_path, class_no=[0]):
        self.__model = ultralytics.YOLO(model_path)
        self.__class_number = class_no
        self.__results = None

    def track(self, image):
        self.__results = self.__model.predict(image, classes=self.__class_number)
        self.__returns = []  # Initialize returns as a list

        for result in self.__results:
            for box in result.boxes:
                bbox = box.xyxy[0] if hasattr(box, 'xyxy') else None

                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    self.__returns.append(((x1, y1), (x2, y2)))

        return self.__returns


class TennisTracker:
    def __init__(self, model_path):
        self.__model = ultralytics.YOLO(model_path)
        self.__results = None

    def track(self, image):
        # do predictions with trained model for tennis balls 
        self.__results = self.__model.predict(image, conf=0.15) 
        if self.__results and len(self.__results[0].boxes) > 0:
            self.__bound = []  
            self.__confs = []  

            for result in self.__results:
                self.__confs = list(result.boxes.conf)
                self.__bound = list(result.boxes.xyxy)

            if len(self.__results) > 0:
                max_index = self.__confs.index(max(self.__confs))
                return self.__bound[max_index].tolist()
        return []  
    

class CourtTracker:
    def __init__(self, model_path):
        self.__model = ultralytics.YOLO(model_path)
        self.__image = None
        self.__results = None

    def __predict_by_model(self):
        image = self.__image
        self.__results = self.__model.predict(image, conf=0.6)

        if self.__results and len(self.__results[0].boxes) > 0:
            self.__bound = []  
            self.__confs = []  

            for result in self.__results:
                self.__confs = list(result.boxes.conf)
                self.__bound = list(result.boxes.xyxy)

            if len(self.__results) > 0:
                max_index = self.__confs.index(max(self.__confs))
                print("self.__bound[max_index].tolist()", self.__bound[max_index].tolist())
                return self.__bound[max_index].tolist()
        return [] # if no detection, then middle of the image
    
    def __crop_predicted(self, coordinates):
        coordinates = [int(x) for x in coordinates]
        x1, y1, x2, y2 = coordinates
        cropped_image = self.__image[y1-10 : y2+10, x1 : x2]

        return cropped_image
    
    def __image_processing(self, image):
        # lamda functions for calculate slope intercept        
        calculate_slope_intercept = lambda x1, y1, x2, y2: (
            (y2 - y1) / (x2 - x1 + 1e-6), y1 - ((y2 - y1) / (x2 - x1 + 1e-6)) * x1
        )
        # lamda functions for calculate average lines   
        average_lines = lambda cluster: (
            int(np.mean([line[0] for line in cluster])),
            int(np.mean([line[1] for line in cluster])),
            int(np.mean([line[2] for line in cluster])),
            int(np.mean([line[3] for line in cluster]))
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=150, minLineLength=50, maxLineGap=40)

        if lines is not None:
            clusters = []
            max_value, min_value = float('-inf'), float('inf')
            horizontal_lines = [None, None]
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)

                if not (-0.1 < angle < 0.1 or np.pi - 0.1 < angle < np.pi + 0.1):  # Ignore horizontal lines
                    slope, intercept = calculate_slope_intercept(x1, y1, x2, y2)
                    matched = False
                    for cluster in clusters:
                        cluster_slope, cluster_intercept = calculate_slope_intercept(*cluster[0])
                        if abs(slope - cluster_slope) < 0.3 and abs(intercept - cluster_intercept) < 100:
                            cluster.append(line[0])
                            matched = True
                            break
                    if not matched:
                        clusters.append([line[0]])
                else:
                    m, c = calculate_slope_intercept(x1, y1, x2, y2)
                    if c > max_value: max_value, horizontal_lines[0] = c, (x1, y1, x2, y2)
                    if c < min_value: min_value, horizontal_lines[1] = c, (x1, y1, x2, y2)

            for cluster in clusters:
                x1_avg, y1_avg, x2_avg, y2_avg = average_lines(cluster)
                m, c = calculate_slope_intercept(x1_avg, y1_avg, x2_avg, y2_avg)
                if abs(m) < 5:
                    vertical_lines.append((x1_avg, y1_avg, x2_avg, y2_avg))
                    #cv2.line(image, (x1_avg, y1_avg), (x2_avg, y2_avg), (0, 0, 255), 2)
                    #print(m, c)

            #cv2.line(image, (horizontal_lines[0][0], horizontal_lines[0][1]), (horizontal_lines[0][2], horizontal_lines[0][3]), (0, 0, 255), 2)
            #cv2.line(image, (horizontal_lines[1][0], horizontal_lines[1][1]), (horizontal_lines[1][2], horizontal_lines[1][3]), (0, 0, 255), 2)

        #cv2.imshow('Edges', edges)
        #cv2.imshow('Detected Lines', image)
        #cv2.waitKey(0)

        return vertical_lines, horizontal_lines
    
    def __find_keypoints(self, h_lines, v_lines):
        # Lambda function to calculate slope and intercept
        calculate_slope_intercept = lambda x1, y1, x2, y2: (
            (y2 - y1) / (x2 - x1 + 1e-6), y1 - ((y2 - y1) / (x2 - x1 + 1e-6)) * x1
        )
        keypoints = []

        for h_line in h_lines:
            x1, y1, x2, y2 = h_line
            h_slope, h_intercept = calculate_slope_intercept(x1, y1, x2, y2)
            for v_line in v_lines:
                x3, y3, x4, y4 = v_line
                v_slope, v_intercept = calculate_slope_intercept(x3, y3, x4, y4)

                if np.isinf(v_slope):
                    x_intersect = x3
                    y_intersect = h_slope * x_intersect + h_intercept
                else:
                    if h_slope != v_slope:
                        x_intersect = (v_intercept - h_intercept) / (h_slope - v_slope)
                        y_intersect = h_slope * x_intersect + h_intercept
                    else:
                        continue  # Skip parallel lines
                keypoints.append((int(x_intersect), int(y_intersect)))
        return keypoints
    
    def __find_boundary_corners(self, keypoints):
        key_p_1 = keypoints[:len(keypoints) // 2]
        key_p_2 = keypoints[len(keypoints) // 2:]

        key_p_1.sort(key=lambda point: point[0])
        A = key_p_1[0]
        B = key_p_1[-1]

        key_p_2.sort(key=lambda point: point[0])
        C = key_p_2[0]
        D = key_p_2[-1]

        return A, B, C, D


    
    def track(self, image):
        try:
            self.__image = image
            coord = self.__predict_by_model()
            fimage = self.__crop_predicted(coord)
            vertical_lines, horizontal_lines = self.__image_processing(fimage)
            key_points = self.__find_keypoints(horizontal_lines, vertical_lines)

            # converting coordinates of key points from cropped image coordinate system to self.__image 's coordinate system
            key_points = [(int(x + coord[0]), int(y + coord[1] - 10)) for x, y in key_points]
            boundary_corners = self.__find_boundary_corners(key_points)
            #print("key points: ", boundary_corners)

            #cv2.imshow('Detected Lines', self.__image)
            #cv2.waitKey(0)
        except:
            boundary_corners = []

        return boundary_corners
