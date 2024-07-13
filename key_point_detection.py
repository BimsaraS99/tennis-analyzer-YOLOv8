import cv2
import numpy as np
from sklearn.cluster import KMeans

class KeypointsDetector:
    def __init__(self, min_contour_area=500, sides=4):
        self._min_contour_area = min_contour_area
        self._n_clusters = sides
        self._lower_white = np.array([0, 0, 170])
        self._upper_white = np.array([255, 255, 255])
        self._kernel = np.ones((5, 5), np.uint8)

        self._kmeans_model = KMeans(n_clusters = self._n_clusters) # machine learning model for clustering lines

        self._calculate_line_equation = lambda x1, y1, x2, y2: \
            (None, x1) if x1 == x2 else ((y2 - y1) / (x2 - x1), y1 - ((y2 - y1) / (x2 - x1)) * x1) # lambda priavte functions
        
        self._find_intersection = lambda m1, b1, m2, b2: \
            None if m1 == m2 else ((b2 - b1) / (m1 - m2), m1 * ((b2 - b1) / (m1 - m2)) + b1)
        

    def _process_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower_white, self._upper_white)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self._min_contour_area]
        contour_image = cv2.drawContours(np.zeros_like(image), filtered_contours, -1, (255, 255, 255), 2) # check this one

        contour_image = cv2.dilate(contour_image, self._kernel, iterations=10)
        contour_image = cv2.erode(contour_image, self._kernel, iterations=8)
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
        _, contour_image = cv2.threshold(contour_image, 127, 255, cv2.THRESH_BINARY) # check this one

        contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        result = cv2.drawContours(np.zeros_like(image), [largest_contour], -1, (255, 255, 255), 2)

        cv2.imshow("WTFG", result)

        return result
    
    def _line_equations(self, image):
        image = self._hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=150, maxLineGap=200)
        line_equations = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                m, b = self._calculate_line_equation(x1, y1, x2, y2)
                if m is not None:
                    line_equations.append([m, b])

            print("number of lines: ", len(line_equations))
            return line_equations
        else:
            return None


    def _side_equations(self, lines):
        '''finds the equations of the each side of the court'''
        line_equations = np.array(lines)
        self._kmeans_model.fit(line_equations)
        clustered_lines = self._kmeans_model.cluster_centers_
        clustered_lines = clustered_lines.tolist()

        print("clustered lines", clustered_lines)

        return clustered_lines
    

    def _intersections(self, side_lines):
        '''calculating key points'''
        intersections = []
        n_lines = len(side_lines)
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                m1, b1 = side_lines[i]
                m2, b2 = side_lines[j]
                intersection = self._find_intersection(m1, b1, m2, b2)
                if intersection is not None:
                    if intersection[0] > 0 and intersection[1] > 0:
                        intersections.append(intersection)

        return intersections
    

    def find_keypoints(self, image):
        processed_image = self._process_image(image)
        line_eqs = self._line_equations(processed_image)
        side_eqs = self._side_equations(line_eqs)
        key_points = self._intersections(side_eqs)
        int_key_points = [(int(x), int(y)) for x, y in key_points]

        return int_key_points
