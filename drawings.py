import cv2
import numpy as np

class Drawing:
    def __init__(self):
        self.__player_color = (255, 50, 0)
        self.__people_color = (255, 255, 0)
        self.__tennis_color = (0, 100, 255)
        self.__court_color = (0, 0, 255)

    def __draw_player(self, image, bounding_boxes):
        for (bb1, bb2) in bounding_boxes:
            image = cv2.rectangle(image, bb1, bb2, self.__player_color, thickness=2)
            label = f"Player"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            label_position = (bb1[0], bb1[1] - label_size[1] - 5)  # Adjusted to be above the bounding box
            label_background_top_left = label_position
            label_background_bottom_right = (label_position[0] + label_size[0], label_position[1] + label_size[1] + 5)
            image = cv2.rectangle(image, label_background_top_left, label_background_bottom_right, (255, 0, 0), cv2.FILLED)
                
            text_position = (bb1[0], bb1[1] - 5)
            image = cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image
    
    def __draw_people(self, image, bounding_boxes):
        for (bb1, bb2) in bounding_boxes:
            image = cv2.rectangle(image, bb1, bb2, self.__people_color, thickness=2)

        return image
    
    def __draw_tennis(self, image, bounding_box):
        if len(bounding_box) != 0:
            bb1 = (int(bounding_box[0]), int(bounding_box[1]))
            bb2 = (int(bounding_box[2]), int(bounding_box[3]))
                
            image = cv2.rectangle(image, bb1, bb2, self.__tennis_color, thickness=2)
            label = "Tennis ball"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            label_position = (bb1[0], bb1[1] - label_size[1] - 5)  # Adjusted to be above the bounding box

            label_background_top_left = label_position
            label_background_bottom_right = (label_position[0] + label_size[0], label_position[1] + label_size[1] + 5)
            image = cv2.rectangle(image, label_background_top_left, label_background_bottom_right, self.__tennis_color, cv2.FILLED)
                
            text_position = (bb1[0], bb1[1] - 5)
            image = cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image
    

    def __draw_court(self, image, coordinates):
        if len(coordinates) != 0:
            pts = np.array(coordinates, np.int32)
            pts = np.array(sorted(pts, key=lambda x: x[0]))
            pts = pts.reshape((-1, 1, 2))
            image = cv2.polylines(image, [pts], isClosed=True, color=self.__court_color, thickness=2)

            position = coordinates[2][0], coordinates[2][1] - 10
            cv2.rectangle(image, (position[0], position[1] - 16), (position[0]+140, position[1]+10), self.__court_color, -1)
            image = cv2.putText(image, "Tennis Court", position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image
    

    def draw_all_obj_image(self, image, player_boxes, people_boxes, tennis_box, key_points):
        '''drawing on object detection image'''
        image = self.__draw_court(image, key_points)
        image = self.__draw_people(image, people_boxes)
        image = self.__draw_player(image, player_boxes)
        image = self.__draw_tennis(image, tennis_box)

        return image
    
    def draw_map_image(self, players, people, tennis, image_path):
        '''drawing on object detection mapping image'''
        image = cv2.imread(image_path)
        if people is not None:
            for coord in people:
                cv2.circle(image, coord, radius=5, color=self.__people_color, thickness=-1)
                cv2.circle(image, coord, radius=7, color=(255, 255, 255), thickness=1)
        if players is not None:
            for coord in players:
                cv2.circle(image, coord, radius=7, color=self.__player_color, thickness=-1)
                cv2.circle(image, coord, radius=9, color=(255, 255, 255), thickness=1)
        if tennis is not None:
            pass
            cv2.circle(image, tennis, radius=5, color=self.__tennis_color, thickness=-1)
            cv2.circle(image, tennis, radius=7, color=(255, 255, 255), thickness=1)

        return image

