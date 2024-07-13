import cv2
import numpy as np

class CameraViewToPlan: # camera view to plan view
    def __init__(self, ref_points):
        self.__ref_points = ref_points
        self._homography_matrix = None

        self.__tennis_position = None
        self.__people_positions = []

    def __find_homography_matrix(self, plan_points):
        '''
        Calculate the homography matrix representing the relationship between 
        the camera view and the planar view.
        '''
        camera_points = np.array(list(plan_points), dtype=np.float32)  # Points in camera view
        planar_points = np.array(self.__ref_points, dtype=np.float32)  # Corresponding points in planar view
        self._homography_matrix, _ = cv2.findHomography(camera_points, planar_points)

        return self._homography_matrix
    
    def __coordinate_conversion(self, coordinate):
        camera_coordinate = np.array([[coordinate[0]], [coordinate[1]], [1]])
        planar_coordinate = np.dot(self._homography_matrix, camera_coordinate)
        planar_coordinate = planar_coordinate[:2] / planar_coordinate[2]
        x_planar, y_planar = planar_coordinate.flatten()

        return x_planar, y_planar
    
    def __people_position_convertion(self, people_pos):
        '''convert people position from camera to plan'''
        people_pos_list = []
        for box in people_pos:
            bottom_center = ((box[0][0] + box[1][0]) // 2, box[1][1])
            x, y = self.__coordinate_conversion(bottom_center)
            self.__people_positions.append((int(x), int(y)))
        return self.__people_positions

    def __tennisball_position_convertion(self, ball_pos):
        try:
            mid = int((ball_pos[0] + ball_pos[2]) / 2), int((ball_pos[1] + ball_pos[3]) / 2)
            x, y = self.__coordinate_conversion(mid)
            self.__tennis_position = (int(x), int(y))
            return self.__tennis_position
        except:
            return self.__tennis_position

    def map(self, points_cam, people_pos, tennisball_pos): # pos mean postion
        self.__tennis_position = None # reset
        self.__people_positions = [] # reset 

        self.__find_homography_matrix(points_cam)
        self.__tennisball_position_convertion(tennisball_pos)
        self.__people_position_convertion(people_pos)

        return self.__tennis_position, self.__people_positions
