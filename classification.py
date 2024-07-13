class PeoplePlayerClassifier:
    '''This is responsible for classifying players out of people.'''

    def __init__(self, map_image_keypoints):
        self.__map_key_points = map_image_keypoints
        self.__players_indices = list()
        self.__peoples_indices = list()

    def classify(self, people_pos):
        print("people: ", people_pos)
        self.__players_indices = []
        self.__peoples_indices = []
        (x1, y1), (x2, y2) = self.__map_key_points
        y_mid = (y1 + y2) / 2

        for index, point in enumerate(people_pos):
            x, y = point           
            # define the area where players spend most of time 
            if (x1-40 <= x <= x2+40 and y1-100 <= y <= y2+90) and not(y_mid-50 < y < y_mid+50): 
                self.__players_indices.append(index)
            else:
                if (x1 <= x <= x2 and y1 <= y <= y2):
                    self.__players_indices.append(index)
                else:
                    self.__peoples_indices.append(index)
        
        return self.__players_indices, self.__peoples_indices
    