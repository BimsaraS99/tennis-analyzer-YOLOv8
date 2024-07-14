# Tennis Play Analyzer with YOLOv8

## Project Overview
The Tennis Match Analyzer is a computer vision project that tracks and analyzes tennis matches from video input. It uses advanced object detection and tracking techniques to identify players, the tennis ball, and court boundaries, mapping these elements onto a 2D court representation.

https://github.com/user-attachments/assets/8cdceb9e-8fcb-48a0-a2b9-20ff969afdc0

## Features
- Player detection and tracking
- Tennis ball tracking
- Court boundary detection
- Real-time mapping of players and ball positions to a 2D court view
- Video processing with side-by-side display of detection results and 2D mapping

## Technologies Used
- Python
- OpenCV
- YOLOv8 (for object detection)
- NumPy
- Scikit-learn (for K-means clustering)

## Installation

1. Clone the repository:

git clone [https://github.com/your-username/tennis-match-analyzer.git](https://github.com/BimsaraS99/tennis-analyzer-YOLOv8)

2. Install the required dependencies:

pip install -r requirements.txt


## Usage

1. Ensure you have the trained models in the `trained_models` directory:
- Tennis ball model: `trained_models/tennis_ball_model/best.pt`
- Human detection model: `trained_models/yolo_model/yolov8x.pt`
- Court detection model: `trained_models/court_model/last.pt`

2. Place your input video in the `inputs/videos/` directory.

3. Run the main script:

python main.py

4. The processed video will be saved as `runs/detect/processed_tennis_match.mp4`.

## Project Structure
- `main.py`: The main script that orchestrates the entire process
- `yolo_interference.py`: Contains classes for tennis ball, human, and court tracking
- `mapping.py`: Handles the mapping from camera view to 2D court view
- `classification.py`: Classifies detected people as players or spectators
- `drawings.py`: Handles all the drawing functions for visualization

## Dependencies
- numpy==1.21.0
- opencv-python==4.5.3.56
- ultralytics==8.0.0

## Used Datasets
- Tennis bal detection : https://universe.roboflow.com/tennisball-3eqxr/tennis-ball-detection-qaxae/dataset/1
- Tennis Court detection : https://universe.roboflow.com/tennistracker-dogbm/tennis-court-detection

## Contributing
Contributions to improve the Tennis Match Analyzer are welcome. Please feel free to submit a Pull Request.

## License
[MIT License](https://opensource.org/licenses/MIT)

## Acknowledgements
- YOLOv8 by Ultralytics
- OpenCV community

## Future Improvements
- Implement player identification
- Add match statistics tracking
- Improve accuracy of ball tracking in high-speed scenarios

## Contact
For any queries or suggestions, please open an issue in this repository.
