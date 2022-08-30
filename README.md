![banner](https://user-images.githubusercontent.com/36157933/181859192-3a88bc30-087f-4555-8ff8-b8ad49bde38b.png)

# Handball Detection
**handball-detection** is a project which uses computer vision techniques to detect handballs in football matches.

# Setup
##### Initial Setup
1. `git clone https://github.com/nadimra/handball-detection.git` 

##### HRNet Setup
2. Go to the `project_HRNet` directory and `pip install -r requirements.txt`.
3. Download the pretrained weights for the HRNet object detector and place it within `/project_HRNet/models/detectors/yolo/weights`. We used [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights).
4. Download the pretrained weights for HRNet and place it within `/project_HRNet/weights`. We used [pose_hrnet_w48_384x288.pth](https://drive.google.com/open?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS).

##### yolo5 Setup
5. Go to the `project_yolo5` directory and `pip install -r requirements.txt`.
6. Download the pretrained weights for yolov5 and place it within `/project_yolo5/weights`. We use [yolo5s.pt](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt).

# How to use
Run the following command: (make sure to change the ``vid_path`` variable to pass a video).
```
python main.py
```
Outputs are saved in ``/project_HRNet/outputs``. If a handball occured, an additional image ``decision.png`` will show the frame of when the handball occured. 

# Snapshots
![handbal-detection-res-1](https://user-images.githubusercontent.com/36157933/187557703-11c03fb2-b028-4600-b114-c6af6a407f8b.png)
*Textual Output: The ball did hit the player’s hand. The decision is to award a handball since the ball hit the player’s left arm. The arm was at an angle of 78 degrees.*

![handbal-detection-res-2](https://user-images.githubusercontent.com/36157933/187557707-5cb105ad-17b9-4d05-91f6-a2b8fba913b1.png)
*Textual Output:The ball did hit the player’s hand.The decision is to award a handball since the ball hit the player’s right arm.The arm was at an angle of 50 degrees.*

![handbal-detection-res-3](https://user-images.githubusercontent.com/36157933/187557709-3e0931a0-a9c6-4e76-8457-fec8e4f49f09.png)
*Textual Output: The ball did hit the player’s hand. However, the decision is to not award a handball since the ball hit the player’s right arm at an angle of 38 degrees*

![handbal-detection-res-4](https://user-images.githubusercontent.com/36157933/187557711-6cf974a7-6c49-4f0b-a77c-168116af2551.png)
*Textual Output: The ball did hit the player’s hand. However, the decision is to not award a handball since the ball hit the player’s left arm at an angle of 14 degrees.*

# Acknowledgements
Our code is built on [HRNet](https://github.com/stefanopini/simple-HRNet) and [YOLOv5](https://github.com/ultralytics/yolov5). We thank the authors for sharing their codes.
