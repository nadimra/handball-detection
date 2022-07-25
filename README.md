![banner](https://user-images.githubusercontent.com/36157933/180853192-bb67c9e8-86c4-4f86-a090-74d9ec6b6156.png)

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

# Acknowledgements
Our code is built on [HRNet](https://github.com/stefanopini/simple-HRNet) and [YOLOv5](https://github.com/ultralytics/yolov5). We thank the authors for sharing their codes.
