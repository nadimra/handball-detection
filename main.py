import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
#sys.path.append(os.path.dirname(__file__))
#sys.path.append(os.path.join(os.path.dirname(__file__),'project_HRNet'))
#sys.path.append(os.path.join(os.path.dirname(__file__),'project_yolo5'))

import project_yolo5.detect_simple
import project_HRNet.scripts.live_demo
import pickle


def handball_detection(vid_path,out_path):
    direction_changers,direction_changers_position,direction_changers_frames = project_yolo5.detect_simple.run(source=vid_path)

    # Saving the objects:
    #with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #    pickle.dump([direction_changers,direction_changers_position,direction_changers_frames], f)

    #with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
    #    direction_changers,direction_changers_position,direction_changers_frames = pickle.load(f)

    """
    direction_changers = [direction_changers[4]]
    direction_changers_position = [direction_changers_position[4]]
    direction_changers_frames = [direction_changers_frames[4]]
    print(direction_changers,direction_changers_position,direction_changers_frames)

    print(direction_changers_position[0][0])
    print(direction_changers_position[0][1])
    print(direction_changers_position[0][2])
    print(direction_changers_position[0][3])
    """
    hit_hand,handball_decision,handball_part,handball_angle,msg = project_HRNet.scripts.live_demo.main(filename=vid_path,frames_directions=direction_changers_frames,direction_changers=direction_changers_position,out_path=out_path)
    print(msg)
    return hit_hand,handball_decision,handball_part,handball_angle,msg

#vid_path = os.getcwd() +"/inputs/handball_2.mp4"
#handball_detection(vid_path,"/vol/bitbucket/nr421/project-varaid/modules/handball_detection/outputs/")
