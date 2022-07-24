import numpy as np

def handball_collider(points,ball_bbox,confidence_threshold=0.5):
    left_upper_joint = [5,7]
    left_lower_joint = [7,9]
    right_upper_joint = [6,8]
    right_lower_joint = [8,10]

    check_left_upper = get_collision(left_upper_joint,points,ball_bbox,confidence_threshold)
    check_left_lower = get_collision(left_lower_joint,points,ball_bbox,confidence_threshold)
    check_right_upper = get_collision(right_upper_joint,points,ball_bbox,confidence_threshold)
    check_right_lower = get_collision(right_lower_joint,points,ball_bbox,confidence_threshold)

    handball_decision = False
    handball_part = None
    handball_angle = None

    if (check_left_upper or check_left_lower):
        elbow = points[7]
        shoulder= points[5]
        hip=points[11]
        wrist = points[9]
        inside_arm_angle = calculate_angle(elbow,shoulder,hip)
        handball_part = "left"
        handball_angle = inside_arm_angle
        if inside_arm_angle >50:
            handball_decision = True

    if (check_right_upper or check_right_lower):
        elbow = points[8]
        shoulder= points[6]
        hip=points[12]
        wrist = points[10]
        inside_arm_angle = calculate_angle(elbow,shoulder,hip)
        handball_part = "right"
        handball_angle = inside_arm_angle
        if inside_arm_angle >10:
            handball_decision = True

    hit_hand = check_left_upper or check_left_lower or check_right_upper or check_right_lower
    msg = msg_builder(hit_hand,handball_decision,handball_part,handball_angle)
    print(hit_hand)
    return hit_hand,handball_decision,handball_part,handball_angle,msg

def get_collision(joint,points,ball_bbox,confidence_threshold):
    
    pt1,pt2 = points[joint]
    pt1_coord = [pt1[1],pt1[0]]
    pt2_coord = [pt2[1],pt2[0]]
    #print("Checking Position of left lower: {} {}".format(pt1,pt2))
    if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
        pass
    else:
        return False

    line1,line2,line3,line4 = ball_bbox_line_segments(ball_bbox)
    lines = [line1,line2,line3,line4]
    #print("Checking Lines:")
    #print(lines)

    any_line_intersect = False
    for line in lines:
        any_line_intersect = any_line_intersect or intersect(pt1_coord,pt2_coord,line[0],line[1])
    
    return any_line_intersect

def ball_bbox_line_segments(xyxy):
    expand_x = 7
    expand_y = 7
    cx1 = int(xyxy[0])-expand_x
    cy1 = int(xyxy[1])-expand_y
    cx2 = int(xyxy[2])+expand_x
    cy2 = int(xyxy[3])+expand_y
    cx3 = cx2
    cy3 = cy1
    cx4 = cx1
    cy4 = cy2
    return [[cx1,cy1],[cx3,cy3]],[[cx3,cy3],[cx2,cy2]],[[cx2,cy2],[cx4,cy4]],[[cx4,cy4],[cx1,cy1]]

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return int(angle) 

def msg_builder(hit_hand,handball_decision,handball_part,handball_angle):
    msg = ""

    if hit_hand:
        msg += "The ball did hit the player's hand."
        if handball_decision:
            msg += "The decision is to award a handball since the ball hit the player's {} arm.".format(handball_part)
            msg += "The arm was at an angle of {} degrees.".format(handball_angle)
        else:
            msg += "However, the decision is to not award a handball since the ball hit the player's {} arm ".format(handball_part)
            msg += "an angle of {} degrees.".format(handball_angle)
    else: 
        msg += "The ball did not hit the player's hand."
    
    return msg