#################
#################
# Abdullah Hasan
# Finished 01/07/2020
# Baseball centerpoint and radius detector
#################
#################

import cv2
import numpy as np
import math


def main():
    
    #Load in all fifteen images

    fpath = "images/IMG%i.bmp"
    d = []
    for i in range(1,16):
        d.append(cv2.imread(fpath % i,0))

    #boxes will store whatever bounding boxes we can track
    boxes = []

    #Manual initialisation for the first ball is required
    boxes.append((530,775,40,40))

    #Create and initialise the CSRT tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(d[0],boxes[0])

    #Kernel filtering and base image improves baseball detection later
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    base = cv2.morphologyEx(d[0],cv2.MORPH_OPEN,kernel)

    #Track if the ball has been hit yet, and what frame it was hit at
    hit = False
    hit_i = None

    #Monitor if tracking has failed, at what frame it fails at
    lost = False
    lost_i = None
    
    #Store bounding box returned by the tracker
    temp_box = None
    
    #Store circles of detected baseballs
    c = []

    #Loop over the remaining images and track the ball
    for i in range(1,15):
        
        #get the bounding box returned by the tracker
        _, temp_box = tracker.update(d[i])

        #if we havent hit the ball yet
        if not hit:
            boxes.append(temp_box)

            #check if we hit
            hit = is_hit(boxes)

            if hit:
                #hit must have happened last frame
                hit_i = i-1
        elif lost:
            #if the tracker failed, we don't need to do anything else, boxes already generated
            pass
        #predictions are based on having at least two frames tracked correctly
        elif len(boxes) > 2:
            #check if tracking failed
            lost = is_lost(boxes,hit_i,temp_box)

            if lost:
                #we've lost the ball, generate larger bounding boxes for the rest of the frames
                lost_i = i
                boxes = generate_boxes(boxes,i-1)
            else:
                boxes.append(temp_box)
    
    #this loop converts bounding boxes into circles
    for i in range(0,15):

        #background subtraction
        if(i>0):
            t = d[i] - base
        else:
            t = d[i]

        #morph the image
        t = cv2.morphologyEx(t,cv2.MORPH_OPEN,kernel)

        #smooth the image and reduce salt/pepper
        t = cv2.medianBlur(t,5)
        
        #only focus on the bounding box for this frame
        masked = create_mask(t,boxes[i])
        
        #Use houghcircles to fit circles detected
        circles = cv2.HoughCircles(masked,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=5,minRadius=10,maxRadius=30)

        #check we have a circle(s)
        if(circles is not None):
            c.append(circles[0,:])
        else:
            c.append((0,0,0))
    
    #filter out circles that are false positives
    f_c = filter_circles(c,hit_i,lost_i)

    #Purely for display purposes
    for i in range(0,15):
        cir = c[i]
        for j in cir:
            cv2.circle(d[i],(j[0],j[1]),j[2],(0,255,0),2)
            cv2.circle(d[i],(j[0],j[1]),2,(0,0,255),3)
        (x,y,w,h) = [int(v) for v in boxes[i]]
        #cv2.rectangle(d[i],(x,y),(x+w,y+h),(255,0,0),3,1,0)
        cv2.imshow("circles",d[i])
        cv2.waitKey(0)
    save_circles(c)
#function to filter out false positive circles
def filter_circles(circles, hit, lost):
    keyCircles = circles

    #get direction ball is moving in after it gets hit
    xy_dir = get_xy_direction(circles[hit+1][0],circles[hit][0])

    #filter circles after the hit
    for i in range(hit,len(keyCircles)):
        t = keyCircles[i]

        #if only one circle is detected, then there isnt anything to filter
        if len(t) == 1:
            continue

        #get the previous circle
        prev = keyCircles[i-1][0]

        shortest = math.inf
        s = -1

        #loop through all the circles, find the shortest distance to the previous circle AND one that moves in the right direction
        for k in range(len(t)):
            temp_distance = get_distance(prev,t[k])
            if shortest > temp_distance:
                if ((t[k][0] > prev[0]) == xy_dir[0]) and ((t[k][1] > prev[1]) == xy_dir[1]):
                    shortest = temp_distance
                    s = k
        keyCircles[i] = [t[s]]
    return keyCircles

#Function to get distance between two points
def get_distance(A,B):
    d = math.sqrt(math.pow(A[0]-B[0],2)+math.pow(A[1]-B[1],2))
    return d

#function to find direction of two points
#False implies left for x axis and up for y axis
#True implies right for x axis and down for y axis
def get_xy_direction(A,B):
    return (A[0]>B[0],A[1]>B[1])

#Function to calculate the angle between three points
def calc_traj(A,B,C):
    d_x = A[0] - B[0]
    d_y = A[1] - B[0]
    angle_1 = math.atan2(d_y,d_x)
    d_x = C[0] - B[0]
    d_y = C[1] - B[0]
    angle_2 = math.atan2(d_y,d_x)

    angle = abs(angle_1 - angle_2) * 180 / math.pi

    return angle

#function to determine if ball has been hit yet (major change in angle > 5 degrees)
def is_hit(boxes):
    if len(boxes) < 3:
        return False
    for i in range(len(boxes)-2):
        A = boxes[i]
        B = boxes[i+1]
        C = boxes[i+2]
        angle = calc_traj(A,B,C)
        if angle > 5:
            return True
    return False

#Check the average difference between boxes after it was hit
def check_avg_diff(boxes,hit,box,s):
    diff = 0
    count = 0

    for i in range(hit,len(boxes)-1):
        diff += abs(boxes[i][s] - boxes[i+1][s])
        count += 1

    avg = diff / count
    d_diff = abs(box[s] - boxes[-1][s])

    if d_diff< avg:
        return False   
    return True

#check if the tracker has lost the target
def is_lost(boxes, hit, box):
    d_x = check_avg_diff(boxes,hit,box,0)
    d_y = check_avg_diff(boxes,hit,box,1)
    return not (d_x or d_y)

#mask an image
def create_mask(img,box):
    #box = np.asarray(box,dtype="uint8")
    img[0:int(box[1]),:] = 0
    img[:,0:int(box[0])] = 0
    img[int(box[1]+box[3]):,:] = 0
    img[:,int(box[0]+box[2]):] = 0
    return img

#function to generate large bounding boxes in the direction of the ball
def generate_boxes(boxes,p):
    p = p-1
    for i in range(p,15):
        prev = boxes[i-1]
        current = (0,660,500,120)
        if i >= len(boxes):
            boxes.append(current)
        else:
            boxes[i] = current
    return boxes

def save_circles(circles):
    f = open("circles.txt","x")
    for x in circles:
        c = x[0]
        f.write((str(c[0])+","+str(c[1])+","+str(c[2])+"\n"))
    f.close()
    
main()
##
##
#EXTRA CODE THAT WAS UNUSED
##
##

#def fit_line(series):
#    sum_x = sum(map(lambda p:p[0],series))
#    sum_y = sum(map(lambda p:p[1],series))
#    avg_x = sum_x / len(series)
#    avg_y = sum_y / len(series)
#    xy_diff = sum(map(lambda p: (p[0] - avg_x) * (p[1] - avg_y),series))
#    xx_diff = sum(map(lambda p: math.pow(p[0] - avg_x,2),series))
#    m = xy_diff / xx_diff
#    b = avg_y + (m*avg_x)
#    return (m,b)

#def get_vector(boxes,hit):
#    xs =[v[0] for v in boxes[hit:]]
#    ys = [v[1] for v in boxes[hit:]]
#    t = range(0,len(boxes[hit:]))
#    x_t = fit_line(list(zip(xs,t)))
#    y_t = fit_line(list(zip(ys,t)))
#    return (x_t,y_t)


#def estimate_next_pos(vector,boxes,hit,i):
#    x_v = vector[0]
#    y_v = vector[1]
#    i = i-hit
#    x = i*x_v[0] + x_v[1]
#    y = i*y_v[0] + y_v[1]
#    return (x,y)

#def get_xy_diff(boxes,hit):
#    xs =[v[0] for v in boxes[hit:]]
#    ys = [v[1] for v in boxes[hit:]]
#    sum_dx = 0
#    sum_dy = 0
#    for i in range(len(xs)-1):
#        sum_dx += xs[i]-xs[i+1]
#        sum_dy += ys[i]-ys[i+1]
#    sum_dx = sum_dx / len(xs) - 1
#    sum_dy = sum_dy / len(ys) - 1
#    return (-sum_dx,-sum_dy)