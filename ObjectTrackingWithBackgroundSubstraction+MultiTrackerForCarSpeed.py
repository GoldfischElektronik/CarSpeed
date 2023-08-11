import datetime
import cv2
##from helper import create_video_writer  #used when uncommenting the video creation
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import math

# Ps3Eye Specification

# field of view red = 56 degrees
# field of view blue = 75 degrees

# 56 degrees
# distance to road from bathroom 52 m
# distance covered by camera 55.3 m
# https://www.calculator.net/triangle-calculator.html?vc=28&vx=0&vy=52&va=&vz=&vb=90&angleunits=d&x=56&y=37

# 75 degrees
# distance to road from bathroom 52 m
# distance covered by camera  m 79 m
# https://www.calculator.net/triangle-calculator.html?vc=32.5&vx=&vy=62&va=&vz=&vb=90&angleunits=d&x=71&y=20

# define some constants
DISTANCE = 52  #<---- enter your distance-to-road value here
MIN_SPEED = 55  #<---- enter the minimum speed for saving images
FOV = 56    #<---- Field of view
FPS = 30

SAVE_CSV = False  #<---- record the results in .csv format in carspeed_(date).csv

IMAGEWIDTH = 640
IMAGEHEIGHT = 480
# IMAGEWIDTH = 320
# IMAGEHEIGHT = 240

SHOW_BOUNDS = True
SHOW_IMAGE = True

# the following enumerated values are used to make the program more readable
WAITING = 0
TRACKING = 1
SAVING = 2
UNKNOWN = 0
LEFT_TO_RIGHT = 1
RIGHT_TO_LEFT = 2

# calculate the the width of the image at the distance specified
frame_width_ft = 2*(math.tan(math.radians(FOV*0.5))*DISTANCE)
ftperpixel = frame_width_ft / float(IMAGEWIDTH)
print("Image width in feet {} at {} from camera".format("%.0f" % frame_width_ft,"%.0f" % DISTANCE))













# Colors  B (blue), G (green), R (red)
GREEN = (0, 255, 0) # B, G, R
WHITE = (255, 255, 255) # B, G, R
ORANGE = (42, 81, 252) # B, G, R

# initialize the video capture object

video_cap = cv2.VideoCapture('outputKeep.avi')
# video_cap.set(3, IMAGEWIDTH)
# video_cap.set(4, IMAGEHEIGHT)


# initialize the video writer object
#writer = create_video_writer(video_cap, "output.mp4")


tracker = DeepSort(max_age=50) #50 is a good value

# Object detection from Stable camera
# The function cv2.createBackgroundSubtractorMOG2 was added at the beginning 
# without defining parameters, now let’s see how to further improve our result. 
# history is the first parameter, in this case, it is set to 100 because the camera is fixed. 
# var Threshold instead is 40 because the lower the value the greater the possibility of making 
# false positives. In this case, we are only interested in the larger objects.
object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30) #MOG MOG2 GMG

framePlus = 0 #number of pixels added to the rectancle to get better tracker results -> does not improve it

# calculate speed from pixels and time
def get_speed(pixels, ftperpixel, secs):
    if secs > 0.0:
        return ((pixels * ftperpixel)/ secs) * 0.681818  
    else:
        return 0.0
 
# calculate elapsed seconds
def secs_diff(endTime, begTime):
    diff = (endTime - begTime).total_seconds()
    return diff

# record speed in .csv format
def record_speed(res):
    global csvfileout
    f = open(csvfileout, 'a')
    f.write(res+"\n")
    f.close

# mouse callback function for drawing capture area xx todo: implement this into the now fixed defined area
# def draw_rectangle(event,x,y,flags,param):
#     global ix,iy,fx,fy,drawing,setup_complete,image, org_image, prompt
 
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y
 
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             image = org_image.copy()
#             prompt_on_image(prompt)
#             cv2.rectangle(image,(ix,iy),(x,y),(0,255,0),2)
  
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         fx,fy = x,y
#         image = org_image.copy()
#         prompt_on_image(prompt)
#         cv2.rectangle(image,(ix,iy),(fx,fy),(0,255,0),2)




while True:
    start = datetime.datetime.now()

    # ret, frame = video_cap.read()
    # if not ret:
    #     break


    # initialize the list of bounding boxes and confidences
    results = []

    ######################################
    # DETECTION
    ######################################

    ##get interesting ares and mark them with rectangles

    ret, frame = video_cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    # y from:to, x from:to
    roi = frame[150: 260,50: 540]

    # 1. Object Detection
    mask = object_detector.apply(roi)

    # The first argument is the source image, which should be a grayscale image. 
    # The second argument is the threshold value which is used to classify the pixel values. 
    # The third argument is the maximum value which is assigned to pixel values exceeding the threshold. 
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) # 254, 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        
        if area > 150: # define what size of elements should be filtered
            # showing contours
            #cv2.drawContours(roi, [cnt], -1, (42, 81, 252), 2) # B, G, R
            
            # showing bounding rectangles
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (ORANGE), 3) # B, G, R
            results.append([[x-framePlus, y-framePlus, w+framePlus, h+framePlus], 99, 5]) ##todo remove arguments 2 and 3


    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=roi)
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # draw the bounding box and the track id
        cv2.rectangle(roi, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(roi, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(roi, str(track_id), (xmin + 5, ymin - 8),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    ##print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
     #writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
 #writer.release()
cv2.destroyAllWindows()