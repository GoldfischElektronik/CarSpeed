import datetime
import cv2
##from helper import create_video_writer  #used when uncommenting the video creation
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import math
import numpy as np
import csv
from pathlib import Path

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
frame_width_m = 2*(math.tan(math.radians(FOV*0.5))*DISTANCE)
mperpixel = frame_width_m / float(IMAGEWIDTH)
print("-------------------------------------------------------")
print("Image width in m {} at {} from camera".format("%.0f" % frame_width_m,"%.0f" % DISTANCE))

# define the area of interest so the calculations are not applaied to the whole picture

area_width                  = 200 #200
area_height                 = 110
area_timingFrame            = 25  #blue (1) and yellow (2) areas
area_boundry_top            = 150
area_boundry_bottom         = 260
area_boundry_left           = 150 #150
area_boundry_right          = area_boundry_left + area_width
area_dist_roi_timing        = 50 #distance to the left and right area (roi) to give room for the tracker to center the vehicle
area_pixels_between_areas   = area_width - area_dist_roi_timing *2 - area_timingFrame   # number of pixels between the the outer boundry of one area to the inner boundry of the other
area_meters_between         = area_pixels_between_areas * mperpixel                     # todo: calculate the deviation when a vehicle does not enter an area on the first exact pixel within the area



# y from:to, x from:to
area_roi= [area_boundry_top, area_boundry_bottom , area_boundry_left, area_boundry_right]

# defining areas from where an object is captured

area_roi4poli = [(area_boundry_left,area_boundry_top), (area_boundry_right,area_boundry_top), (area_boundry_right, area_boundry_bottom), (area_boundry_left,area_boundry_bottom)]
area_1 = [(area_dist_roi_timing,0), (area_timingFrame+area_dist_roi_timing, 0), (area_timingFrame+area_dist_roi_timing,area_height), (area_dist_roi_timing,area_height)]
area_2 = [(area_width-area_dist_roi_timing-area_timingFrame, 0), (area_width-area_dist_roi_timing,0), (area_width-area_dist_roi_timing,area_height), (area_width-area_dist_roi_timing-area_timingFrame,area_height)]

# Colors  B (blue), G (green), R (red)
GREEN = (0, 255, 0)     # B, G, R
WHITE = (255, 255, 255) # B, G, R
ORANGE = (42, 81, 252)  # B, G, R
BLACK = (0,0,0)         # B, G, R
BLUE = (255, 0, 0)      # B, G, R
RED = (0, 0, 255)
YELLOW = (0, 230, 250)

# initialize the video capture object

video_cap = cv2.VideoCapture('outputKeep.avi')
# video_cap = cv2.VideoCapture(2)

# video_cap.set(3, IMAGEWIDTH)
# video_cap.set(4, IMAGEHEIGHT)


# initialize the video writer object
#writer = create_video_writer(video_cap, "output.mp4")


trackerDeepSort = DeepSort(max_age=50) #50 is a good value

# Object detection from Stable camera
# The function cv2.createBackgroundSubtractorMOG2 was added at the beginning 
# without defining parameters, now let’s see how to further improve our result. 
# history is the first parameter, in this case, it is set to 100 because the camera is fixed. 
# var Threshold instead is 40 because the lower the value the greater the possibility of making 
# false positives. In this case, we are only interested in the larger objects.
object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30) #MOG MOG2 GMG

framePlus = 0 #number of pixels added to the rectancle to get better trackerDeepSortDeppSort results -> does not improve it

def calculate_time_delta_seconds(startTime, endTime):
    diff = endTime - startTime
    return diff.total_seconds()

# calculate speed from pixels and time
def get_speed(pixels, secs):
    if secs > 0.0:
        m = pixels * mperpixel
        km = m / 1000
        # s = secs
        h = secs / 3600
        # print("m:  ", m)
        # print("km: ", km)
        # print("s : ", secs)
        # print("h:  ", h)
        return (km/h) 

    else:
        return 0.0
    

    # speed_car = get_speed(area_pixels_between_areas, mperpixel, ) area_meters_between/1000 / ( time_car_AtoB / 3600.0 )
 
# record speed in .csv format
def record_speed(res):

# writer.writerow(["Isabel Walter", "50", "United Kingdom"])

    global csvfileout
    f = open(csvfileout, 'a')
    f.write(res+"\n")
    f.close

csvFileName = "carspeed_{}.cvs".format(datetime.datetime.now().strftime("%Y%m%d_%H%M"))

my_file = Path(csvFileName)

with open('csvFileName', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Date", "Day", "Time" ,"Speed" ,"Image"]
    writer.writerow(field)
    
def save_picture(timestamp, speed):
    # timestamp the image
    cv2.putText(frame, time_car_end.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 1)
    # write the speed: first get the size of the text
    size, base = cv2.getTextSize( "%.0f kmh" % speed_car, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    # then center it horizontally on the image
    cntr_x = int((IMAGEWIDTH - size[0]) / 2) 
    cv2.putText(frame, "%.0f kmh" % speed_car,
        (cntr_x , int(IMAGEHEIGHT * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 2.00, (0, 255, 0), 3)
    # and save the image to disk
    imageFilename = "car_at_" + time_car_end.strftime("%Y%m%d_%H%M%S") + ".jpg"
    # use the following image file name if you want to be able to sort the images by speed
    #imageFilename = "car_at_%02.0f" % last_mph + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                        
    cv2.imwrite(imageFilename,frame)
                                            
    # record_speed(time_car_end.strftime("%Y.%m.%d")+','+time_car_end.strftime('%A')+','+\
    #     time_car_end.strftime('%H%M')+','+("%.0f" % speed_car) + ','+imageFilename)







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

## variable ist used to skip frames (30 fps -> 10 fps used)
count = 0

# create a dictionary for tracking vehicles

vehiclesCrossed_area_1 = {}
vehiclesCrossed_area_2 = {}


while True:
    # capture the time for the FPS and vehicle speed (time spent between area 1 and 2)
    time_start = datetime.datetime.now()

     # initialize the list of bounding boxes and confidences
    results = []

    # skipping every 5rd frame (30 fps -> 10 fps used) todo: does this really work?
    count +=1  
    if count % 5 != 0:
            # print("skipped frame", count)
            continue
    

    ######################################
    # DETECTION
    ######################################

    ret, frame = video_cap.read()
    height, width, _ = frame.shape

    

    # blacking out sentitive areas which should nogt be monitored
    # cv2.rectangle(frame,(0,0), (640, 100), (BLACK), -1) # B, G, R)
    # cv2.rectangle(frame,(0,300), (640, 480), (BLACK), -1) # B, G, R)

    # Extract Region of interest
    # y from:to, x from:to
    roi = frame[area_roi[0]:area_roi[1], area_roi[2]:area_roi[3]]
    # cv2.rectangle(frame, (area_roi[2], area_roi[0]), (area_roi[3], area_roi[1]), (BLACK), 6) # B, G, R
    
    
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
            # cv2.rectangle(roi, (x, y), (x + w, y + h), (ORANGE), 3) # B, G, R

            cx = int((x+x+w)/2)
            cy = int((y+y+h)/2)

            cv2.circle(roi, (cx, cy), 6, RED, -1)
            
            
            results.append([[x-framePlus, y-framePlus, w+framePlus, h+framePlus], 99, 5]) ##todo remove arguments 2 and 3

    # print("##################################################################")
    # print("results: ", results)

    # update the trackerDeepSort with the new detections
    tracks = trackerDeepSort.update_tracks(results, frame=roi)

    # show the area 1 and 2 only for the displayed picture invisible for the calculations
    # cv2.polylines(roi, [np.array(area_1, np.int32)], 1, (BLUE), 1) 
    # cv2.polylines(roi, [np.array(area_2, np.int32)], 1, (YELLOW), 1)
    # cv2.polylines(frame, [np.array(area_roi4poli, np.int32)], 1, (RED), 1)

    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id # the track_id is a string!
        track_id_int = int(track_id)
   
        ltrb = track.to_ltrb()
        # print("-----------------------------------------------------------")
        # print("track:         ", track)
        # print("id:            ", track_id)

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        track_cx = int(xmin+(xmax-xmin)/2)
        track_cy = int(ymin+(ymax-ymin)/2)

        # check if the center point of the object is located within one of the triangles (area_1, area_2), using numpy arrays as they are much faster than openCV arrays
        isInArea_1 = cv2.pointPolygonTest(np.array(area_1, np.int32), (int(track_cx), int(track_cy)), False) # true would also return distance between polygons, ...
        isInArea_2 = cv2.pointPolygonTest(np.array(area_2, np.int32), (int(track_cx), int(track_cy)), False) # true would also return distance between polygons, ...      
    
        if isInArea_1 >= 0:
            if track_id_int not in vehiclesCrossed_area_1:
                vehiclesCrossed_area_1[track_id_int] = (time_start)
                # print("+ area_2 dict: ", track_id_int) 
                # print("  time       : ", vehiclesCrossed_area_2.get(track_id_int))
                if track_id_int in vehiclesCrossed_area_2:
                    
                    time_car_start = vehiclesCrossed_area_2.get(track_id_int)
                    time_car_end = time_start
                            
                    time_car_diff = calculate_time_delta_seconds(time_car_start, time_car_end)
                   
                    speed_car = get_speed(area_pixels_between_areas, time_car_diff)
                    print("--------------------------------------")
                    print("id:         ", track_id_int)  
                    print("start:      ", time_car_start)
                    print("end:        ", time_car_end)
                    print("diff in s:  ", time_car_diff)
                    print("distance m: ", area_meters_between)
                    print("speed:      ", speed_car)
                    
                    save_picture(time_car_end, speed_car)
                    

        if isInArea_2 >= 0:
            if track_id_int not in vehiclesCrossed_area_2:
                vehiclesCrossed_area_2[track_id_int] = (time_start)
                # print("+ area_2 dict: ", track_id_int) 
                # print("  time       : ", vehiclesCrossed_area_2.get(track_id_int))
                if track_id_int in vehiclesCrossed_area_1:
                    
                    time_car_start = vehiclesCrossed_area_1.get(track_id_int)
                    time_car_end = time_start
                            
                    time_car_diff = calculate_time_delta_seconds(time_car_start, time_car_end)
                   
                    speed_car = get_speed(area_pixels_between_areas, time_car_diff)
                    
                    print("--------------------------------------")
                    print("id:         ", track_id_int)  
                    print("start:      ", time_car_start)
                    print("end:        ", time_car_end)
                    print("diff in s:  ", time_car_diff)
                    print("distance m: ", area_meters_between)
                    print("speed:      ", speed_car)

                    save_picture(time_car_end, speed_car)


        if track_id_int in vehiclesCrossed_area_1:
            # draw the bounding box and the track id
            
            a1time = vehiclesCrossed_area_1.get(track_id_int)

            cv2.circle(roi, (track_cx, track_cy), 5, BLUE, -1)
            cv2.rectangle(roi, (xmin, ymin), (xmax, ymax), BLUE, 2)
            cv2.rectangle(roi, (xmin, ymin - 20), (xmin + 20, ymin), BLUE, -1)
            cv2.putText(roi, str(track_id), (xmin + 2, ymin - 7), cv2.FONT_HERSHEY_DUPLEX, 0.5, BLACK, 1) 
            time_passed = calculate_time_delta_seconds(a1time, time_start)
            temp_format_float = "{:.2f}".format(time_passed)
            cv2.putText(roi, str(temp_format_float), (xmin + 20, ymin - 7), cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE, 1)

            # cv2.putText(roi, str(temp_format_float), (xmin + 60, ymin - 7), cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE, 1)


        if track_id_int in vehiclesCrossed_area_2:
            # draw the bounding box and the track idq
            a2time = vehiclesCrossed_area_2.get(track_id_int)

            cv2.circle(roi, (track_cx, track_cy), 5, YELLOW, -1)
            cv2.rectangle(roi, (xmin, ymin), (xmax, ymax), YELLOW, 2)
            cv2.rectangle(roi, (xmin, ymin - 20), (xmin + 20, ymin), YELLOW, -1)
            cv2.putText(roi, str(track_id), (xmin + 2, ymin - 7), cv2.FONT_HERSHEY_DUPLEX, 0.5, BLACK, 1)
            time_passed = calculate_time_delta_seconds(a2time, time_start)
            temp_format_float = "{:.2f}".format(time_passed)
            cv2.putText(roi, str(temp_format_float), (xmin + 20, ymin - 7), cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE, 1)

            

    # end time to compute the fps
    time_end = datetime.datetime.now()

    # show the time it took to process 1 frame
    ##print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (time_end - time_start).total_seconds():.2f}"
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
