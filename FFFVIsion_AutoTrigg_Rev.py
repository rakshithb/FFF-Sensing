# Import required libraries
from sys import exit
import numpy as np
import cv2
print('OpenCV version: '+cv2.__version__)
import Jetson.GPIO as GPIO
from time import sleep
import datetime
from signal import signal, SIGINT
from sys import exit
import pandas as pd


################ MAIN ################

# Setup IO
GPIO.setmode(GPIO.BCM)
GPIO.setup(26,GPIO.IN) # set pin 26 as digital IN from Robot for camera start trigger

vidCount = 0
start_timestamp = False
end_timestamp = False
acq_trigger_started = False
acq_trigger_ended = False
acq_trigger_wait = False

columns = ['Video_Count', 'Layer', 'Speed', 'Start_Timestamp', 'End_Timestamp', 'Time_Diff']
lst = []
vfc = 1
vR = 10
layer = 3

try:
        
    cap = cv2.VideoCapture("/dev/video1")

    # check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error initializing camera stream, check camera connections. Exiting ...")
        exit(0)
    
    while (cap.isOpened()):

        if not acq_trigger_wait:
            print("Waiting for start video acqusition trigger . . . ")
            acq_trigger_wait = True

        while(GPIO.input(26)==0):
       
            if not acq_trigger_started:             

                start_time = datetime.datetime.now()

                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))

                size = (frame_width, frame_height)                
                
                vidCount = vidCount + 1
                vidName = "vid" + str(vidCount)
                result = cv2.VideoWriter(str(vidName) + ".avi",cv2.VideoWriter_fourcc(*'XVID'),30, size)

                print('The video acqusition for vid{0} has started at {1}'.format(vidCount,start_time))
                acq_trigger_started = True
                acq_trigger_wait = False
                acq_trigger_ended = False
                                

            frame_exists, frame = cap.read()
            #print('frame count: {0}'.format(frameCount))
            #print('ret: {0}'.format(ret))

            if frame_exists:

                    # Write the frame into the
                    # file 'filename.avi'
                    result.write(frame)

                    # show frame
                    cv2.imshow('Live Video', frame)                    

                    # Press S on keyboard to stop the process
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            else:
                print("Can't retrieve frame - stream may have ended. Exiting..")
                break
        
        if not acq_trigger_ended:
            if vidCount >= 1:
                end_time = datetime.datetime.now()
                print('The video acqusition for vid{0} has ended at {1}'.format(vidCount,end_time))
                acq_trigger_ended = True
                acq_trigger_started = False 
                
                # Closes all the frames
                result.release()
                cv2.destroyAllWindows()
                print("The video was successfully saved")
                
                time_diff = (end_time - start_time)
                execution_time = time_diff.total_seconds() * 1000
                lst.append([vidCount,layer,vR,start_time,end_time,time_diff])
                
                # Update layer count and speed
                vR = vR+10
                
                if(vR > 50):
                    vR = 10
                    layer = layer + 1

except KeyboardInterrupt:
    # user pressed ctrl + C
    print("Program terminated by user. Exiting gracefully . . . ")
    if(cap.isOpened()):
        cap.release()
        result.release()
    GPIO.cleanup()
    cv2.destroyAllWindows()

    # Save pandas dataframe to excel/csv
    video_timestamps = pd.DataFrame(lst,columns=columns)
    video_timestamps.to_excel('Video_Timestamps_' + vfc + '.xlsx',index=False)
    exit(0)

