import pygame
import cv2 as cv
import numpy as np 
import os 
import time
import sys
import onnx
import onnxruntime as rt
import json
import datetime

#Inisiate model for detecting and predicting 
faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv.CascadeClassifier("haarcascade_eye.xml")
model = "modelCNN_2.onnx" 

#Set the captured video properties
camera_id= "/dev/video0"
capture = cv.VideoCapture(camera_id, cv.CAP_V4L2)
capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
capture.set(cv.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
capture.set(cv.CAP_PROP_FPS, 30)

#Exit the Program if frames not captured
if not capture.isOpened():
  print("Could not open video")
  sys.exit()

#Inisiate FPS dan inference speed values
start_fps = time.time()
start_time = time.time()
frame_count = 0
fps = 0
fps_list = []

#Create e list for saving inference speed values
face_tList = []
rEye_tList = []
lEye_tList = []
rPred_tList = []
lPred_tList = []

#Inisiate starting Eye prediction state
r_detect = False
l_detect = False

#Inisiate startiing alarm state
counter = 0
play = True

#Predicting both eyes condition 
def getPrediction(roi) :
    if roi is not None:
        resize = cv.resize(roi,(24,24))
        input = resize.reshape(-1, 24, 24, 1)
        input = input/255.0

        data = json.dumps({'data': input.tolist()})
        data = np.array(json.loads(data)['data']).astype('float32')

        session = rt.InferenceSession(model, providers = ['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: data})

        pred=(np.array(result).squeeze())
        prediction = np.round(pred)       
    return prediction

# Playing Alarm when conditions meet
def playAlarm () :
    pygame.mixer.init()
    sound_file = "sound/alarm.wav"
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play(-1)

# Get the Name for saved file 
def fileName(mode):
    if mode == 1 : 
        path = "saved-Video"
    elif mode == 2 : 
        path = "inverence-Results"
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    if mode == 1 : 
        file_name = os.path.join(path, f"video-{formatted_time}.mp4v")
        return file_name
    elif mode == 2 :
        file_name = os.path.join(path, f"data-{formatted_time}.txt")
        with open(file_name, 'w') as file:
            for item in data:
                file.write(str(item) + '\n') 
    
#Save Captured video
videoName = fileName(1)
output = cv.VideoWriter(videoName, cv.VideoWriter_fourcc(*'XVID'), 15, (320,240))

#capturing video
while(True):
    
    ret,frame = capture.read()
    if not ret : 
        break

    #Show FPS while opening video
    frame_count += 1
    cv.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if time.time() - start_fps > 1 :

        elapsed_fps = time.time() - start_fps
        fps = frame_count / elapsed_fps
        fps_list.append(fps)
        frame_count = 0
        start_fps = time.time()

    # Preprosesing - Grayscaleqqqq
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    

    #face detection
    face_t0 = time.time()
    face = faceCascade.detectMultiScale(gray, minNeighbors = 15, scaleFactor = 1.1, minSize = (50,50))
    face_t1 = time.time()
    face_t = face_t1 - face_t0
    face_tList.append(face_t)
    for (x, y, w, h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)

    if face != () :

        # Create detection areas of eyes     
        rRoi_gray = gray[y:y+(h*3//4), x:x+(w//2)] 
        lRoi_gray = gray[y:y+(h*3//4), x+(w//2):x+w] 

        # Crate the ROIs of eyes  
        l_roi = frame[y:y+(h*3//4), x+(w//2):x+w]
        r_roi = frame[y:y+(h*3//4), x:x+(w//2)]    

        # Detecting Left Eye
        lEye_t0 = time.time()
        l_eye = eyeCascade.detectMultiScale(lRoi_gray, minNeighbors = 5) 
        lEye_t1 = time.time()
        lEye_t = lEye_t1 - lEye_t0
        lEye_tList.append(lEye_t)
        for (x1, y1, w1, h1) in l_eye:
            cv.rectangle(l_roi,(x1,y1),(x1+w1,y1+h1),(0,255,0),1)
            l_detect = True
        
        # Detecting Right Eye
        rEye_t0 = time.time()
        r_eye = eyeCascade.detectMultiScale(rRoi_gray, minNeighbors = 5) 
        rEye_t1 = time.time()
        rEye_t = rEye_t1 - rEye_t0
        rEye_tList.append(rEye_t)
        for (x2, y2, w2, h2) in r_eye:
            cv.rectangle(r_roi,(x2,y2),(x2+w2,y2+h2),(0,255,0),1)
            r_detect = True

        # Creating prediction ROIs
        if (r_detect == True) & (l_detect == True):
            lEye_roi = lRoi_gray[y1:y1+h1, x1:x1+w1]
            rEye_roi = rRoi_gray[y2:y2+h2, x2:x2+w2]

            #predicting right eye
            lPred_t0 = time.time()
            l_pred = getPrediction(lEye_roi)
            lPred_t1 = time.time()
            lPred_t = lPred_t1 - lPred_t0
            lPred_tList.append(lPred_t)

            #predicting left eye
            rPred_t0 = time.time()
            r_pred = getPrediction(rEye_roi)
            rPred_t1 = time.time()
            rPred_t = rPred_t1 - rPred_t0
            rPred_tList.append(rPred_t) 

            # Eyes Open condition
            if (l_pred == 1) & (r_pred == 1) :
                counter = 0
                play = True
                pygame.mixer.quit()   
                cv.putText(frame, "Open Eyes", (230,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_4)
            
            elif (l_pred == 0) & (r_pred == 0) : 
                # Calculating closed eyes time
                if counter == 0 :
                    start_count = time.time()
                    counter = 1
                elapsed_count = time.time() - start_count
                seconds = int(elapsed_count)
                cv.putText(frame, "Closed Eyes", (215,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_4)
                cv.putText(frame, str(seconds), (10,225), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_4)
                
                # Turn the Alarm on when conditions meet 
                if seconds >= 2:
                    cv.putText(frame, "ALERT DROWSINESS DETECTED!!", (40,225), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_4)
                    cv.rectangle(frame, (0, 0), (320, 240), (0,0,255), 5)
                    if play == True :
                        play = False
                        playAlarm()

    output.write(frame)

    total_time = time.time() - start_time
    print("System is running with total time", total_time)    
    cv.imshow('frame', cv.resize(frame, (640,480)))

    #break the video if 'q' is pressed
    if (cv.waitKey(1) & 0xFF == ord('q')) or (total_time >= 45):
        pygame.mixer.quit() 
        break

# Release all program
capture.release()
output.release()
cv.destroyAllWindows()

# saving inference values 
data = [f"System FPS Average = {(sum(fps_list)/len(fps_list)):.3f}", 
        f"Detecting Face Average Time = {(sum(face_tList)/len(face_tList)):.3f}",
        f"Detecting Right Eye Average Time = {(sum(rEye_tList)/len(rEye_tList)):.3f}",
        f"Detecting Left Eye Average Time = {(sum(lEye_tList)/len(lEye_tList)):.3f}",
        f"Predicting Right Eye Average Time = {(sum(rPred_tList)/len(rPred_tList)):.3f}",
        f"Predicting Left Eye Average Time = {(sum(lPred_tList)/len(lPred_tList)):.3f}"
        ]
fileName(2)