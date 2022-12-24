# All the imports go here
import cv2
import numpy as np
import mediapipe as mp
import imutils
import time

# default called trackbar function
# def setValues(x):
#    print("")

# Creating the trackbars needed for adjusting the marker colour
# cv2.namedWindow("Color detectors")
# cv2.createTrackbar("a", "Color detectors", 500, 700,setValues)
# cv2.createTrackbar("b", "Color detectors", 0, 255,setValues)
# cv2.createTrackbar("c", "Color detectors", 0, 255,setValues)

# Giving different arrays to handle colour points of different colour
black_points = [[[]]]
blue_points = [[[]]]
green_points = [[[]]]
red_points = [[[]]]
yellow_points = [[[]]]
voilet_points = [[[]]]

# These indexes will be used to mark the points in particular arrays of specific colour
black_index =[0] 
blue_index = [0]
green_index = [0]
red_index = [0]
yellow_index = [0]
voilet_index = [0]

# flag
flag_draw = [1]
erase_flag = [0]

curr_page = 0
total_page = 1

colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0),
           (0, 255, 255), (0, 0, 255), (127, 0, 255)]
colorIndex = [1]

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

start=time.time()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:

    # Here is code for Canvas setup
    paintWindow = np.zeros((525, 700, 3)) + 255
    # for i in range(0,66):
    #     paintWindow[i]-=100
    paintWindow = cv2.rectangle(paintWindow, (70, 5), (150,  50), colors[0], -1)
    paintWindow = cv2.rectangle(paintWindow, (165, 5), (245, 50), colors[1], -1)
    paintWindow = cv2.rectangle(paintWindow, (260, 5), (340, 50), colors[2], -1)
    paintWindow = cv2.rectangle(paintWindow, (355, 5), (435, 50), colors[3], -1)
    paintWindow = cv2.rectangle(paintWindow, (450, 5), (530, 50), colors[4], -1)
    paintWindow = cv2.rectangle(paintWindow, (545, 5), (625, 50), colors[5], -1)

    paintWindow = cv2.rectangle(paintWindow, (10, 100), (90, 145), (0, 0, 0), -1)
    paintWindow = cv2.rectangle(paintWindow, (10, 180), (90, 225), (0, 0, 0), -1)
    paintWindow = cv2.rectangle(paintWindow, (10, 260), (90, 305), (0, 0, 0), -1)
    paintWindow = cv2.rectangle(paintWindow, (10, 340), (90, 385), (0, 0, 0), -1)

    # a = cv2.getTrackbarPos("a", "Color detectors")
    cv2.putText(paintWindow, "BLACK", (87, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (186, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (275, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (366, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (475, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "VOILET", (560, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "CLEAR", (25, 128),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "ERASE", (25, 208),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  
    cv2.putText(paintWindow, "NEXT", (27, 288),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  
    cv2.putText(paintWindow, "PREV", (27, 368),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)      
    page_s=str(curr_page+1)+"/"+str(total_page)
    cv2.putText(paintWindow, page_s, (27, 448),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)                                                

    cv2.namedWindow('BOARD', cv2.WINDOW_AUTOSIZE)


    # Read each frame from the webcam
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=700)

    x, y, c = frame.shape
    print(x, y)

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    # frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    # frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    # frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    # frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)

    frame = cv2.rectangle(frame, (70, 5), (150,  50), colors[0], -1)  # Black
    frame = cv2.rectangle(frame, (165, 5), (245, 50), colors[1], -1)  # Blue
    frame = cv2.rectangle(frame, (260, 5), (340, 50), colors[2], -1)  # green
    frame = cv2.rectangle(frame, (355, 5), (435, 50), colors[3], -1)  # yellow
    frame = cv2.rectangle(frame, (450, 5), (530, 50), colors[4], -1)  # red
    frame = cv2.rectangle(frame, (545, 5), (625, 50), colors[5], -1)  # voilet

    # frame = cv2.rectangle(frame, (20, 470), (100, 515), (0, 0, 0), -1)
    frame = cv2.rectangle(frame, (10, 100), (90, 145), (0, 0, 0), -1)
    frame = cv2.rectangle(frame, (10, 180), (90, 225), (0, 0, 0), -1)
    frame = cv2.rectangle(frame, (10, 260), (90, 305), (0, 0, 0), -1)
    frame = cv2.rectangle(frame, (10, 340), (90, 385), (0, 0, 0), -1)    

    cv2.putText(frame, "BLACK", (87, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (186, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (275, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (366, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (475, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "VOILET", (560, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "CLEAR", (25, 128),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "ERASE", (25, 208),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA) 
    cv2.putText(frame, "NEXT", (27, 288),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  
    cv2.putText(frame, "PREV", (27, 368),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)      
    page_s=str(curr_page+1)+"/"+str(total_page)
    cv2.putText(frame, page_s, (27, 448),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)             

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Draw lines of all the colors on the canvas and frame
    points = [black_points[curr_page], blue_points[curr_page], green_points[curr_page], yellow_points[curr_page], red_points[curr_page], voilet_points[curr_page]]    

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # # print(id, lm)
                # print(lm.x)
                # print(lm.y)
                lmx = int(lm.x * 700)
                lmy = int(lm.y * 525)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            # mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])

        if erase_flag[curr_page]==0: 
            cv2.circle(frame, center, 5, colors[colorIndex[curr_page]], -1)
            cv2.circle(paintWindow, center, 5, colors[colorIndex[curr_page]], -1)
        elif erase_flag[curr_page]==1:
            cv2.circle(frame, center, 10, (0,0,0), -1)
            cv2.circle(paintWindow, center, 10, (0,0,0), -1)

        # print(center[1]-thumb[1])
        if (thumb[1]-center[1] < 30):
          if (flag_draw[curr_page] == 1):
            black_points[curr_page].append([])
            black_index[curr_page] += 1
            blue_points[curr_page].append([])
            blue_index[curr_page] += 1
            green_points[curr_page].append([])
            green_index[curr_page] += 1
            red_points[curr_page].append([])
            red_index[curr_page] += 1
            yellow_points[curr_page].append([])
            yellow_index[curr_page] += 1
            voilet_points[curr_page].append([])
            voilet_index[curr_page] += 1            
            flag_draw[curr_page] = 0
          else:
            pass 

        elif 5 <= center[1] <= 50:
            if 70 <= center[0] <= 150:
                    colorIndex[curr_page] = 0 # Black
                    erase_flag[curr_page]=0
            elif 165 <= center[0] <= 245:
                    colorIndex[curr_page] = 1 # Blue
                    erase_flag[curr_page]=0
            elif 260 <= center[0] <= 340:
                    colorIndex[curr_page] = 2 # Green
                    erase_flag[curr_page]=0
            elif 355 <= center[0] <= 435:
                    colorIndex[curr_page] = 3 # Yellow
                    erase_flag[curr_page]=0
            elif 450 <= center[0] <= 530:
                    colorIndex[curr_page] = 4 # Red
                    erase_flag[curr_page]=0
            elif 545 <= center[0] <= 625:
                    colorIndex[curr_page] = 5 # Voilet
                    erase_flag[curr_page]=0

        elif 10 <= center[0] <= 90:
            if 100 <= center[1] <= 145: # Clear Button
                black_points[curr_page] = [[]]
                blue_points[curr_page] = [[]]
                green_points[curr_page] = [[]]
                red_points[curr_page] = [[]]
                yellow_points[curr_page] = [[]]
                voilet_points[curr_page] = [[]]

                black_index[curr_page] = 0
                blue_index[curr_page] = 0
                green_index[curr_page] = 0
                red_index[curr_page] = 0
                yellow_index[curr_page] = 0
                voilet_index[curr_page] = 0

                paintWindow[51:,91:,:] = 255
                erase_flag[curr_page]=0

            elif 180 <= center[1] <= 225: # Erase Button:    
                erase_flag[curr_page]=1

            elif 260 <= center[1] <= 305: # Next Button:  
              if time.time()-start>1:    
               if curr_page+1==total_page:
                curr_page+=1
                total_page+=1
                black_points.append([[]])
                blue_points.append([[]])
                green_points.append([[]])
                red_points.append([[]])
                yellow_points.append([[]])
                voilet_points.append([[]])
                black_index.append(0)
                blue_index.append(0)
                green_index.append(0)
                red_index.append(0)
                yellow_index.append(0)
                voilet_index.append(0)
                colorIndex.append(1)
                flag_draw.append(1)
                erase_flag.append(0)
               else:
                curr_page+=1 
               start=time.time() 
              else:
                pass

            elif 340 <= center[1] <= 385: # Prev Button:    
              if time.time()-start>1:
                if(curr_page==0):
                    pass
                else:
                    curr_page-=1  
                start=time.time()       
              else:
                pass    

        else :
            flag_draw[curr_page]=1

            if erase_flag[curr_page]==0:
             if colorIndex[curr_page] == 0:
                black_points[curr_page][black_index[curr_page]].append(center)            
             if colorIndex[curr_page] == 1:
                blue_points[curr_page][blue_index[curr_page]].append(center)
             elif colorIndex[curr_page] == 2:
                green_points[curr_page][green_index[curr_page]].append(center)
             elif colorIndex[curr_page] == 3:
                yellow_points[curr_page][yellow_index[curr_page]].append(center)
             elif colorIndex[curr_page] == 4:
                red_points[curr_page][red_index[curr_page]].append(center)
             elif colorIndex[curr_page] == 5:
                voilet_points[curr_page][voilet_index[curr_page]].append(center)
            
            elif erase_flag[curr_page]==1:
                for i in range(len(points)):
                  for j in range(len(points[i])):
                    for k in range(0, len(points[i][j])):
                       x=center[0]-points[i][j][k][0]
                       a=x*x
                       y=center[1]-points[i][j][k][1]
                       b=y*y
                       if (a+b)<100:
                        points[i][j][k]=(-1,-1)

    # Append the next deques when nothing is detected to avois messing up
    else:
      if (flag_draw == 1):  
        black_points[curr_page].append([])
        black_index[curr_page] += 1        
        blue_points[curr_page].append([])
        blue_index[curr_page] += 1
        green_points[curr_page].append([])
        green_index[curr_page] += 1
        red_points[curr_page].append([])
        red_index[curr_page] += 1
        yellow_points[curr_page].append([])
        yellow_index[curr_page] += 1
        voilet_points[curr_page].append([])
        voilet_index[curr_page] += 1        
      else:
        pass 

    # Draw lines of all the colors on the canvas and frame
    # points = [black_points, blue_points, green_points, yellow_points, red_points, voilet_points]

    # for j in range(len(points[0])):
    #         for k in range(1, len(points[0][j])):
    #             if points[0][j][k - 1] is None or points[0][j][k] is None:
    #                 continue
    #             cv2.line(paintWindow, points[0][j][k - 1], points[0][j][k], colors[0], 2)
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if (points[i][j][k - 1] is None or points[i][j][k] is None) or points[i][j][k - 1]==(-1,-1) or points[i][j][k]==(-1,-1):
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("CAMERA", frame) 
    cv2.imshow("BOARD", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

