#imports
import numpy as np 
import cv2
import keyboard
import webbrowser
import time

url = "https://gemioli.com/hooligans/"

gameStarted = 0
thresh = 0.3

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cap = cv2.VideoCapture(0)
ret,img = cap.read()

#size of the Camera image 
rows,cols,_ = img.shape


while(1):

	ret,img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray)

	maxArea = 0
	area = 0

	#finding the biggest face in the image
	for (x,y,w,h) in faces:
		area = w*h
		if(maxArea<area):
			xface = x
			yface = y
			wface = w
			hface = h
			maxArea = area

	#mark the face in the image
	if(maxArea>0):
		cv2.rectangle(img,(xface,yface),(xface+wface,yface+hface),(0,255,0),5)
	cv2.imshow('Faces Detection',img)

	#once game has started
	if(gameStarted==1 and maxArea>0):

		#center point of the face
		xc = xface + wface*0.5;
		yc = yface + hface*0.5;

		#game controls
		if(xc>((1-thresh)*cols) and ((thresh*rows)<yc<((1-thresh)*rows))):
			keyboard.send('left')
			print("LEFT")

		if(xc<(thresh*cols) and ((thresh*rows)<yc<((1-thresh)*rows))):
			keyboard.send('right')
			print("RIGHT")	
		
		if(yc>((1-thresh)*rows) and ((thresh*cols)<xc<((1-thresh)*cols))):
			keyboard.send('down')
			print("DOWN")

		if(yc<(thresh*rows) and ((thresh*cols)<xc<((1-thresh)*cols))):
			keyboard.send('up')
			print("UP")	

	#user presses 'a' to start the game
	if cv2.waitKey(1) & 0xFF == ord('a'):
		gameStarted = 1
		webbrowser.open_new(url)
		time.sleep(10)
		keyboard.send('space')
		time.sleep(5)
		keyboard.send('esc')
		time.sleep(5)
		keyboard.send('space')

	#user presses 'q' to end
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break	

cap.release()
cv2.destroyAllWindows()