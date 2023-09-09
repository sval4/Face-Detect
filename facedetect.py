import cv2
from fer import FER

cascadeFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotionDetector = FER()

capture = cv2.VideoCapture(0)

while True:
    rectangle, img = capture.read()
    #Convert to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Detect Face
    face = cascadeFace.detectMultiScale(grayscale, 1.3, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x,y), (x + w, y + h), (255, 0, 0), 2)

        faceROI = img[y:y + h, x:x + w]

        emotion = emotionDetector.detect_emotions(faceROI)

        if emotion:
            #print(emotion[0]["emotions"])
            mainEmotion = max(emotion[0]["emotions"], key=lambda x: emotion[0]["emotions"][x])
            cv2.putText(img, mainEmotion, (x,y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("img", img)
    wait = cv2.waitKey(50)
    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1 or wait == 27:
        break

capture.release()
cv2.destroyAllWindows()