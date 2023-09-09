import cv2

cascadeFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)

while True:
    rectangle, img = capture.read()
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cascadeFace.detectMultiScale(grayscale, 1.3, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x,y), (x + w, y + h), (255, 0, 0), 5)

    cv2.imshow("img", img)
    wait = cv2.waitKey(1)
    print(wait)
    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1 or wait == 27:
        break

capture.release()
cv2.destroyAllWindows()