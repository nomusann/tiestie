import cv2

tracker = cv2.legacy.TrackerMedianFlow_create()
tracker_name = str(tracker).split()[0][1:]

# 動画ファイルのパスをここに設定
video_path = "tiesample4.mp4"  # 自分の動画ファイルのパスに置き換えてください

cap = cv2.VideoCapture(video_path)


#webカメラの軌道に時間がかかる場合
import time
time.sleep(1)

ret, frame = cap.read()

roi = cv2.selectROI(frame, False)

ret = tracker.init(frame, roi)

while True:

    ret, frame = cap.read()

    success, roi = tracker.update(frame)

    (x,y,w,h) = tuple(map(int,roi))

    if success:
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0,255,0), 3)
    else :
        cv2.putText(frame, "Tracking failed!!", (500,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

    cv2.imshow(tracker_name, frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27 :
        break

cap.release()
cv2.destroyAllWindows()