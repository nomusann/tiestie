import cv2
import numpy as np

# 動画ファイルのパス
video_path = "tiesample4.mp4"

# 動画キャプチャオブジェクトの作成
cap = cv2.VideoCapture(video_path)

# 検出したい色のHSV範囲 
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,45])

while True:
    # フレームを読み込む
    ret, frame = cap.read()

    if not ret:
        break

    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cannyエッジ検出
    edges = cv2.Canny(gray, 50, 150)

    # 輪郭検出
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭を元の画像に描画
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

     # 結果を表示
    cv2.imshow('frame', frame)
    
    # 'q' キーを押すと終了
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()