import cv2
import numpy as np

# 動画ファイルのパス
video_path = "tiesample4.mp4"

# 動画キャプチャオブジェクトの作成
cap = cv2.VideoCapture(video_path)

# 検出したい色のHSV範囲 
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,45])

# 過去の直線情報を保存するためのリスト
prev_lines = []
MAX_FRAMES = 4  # 最大で4フレーム分の残像を表示

while True:
    # フレームを読み込む
    ret, frame = cap.read()

    if not ret:
        break

    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cannyエッジ検出
    edges = cv2.Canny(gray, 50, 150)

    # 色のマスクを作成
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower_black, upper_black)

    # 輪郭検出
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭を塗りつぶしたマスクを作成
    contour_mask = np.zeros_like(color_mask)
    cv2.drawContours(contour_mask, contours, -1, 255, -1)

    # マスク画像と輪郭マスクの論理積
    result = cv2.bitwise_and(color_mask, contour_mask)

    # ハフ変換で直線検出
    lines = cv2.HoughLinesP(result, rho=1, theta=np.pi/360, threshold=20, minLineLength=15, maxLineGap=20)

   # マスク画像を作成
    mask = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 3)  # マスク画像に直線を緑で描く
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # フレームに直線を赤で描く

    # 過去のマスク画像と現在のマスク画像を重ね合わせる
    for prev_mask in prev_lines:
        cv2.addWeighted(prev_mask, 0.3, mask, 0.7, 0, mask)

    # 現在のマスク画像をリストに追加 (古いマスク画像を削除)
    prev_lines.append(mask.copy())
    if len(prev_lines) > MAX_FRAMES:
        prev_lines.pop(0)

    # マスク画像を使って元のフレームに重ね合わせる
    cv2.addWeighted(mask, 0.5, frame, 0.5, 0, frame) #ネクタイの動きの確認
    #cv2.addWeighted(mask, 0.9, frame, 0.1, 0, frame) #マスク画像の確認
    #cv2.addWeighted(mask, 0.0, frame, 1, 0, frame) #残像無し画像の確認
    
    # 結果を表示
    cv2.imshow('result', frame)

    
    # 'q' キーを押すと終了
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()