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

    # 反転
    #gray2 = cv2.bitwise_not(gray)

    # ガウシアンブラーでノイズ除去 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # アダプティブ閾値処理
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)

    # Cannyエッジ検出 
    edges = cv2.Canny(thresh, 50, 150)    

    # HSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 色のマスクを作成
    color_mask = cv2.inRange(hsv, lower_black, upper_black)

    # マスク画像とエッジ画像の論理積
    result = cv2.bitwise_and(color_mask, edges)

    # ハフ変換で直線検出
    lines = cv2.HoughLinesP(result, rho=1, theta=np.pi/360, threshold=35, minLineLength=25, maxLineGap=15)

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
    cv2.imshow('frame', frame)
    
    # 'q' キーを押すと終了
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()