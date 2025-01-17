import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# 動画ファイルのパス
video_path = "tiesample4.mp4"

# 動画キャプチャオブジェクトの作成
cap = cv2.VideoCapture(video_path)

#フレームの幅と高さを取得
width_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("幅:", width_original)
print("高さ:", height_original)

# 低解像画像のサイズ
resize_size = 8
width_resized = width_original/resize_size
height_resized = height_original/resize_size

# ネクタイ色のHSV範囲 
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,60])

# 肌部分の色のHSV範囲 
lower_skin = np.array([0, 45, 90])
upper_skin = np.array([20, 255, 255])

# 何フレームごとに処理をするかのカウンタ
counter = 0

# 一番はじめの処理かどうか
firsttime = 0

#ネクタイの状態の初期値
tiestep = 0
step_counter = 0

# 結果の画像をversion_nameのフォルダに保存
version_name = "linetrace4_4"
folder_path = f"version_name_{version_name}"
os.makedirs(folder_path, exist_ok=True)  # フォルダが既に存在する場合でもエラーを出さない

# 全て白正方形とマスク画像のテンプレートマッチング（ネクタイの太さ認識用）
def find_largest_white_square(img):
    """
    2値画像内で最大の白正方形を見つける

    Args:
        img: 2値化された入力画像

    Returns:
        最大正方形の左上の座標と辺の長さ
    """

    h, w = img.shape[:2]

    # テンプレートの初期化 (小さな正方形から始める)
    template_size = 4
    max_val = 0
    max_loc = (0, 0)

    while template_size < min(h, w):
        # 全て白の正方形テンプレートを作成
        template = np.ones((template_size, template_size), dtype=np.uint8) * 255
        cv2.imshow('template', template)
        # テンプレートマッチング
        res = cv2.matchTemplate(img, template, cv2.TM_CCORR)
        # 最も類似する位置を取得
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # しきい値を超える場合、最大値を更新
        print(str(max_val))
        if max_val < 2010000.0:  # しきい値は適宜調整
            template_size += 1
            print(str(template_size))
        else:
            break

    return max_loc, template_size

# 指定された中心と半径の円形のマスク画像を作成(つかってない)
def create_circle_mask(img_shape, center, radius):
    """
    Args:
        img_shape: 元画像の形状 (高さ, 幅)
        center: 円の中心座標 (x, y)
        radius: 円の半径

    Returns:
        np.ndarray: マスク画像
    """
    circle_mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)
    return circle_mask

# 指定した位置を黒四角形で塗りつぶす
def draw_black_rectangle(img, x, y, width, height):
    """
    指定した位置に黒い四角形を描画する関数

    Args:
        img_path: 入力画像
        x (int): 四角形の左上のx座標
        y (int): 四角形の左上のy座標
        width (int): 四角形の幅
        height (int): 四角形の高
    """
    
    # 黒色で塗りつぶす
    cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 0), -1)

# カラーセンサ
def colorsensor(mask_tie, mask_skin, top_left, rec_size):
    """
    Args:
        mask_tie: ネクタイマスク画像
        mask_ksin: 肌マスク画像
        colorsensor_load: カラーセンサの通った道
        top_left: 正方形の左上角の座標(x,y)
        square_size: 正方形の辺の長さ

    Returns:
        area_number: 1(tie_100%) 2(tie_50%) 3(not_tie) 4(skin) 5(tie_20%) 
    """

    area_number = 0
    # 小さな正方形を作成
    rec_white = np.ones((rec_size, rec_size), dtype=np.uint8) * 255
    # トリミングした元画像（サイズ・位置指定）と全部白正方形論理積
    mask_tie_trim = mask_tie[top_left[1]:top_left[1]+rec_size, top_left[0]:top_left[0]+rec_size]
    mask_skin_trim = mask_skin[top_left[1]:top_left[1]+rec_size, top_left[0]:top_left[0]+rec_size]

    rec_white_shape = rec_white.shape
    mask_tie_trim_shape = mask_tie_trim.shape

    if rec_white_shape == mask_tie_trim_shape:
        result_tie = cv2.bitwise_and(mask_tie_trim, rec_white)
        total_sum_tie = np.sum(result_tie)/255
        result_skin = cv2.bitwise_and(mask_skin_trim, rec_white)
        total_sum_skin = np.sum(result_skin)/255

        if total_sum_skin > 5:
            area_number = 4
        elif total_sum_tie >= 18 and total_sum_tie < 30:
            area_number = 2
        elif total_sum_tie > 5 and total_sum_tie < 18:
            area_number = 5
        elif total_sum_tie >= 30:
            area_number = 1
        else:
            area_number = 3

    return area_number

# 白正方形でマスク画像をスキャン
def scan_binary_image(img, rec_size):
    """
    2値画像をスキャンし、小さな正方形との論理積を計算する関数

    Args:
        img: 入力画像
        square_size (int): 小さな正方形のサイズ

    Returns:
        tie_found_point(x,y): ネクタイを見つけた座標
    """

    # 小さな正方形を作成
    rec = np.ones((rec_size, rec_size), dtype=np.uint8) * 255

    # 画像の高さ、幅を取得
    height, width = img.shape

    counter_x = 0
    counter_y = 0
    tie_found_point = (0,0)
    tie_found = False #ネクタイの端を見つけたらTrueにする

    # ネクタイの端を見つけるまで、6×6の白画像でネクタイの色マスク画像を下からスキャン
    for y in range(height - rec_size, -1, -1): 
        if tie_found == True:
            break
        if counter_y % 3 == 0:    # ここ変えると、1ビット飛ばしとかでスキャンできる
            for x in range(width - rec_size):
                if (counter_x % 2 == 0) :  # ここ変えると、1ビット飛ばしとかでスキャンできる
                    total_sum = 0    
                    roi = img[y:y+rec_size, x:x+rec_size]
                    result = cv2.bitwise_and(roi, rec)
                    total_sum = np.sum(result)
                    if total_sum > 9000: # 1ピクセル255で、論理積の最大値255*36
                        tie_found_point = (x,y)
                        #filename_result = f"{folder_path}/tie_found_point_x{x}_y{y}_{total_sum}.png"
                        #cv2.imwrite(filename_result, result)
                        #filename_mask_cut = f"{folder_path}/scan_mask_over4000_x{x}_y{y}_{total_sum}.png"
                        #cv2.imwrite(filename_mask_cut, result)
                        print(tie_found_point)
                        tie_found = True
                        break
                counter_x += 1
        counter_y += 1
    return tie_found_point

# 元画像に、低解像画像で処理した結果を表示
def write_rec_on_original_img(img_original, resize_size, top_left, rec_size, message):
    """
    元画像に、低解像画像で処理した結果を表示

    Args:
        img_original: 元画像
        resize_size (int): 画像縮小が何分の１か
        top_left: 表示する正方形の左上の座標(x,y)（縮小画像での）
        rec_size: 正方形のサイズ（縮小画像での）
        message: 元画像に表示させたい数字とか

    Returns:
        無いままで良いのでは？
    """
    top_left_for_original = tuple(x * resize_size for x in top_left)
    rec_size_for_original = rec_size*resize_size
    print(top_left)
    print(top_left_for_original)
    cv2.rectangle(img_original, top_left_for_original, 
                (top_left_for_original[0] + rec_size_for_original, 
                top_left_for_original[1] + rec_size_for_original), (0, 0, 255), 2)
    #cv2.putText(frame_original,str(top_left_for_original),(top_left_for_original[0],top_left_for_original[1]),0,0.5,(0,255,255),2,cv2.FILLED)
    cv2.putText(frame_original,str(message),(top_left_for_original[0],top_left_for_original[1]),0,0.5,(0,255,255),2,cv2.FILLED)
    #filename_original = f"{folder_path}/original_{counter}.jpg"   
    #cv2.imwrite(filename_original, img_original)

# テンプレートマッチング
def template_matting():
    # 画像を読み込む
    template = cv2.imread("template.png",cv2.IMREAD_GRAYSCALE)
    h, w = template.shape[:2]
    # テンプレートマッチング
    res = cv2.matchTemplate(color_mask_tie, template, cv2.TM_CCORR)
    # 最も類似する位置を取得
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    write_rec_on_original_img(frame_original, resize_size, max_loc, w, max_val)
    cv2.rectangle(frame_resized,max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
    return max_val

# カラーセンサで探索
def use_colorsensor(scan_result, color_mask_tie, color_mask_skin, max_rec_size, resize_size): 
    #speed = 6
    default_point = scan_result
    counter_colorsensor = 0
    x_speed_per_step = 0
    x_speed_per_step_0 = 0
    y_speed_per_step = 0
    y_speed_per_step_0 = 1
    flag = 0
    center_point_x_min = default_point[0]
    center_point_x_max = default_point[0]
    while True:
        x_speed = x_speed_per_step*2
            # ↑ここの係数、状態１の時だけ１でそれ以降は２だと上手くいく
        y_speed = y_speed_per_step*6

        center_point = (default_point[0] + x_speed, default_point[1] - y_speed)

        # カラーセンサの中心が通ったｘ座標の最大と最小
        if center_point[0] > center_point_x_max:
            center_point_x_max = center_point[0]
        if center_point[0] < center_point_x_min:
            center_point_x_min = center_point[0]

        # カラーセンサ（中心）役
        colorsensor_center = colorsensor(color_mask_tie, color_mask_skin, center_point, max_rec_size)
        # 元画像に、低解像画像で処理した結果を表示
        write_rec_on_original_img(frame_original, resize_size, center_point, max_rec_size,colorsensor_center)

        # カラーセンサ（左）役
        colorsensor_left_point = (center_point[0] - (max_rec_size+0), center_point[1]) 
        colorsensor_left = colorsensor(color_mask_tie, color_mask_skin, colorsensor_left_point, max_rec_size)
        write_rec_on_original_img(frame_original, resize_size, colorsensor_left_point, max_rec_size,colorsensor_left)

        # カラーセンサ（右）役
        colorsensor_right_point = (center_point[0] + (max_rec_size+0), center_point[1]) 
        colorsensor_right = colorsensor(color_mask_tie, color_mask_skin, colorsensor_right_point, max_rec_size)
        write_rec_on_original_img(frame_original, resize_size, colorsensor_right_point, max_rec_size,colorsensor_right)

        if flag == 0:
            # カラーセンサが通った道を黒で塗りつぶす
            draw_black_rectangle(color_mask_tie, center_point[0]-max_rec_size, center_point[1], max_rec_size*3, max_rec_size)
            filename_mask_tie_road = f"{folder_path}/mask_tie_road_{counter}.jpg"    
            cv2.imwrite(filename_mask_tie_road, color_mask_tie)
            cv2.imshow('filename_mask_tie_road', color_mask_tie)
        
        # カラーセンサを左右に動かす
        if (colorsensor_left <= 2) and (colorsensor_center <= 2) and (colorsensor_right >= 3):
            x_speed_per_step_0 -= 1
        elif (colorsensor_left <= 2) and (colorsensor_center >= 3) and (colorsensor_right >= 3):
            x_speed_per_step_0 -= 3
        elif (colorsensor_left >= 3) and (colorsensor_center <= 2) and (colorsensor_right <= 2):
            x_speed_per_step_0 += 1
        elif (colorsensor_left >= 3) and (colorsensor_center >= 3) and (colorsensor_right <= 2):
            x_speed_per_step_0 += 3

        x_speed_per_step += x_speed_per_step_0
        y_speed_per_step += y_speed_per_step_0
        counter_colorsensor += 1
        if counter_colorsensor == 13:
            y_speed_per_step_0 = 0
            y_speed_per_step = 0
            x_speed_per_step = 0
            x_speed_per_step_0 = 0
            x_speed_sub = 10
            other_side = 0
            while flag == 0:
                center_point = (default_point[0] + x_speed_sub, default_point[1])
                colorsensor_sub = colorsensor(color_mask_tie, color_mask_skin, center_point, max_rec_size)
                write_rec_on_original_img(frame_original, resize_size, center_point, max_rec_size,colorsensor_sub)
                # 画面下ギリギリを左右に動いてネクタイの端のもう片方を探す
                # 右に動いて画面の端にたどり着いた（見つけられなかった）時、other_side = 1　として、左側を探す
                if (center_point[0] + max_rec_size*2) > width_resized:
                    other_side = 1
                    x_speed_sub = 0
                # 左に動いて画面の端にたどり着いた時、flag = 1として、whileから出る
                if center_point[0] < max_rec_size*2:
                    flag = 1
                # 右側を探す
                if other_side == 0:
                    x_speed_sub += 3
                # 左側を探す
                elif other_side == 1:
                    x_speed_sub -= 3
                # ネクタイの端を見つけた時、その地点をカラーセンサの探索開始地点とする。flag = 1として、whileから出る
                if colorsensor_sub == 1 or colorsensor_sub == 2:
                    default_point = (center_point[0], center_point[1])
                    y_speed_per_step_0 = 1
                    flag = 1
                
        if counter_colorsensor == 26:
            cv2.putText(frame_original,str(center_point_x_min),(10,10),0,0.5,(0,255,255),2,cv2.FILLED)
            cv2.putText(frame_original,str(center_point_x_max),(10,30),0,0.5,(0,255,255),2,cv2.FILLED)
            break
        
    return center_point_x_min, center_point_x_max

while True:
    counter = counter + 1
    # フレームを読み込む
    ret, frame_original = cap.read()

    if not ret:
        break
    
    if counter%100 == 0:

        # 低解像化
        frame_resized = cv2.resize(frame_original, (int(width_resized), int(height_resized)))
        filename_resized = f"{folder_path}/resized_determine_original_img_{counter}.jpg"
        #cv2.imwrite(filename_resize, frame)

        # HSVに変換
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        filename_hsv = f"{folder_path}/hsv_determine_original_img_{counter}.jpg"    
        cv2.imwrite(filename_hsv, hsv)

        # 色のマスクを作成(ネクタイ)
        color_mask_tie = cv2.inRange(hsv, lower_black, upper_black)
        color_mask_tie_original = color_mask_tie
        filename_mask_tie = f"{folder_path}/mask_tie_{counter}.jpg"    
        cv2.imwrite(filename_mask_tie, color_mask_tie)
        filename_mask_tie_original = f"{folder_path}/mask_tie_original_{counter}.jpg"    
        

        # 色のマスクを作成(肌)
        color_mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        filename_mask_skin = f"{folder_path}/mask_skin_{counter}.jpg"    
        cv2.imwrite(filename_mask_skin, color_mask_skin)
        cv2.imshow('filename_mask_skin', color_mask_skin)

        if firsttime == 0:
            # 最大の白正方形を探す
            max_rec_left_top, max_rec_size = find_largest_white_square(color_mask_tie)
            firsttime = -1

        # ネクタイの下端の座標
        scan_result = scan_binary_image(color_mask_tie,max_rec_size)


        x_min, x_max = use_colorsensor(scan_result, color_mask_tie, color_mask_skin, max_rec_size, resize_size)
        
        # 条件分岐で状態判別
        if tiestep==0 and x_min > 50 and x_max < 95:
            tiestep = 1
            step_counter = 0
        elif tiestep==1 and x_min > 50 and x_max > 110 :
            tiestep = 2
            step_counter = 0
        elif tiestep==2 and x_min < 50 and x_max < 90: 
            tiestep = 3
            step_counter = 0
        elif tiestep==3 and x_min > 50 and x_max > 110:
            tiestep = 4
            step_counter = 0
        elif tiestep==4 and x_min > 65 and x_max < 100:
            tiestep = 5
            step_counter = 0
        elif tiestep==5 and x_min > 65 and x_max > 110:
            tiestep = 6
            step_counter = 0
        elif tiestep==6 and template_matting() > 2000000:
            tiestep = 7
            step_counter = 0
        else:
            cv2.putText(frame_original,"unknown",(40,80),0,0.8,(255,0,0),2,cv2.FILLED)  
        if tiestep == 6:
            val_template_matting = template_matting() 
            cv2.putText(frame_original,str(val_template_matting),(40,160),0,0.8,(255,0,0),2,cv2.FILLED)  
        step_counter += 1
        cv2.putText(frame_original,str(tiestep),(40,120),0,0.8,(255,0,0),2,cv2.FILLED)        
          

        # 結果を表示
        cv2.rectangle(frame_resized,scan_result, (scan_result[0] + max_rec_size, scan_result[1] + max_rec_size), (0, 0, 255), 2)
        cv2.putText(frame_resized,str(max_rec_size),(scan_result[0],scan_result[1]),0,0.5,(0,255,255),2,cv2.FILLED)

        # 結果を表示
        cv2.imshow('frame_original', frame_original)
        cv2.imshow('frame_resized', frame_resized)
        #cv2.imshow('colorsensor_center', colorsensor_center)
        cv2.imshow('color_mask',color_mask_tie)
   
        # 結果の画像を保存
        cv2.imwrite(filename_resized, frame_resized)
        filename_original = f"{folder_path}/original_{counter}.jpg"   
        cv2.imwrite(filename_original, frame_original)
        cv2.imwrite(filename_mask_tie_original, color_mask_tie_original)
    # 'q' キーを押すと終了
    if cv2.waitKey(1) == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()