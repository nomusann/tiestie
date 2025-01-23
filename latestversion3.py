import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# カメラのデバイス番号（通常0）
camera_index = 0

# カメラキャプチャオブジェクトの作成
#cap = cv2.VideoCapture(camera_index)
# 希望の幅と高さ
#desired_width = 1280
#desired_height = 720
# フレームの幅と高さを設定
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# 動画ファイルのパス
video_path = "tiesample4.mp4"

# 動画キャプチャオブジェクトの作成
cap = cv2.VideoCapture(video_path)

#フレームの幅と高さを取得
width_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
cross = 0
lpoint_out = (0,0)
rpoint_out = (0,0)
step5_crosspoint = (0,0)

# 結果の画像をversion_nameのフォルダに保存
version_name = "linetrace4_6"
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
        if max_val < 2010000.0:  # しきい値は適宜調整
            template_size += 1
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

# 指定した位置を白四角形で塗りつぶす
def draw_white_rectangle(img, x, y, width, height):
    """
    指定した位置に黒い四角形を描画する関数

    Args:
        img_path: 入力画像
        x (int): 四角形の左上のx座標
        y (int): 四角形の左上のy座標
        width (int): 四角形の幅
        height (int): 四角形の高
    """
    
    # 白色で塗りつぶす
    cv2.rectangle(img, (x, y), (x+width, y+height), (255, 255, 255), -1)

# カラーセンサ
def colorsensor(mask_tie, mask_skin, mask_road, top_left, rec_size):
    """
    Args:
        mask_tie: ネクタイマスク画像
        mask_ksin: 肌マスク画像
        colorsensor_load: カラーセンサの通った道
        top_left: 正方形の左上角の座標(x,y)
        square_size: 正方形の辺の長さ

    Returns:
        area_number: 1(tie_100%) 2(tie_50%) 3(not_tie) 4(skin) 5(tie_20%) 6(tie road on tie)
    """

    area_number = 0
    # 小さな正方形を作成
    rec_white = np.ones((rec_size, rec_size), dtype=np.uint8) * 255
    # トリミングした元画像（サイズ・位置指定）と全部白正方形論理積
    mask_tie_trim = mask_tie[top_left[1]:top_left[1]+rec_size, top_left[0]:top_left[0]+rec_size]
    mask_skin_trim = mask_skin[top_left[1]:top_left[1]+rec_size, top_left[0]:top_left[0]+rec_size]
    mask_road_trim = mask_road[top_left[1]:top_left[1]+rec_size, top_left[0]:top_left[0]+rec_size]


    rec_white_shape = rec_white.shape
    mask_tie_trim_shape = mask_tie_trim.shape

    if rec_white_shape == mask_tie_trim_shape:
        result_tie = cv2.bitwise_and(mask_tie_trim, rec_white)
        total_sum_tie = np.sum(result_tie)/255
        result_skin = cv2.bitwise_and(mask_skin_trim, rec_white)
        total_sum_skin = np.sum(result_skin)/255

    total_sum_road = 0
    if mask_road_trim.shape == rec_white.shape:
        print("mask_road_trim")
        print(mask_road_trim.shape)
        print("rec_white")
        print(rec_white.shape)
        result_road = cv2.bitwise_and(mask_road_trim, rec_white)
        total_sum_road = np.sum(result_road)/255

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
        
        if total_sum_road > 15:
            area_number = 6

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
def use_colorsensor(scan_result, color_mask_tie, color_mask_skin, color_mask_road, max_rec_size, resize_size): 
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
    cross_score = 0
    most_cross_score = 0
    most_cross_point = (0,0)
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
        colorsensor_center = colorsensor(color_mask_tie, color_mask_skin, color_mask_raod, center_point, max_rec_size)
        # 元画像に、低解像画像で処理した結果を表示
        write_rec_on_original_img(frame_original, resize_size, center_point, max_rec_size,colorsensor_center)

        # カラーセンサ（左）役
        colorsensor_left_point = (center_point[0] - (max_rec_size+0), center_point[1]) 
        colorsensor_left = colorsensor(color_mask_tie, color_mask_skin, color_mask_raod, colorsensor_left_point, max_rec_size)
        write_rec_on_original_img(frame_original, resize_size, colorsensor_left_point, max_rec_size,colorsensor_left)

        # カラーセンサ（右）役
        colorsensor_right_point = (center_point[0] + (max_rec_size+0), center_point[1]) 
        colorsensor_right = colorsensor(color_mask_tie, color_mask_skin, color_mask_raod, colorsensor_right_point, max_rec_size)
        write_rec_on_original_img(frame_original, resize_size, colorsensor_right_point, max_rec_size,colorsensor_right)

        if flag == 0:
            # カラーセンサが通った道を黒で塗りつぶす
            draw_black_rectangle(color_mask_tie, center_point[0]-max_rec_size, center_point[1], max_rec_size*3, max_rec_size)
            #filename_mask_tie_road = f"{folder_path}/mask_tie_road_{counter}.jpg"    
            #cv2.imwrite(filename_mask_tie_road, color_mask_tie)
            #cv2.imshow('filename_mask_tie_road', color_mask_tie)
            # カラーセンサが通った道をマスクする
            draw_white_rectangle(color_mask_road, center_point[0]-max_rec_size, center_point[1], max_rec_size*3, max_rec_size)
            filename_colorsensor_road_mask = f"{folder_path}/colorsensor_road_mask_{counter}.jpg"   
            cv2.imwrite(filename_colorsensor_road_mask, color_mask_road)
            cv2.imshow('filename_colorsensor_road_mask', color_mask_road)
        
        # カラーセンサを左右に動かす
        if (colorsensor_left <= 2) and (colorsensor_center <= 2) and (colorsensor_right >= 3):
            x_speed_per_step_0 -= 1
        elif (colorsensor_left <= 2) and (colorsensor_center >= 3) and (colorsensor_right >= 3):
            x_speed_per_step_0 -= 3
        elif (colorsensor_left >= 3) and (colorsensor_center <= 2) and (colorsensor_right <= 2):
            x_speed_per_step_0 += 1
        elif (colorsensor_left >= 3) and (colorsensor_center >= 3) and (colorsensor_right <= 2):
            x_speed_per_step_0 += 3

        if counter_colorsensor == 4:
            checkpoint1 = center_point
        if counter_colorsensor == 17:
            checkpoint2 = center_point
        
        if colorsensor_left == 6 and colorsensor_center == 6 and colorsensor_right == 6:
            cross_score = 3
        elif (colorsensor_left == 6 and colorsensor_center) or (colorsensor_center == 6 and colorsensor_right == 6):
            cross_score = 2
        #elif colorsensor_center == 6:
        #    cross_score = 1

        if most_cross_score < cross_score:
            most_cross_point = ((center_point[0]+20)*resize_size, (center_point[1]+20)*resize_size)
        
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
                colorsensor_sub = colorsensor(color_mask_tie, color_mask_skin, color_mask_raod, center_point, max_rec_size)
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
                if colorsensor_sub == 1 or colorsensor_sub == 2 or colorsensor_sub == 5:
                    default_point = (center_point[0], center_point[1])
                    y_speed_per_step_0 = 1
                    flag = 1
                
        if counter_colorsensor == 26:
            if checkpoint1[0]<checkpoint2[0]:
                leftpoint = (checkpoint1[0]*resize_size, checkpoint1[1]*resize_size)
                rightpoint = (checkpoint2[0]*resize_size, checkpoint2[1]*resize_size)
            else:
                leftpoint = (checkpoint2[0]*resize_size, checkpoint2[1]*resize_size)
                rightpoint = (checkpoint1[0]*resize_size, checkpoint1[1]*resize_size)
            cv2.putText(frame_original,str(center_point_x_min),(10,10),0,0.5,(0,255,255),2,cv2.FILLED)
            cv2.putText(frame_original,str(center_point_x_max),(10,30),0,0.5,(0,255,255),2,cv2.FILLED)
            break
        
    return center_point_x_min, center_point_x_max, leftpoint, rightpoint, most_cross_point

# 直線矢印描画
def draw_arrow_straight(frame, current_time, point1, point2):
    """
    指定された2つの座標を使用して、フレーム上に円弧と矢印を描画する関数。

    :param frame: 現在のフレーム
    :param current_time: 現在の時間 (秒単位)
    :param point1: 座標1 (x1, y1)
    :param point2: 座標2 (x2, y2)
    :return: 描画済みのフレーム
    """

    # 矢印の先端部分を描画する関数
    def draw_arrow_head(frame, start_point, end_point, color, thickness):
        direction = np.array(end_point) - np.array(start_point)
        length = np.linalg.norm(direction)
        if length == 0:
            return

        direction = direction / length
        tip_length = 30
        side_length = 15
        perpendicular = np.array([-direction[1], direction[0]])

        tip_point = np.array(end_point)
        left_point = tip_point - tip_length * direction + side_length * perpendicular
        right_point = tip_point - tip_length * direction - side_length * perpendicular

        cv2.line(frame, tuple(tip_point.astype(int)), tuple(left_point.astype(int)), color, thickness)
        cv2.line(frame, tuple(tip_point.astype(int)), tuple(right_point.astype(int)), color, thickness)

    # 直線を描画
    cv2.line(frame, point1, point2, (0, 0, 255), 3)

    # 矢印の先端を描画
    draw_arrow_head(frame, point1, point2, (0, 0, 255), 5)  

    return frame


# 矢印を描画
def draw_arrow(frame, current_time, point1, point2):
    """
    指定された2つの座標を使用して、フレーム上に円弧と矢印を描画する関数。

    :param frame: 現在のフレーム
    :param current_time: 現在の時間 (秒単位)
    :param point1: 座標1 (x1, y1)
    :param point2: 座標2 (x2, y2)
    :return: 描画済みのフレーム
    """
    dir = -1 if point1[0] > point2[0] else 1

    # 矢印の先端部分を描画する関数
    def draw_arrow_head(frame, start_point, end_point, color, thickness):
        direction = np.array(end_point) - np.array(start_point)
        length = np.linalg.norm(direction)
        if length == 0:
            return

        direction = direction / length
        tip_length = 30
        side_length = 15
        perpendicular = np.array([-direction[1], direction[0]])

        tip_point = np.array(end_point)
        left_point = tip_point - tip_length * direction + side_length * perpendicular
        right_point = tip_point - tip_length * direction - side_length * perpendicular

        cv2.line(frame, tuple(tip_point.astype(int)), tuple(left_point.astype(int)), color, thickness)
        cv2.line(frame, tuple(tip_point.astype(int)), tuple(right_point.astype(int)), color, thickness)

    x1, y1 = point1
    x2, y2 = point2
    th = 90
    theta = math.radians(th)

    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r = d / (2 * math.sin(theta / 2))
    r2 = int(r)

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    vx, vy = x2 - x1, y2 - y1
    px, py = -vy, vx
    delta = math.sqrt(r**2 - (d**2) / 4)

    norm = math.sqrt(px**2 + py**2)
    h = mx - delta * (px / norm)
    k = my - delta * (py / norm)
    h2, k2 = int(h), int(k)

    angle_start = int(math.degrees(math.atan2(y1 - k2, x1 - h2)))
    angle_end = int(math.degrees(math.atan2(y2 - k2, x2 - h2)))
    speed = 0.7
    angle_end -= (int(speed * th * current_time) % (angle_end - angle_start))

    cv2.ellipse(frame, (h2, k2), (r2, r2), 0, angle_start, angle_end, (0, 0, 255), 3)

    rad = np.radians(angle_end)
    end_x = int(h2 + r2 * np.cos(rad))
    end_y = int(k2 + r2 * np.sin(rad))

    if end_x - h2 == 0:
        b = 0
    else:
        a = ((end_y - k2) / (end_x - h2))
        b = -1 / a if a != 0 else 1e5

    bec = (-1 * dir * 100, int(-1 * dir * 100 * b))
    start_point = tuple(np.add((end_x, end_y), bec))
    end_point = (end_x, end_y)

    draw_arrow_head(frame, start_point, end_point, (0, 0, 255), 5)

    return frame

# 画像を表示
def embed_image(background_frame, overlay_image_path, position=(0, 0), overlay_size=(100, 150), text=None, text_position=(0, 0), text_color=(255, 255, 255), text_font=cv2.FONT_HERSHEY_SIMPLEX, text_size=1, text_thickness=2):
    """
    背景画像にオーバーレイ画像を埋め込み、その上に文字を描画する関数。

    :param background_frame: 背景画像 (フレーム)
    :param overlay_image_path: オーバーレイ画像のファイルパス
    :param position: オーバーレイ画像を配置する位置 (x, y)
    :param overlay_size: オーバーレイ画像のリサイズサイズ (width, height)
    :param text: 画像上に描画するテキスト
    :param text_position: テキストを配置する位置 (x, y)
    :param text_color: テキストの色 (BGR)
    :param text_font: フォント
    :param text_size: フォントサイズ
    :param text_thickness: フォントの太さ
    :return: 画像が埋め込まれた背景画像
    """
    # オーバーレイ画像を読み込む
    overlay_image = cv2.imread(overlay_image_path)

    # 画像が読み込めたか確認
    if overlay_image is None:
        print(f"画像の読み込みに失敗しました: {overlay_image_path}")
        return background_frame  # 背景フレームをそのまま返す

    # オーバーレイ画像を指定サイズにリサイズ
    overlay_image = cv2.resize(overlay_image, overlay_size)

    # オーバーレイ画像の高さと幅を取得
    h, w = overlay_image.shape[:2]

    # 配置位置のx, y座標を取得
    x, y = position

    # 背景にオーバーレイ画像を埋め込む (画像の部分コピー)
    overlay_frame = background_frame.copy()  # 背景フレームをコピー
    overlay_frame[y:y+h, x:x+w] = overlay_image  # 画像を指定位置に埋め込む

    # テキストを描画する (もしtextが指定されていれば)
    if text:
        cv2.putText(overlay_frame, text, text_position, text_font, text_size, text_color, text_thickness)

    return overlay_frame

while True:
    counter = counter + 1
    # フレームを読み込む
    ret, frame_original = cap.read()

    if not ret:
        break
    
    if counter%25 == 0:

        # 低解像化
        frame_resized = cv2.resize(frame_original, (int(width_resized), int(height_resized)))
        #filename_resized = f"{folder_path}/resized_determine_original_img_{counter}.jpg"
        #cv2.imwrite(filename_resize, frame)

        # HSVに変換
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        #filename_hsv = f"{folder_path}/hsv_determine_original_img_{counter}.jpg"    
        #cv2.imwrite(filename_hsv, hsv)

        # 色のマスクを作成(ネクタイ)
        color_mask_tie = cv2.inRange(hsv, lower_black, upper_black)
        #color_mask_tie_original = color_mask_tie
        #filename_mask_tie = f"{folder_path}/mask_tie_{counter}.jpg"    
        #cv2.imwrite(filename_mask_tie, color_mask_tie)
        #filename_mask_tie_original = f"{folder_path}/mask_tie_original_{counter}.jpg"    

        # 色のマスクを作成(肌)
        color_mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        #filename_mask_skin = f"{folder_path}/mask_skin_{counter}.jpg"    
        #cv2.imwrite(filename_mask_skin, color_mask_skin)
        #cv2.imshow('filename_mask_skin', color_mask_skin)

        # 一回カラーセンサが通った道のマスクをこの変数に後から書き足す
        color_mask_raod = np.zeros((int(height_resized), int(width_resized)), dtype=np.uint8) * 255

        if firsttime == 0:
            # 最大の白正方形を探す
            max_rec_left_top, max_rec_size = find_largest_white_square(color_mask_tie)
            firsttime = -1

        # ネクタイの下端の座標
        scan_result = scan_binary_image(color_mask_tie,max_rec_size)

        x_min, x_max, lpoint, rpoint, crosspoint = use_colorsensor(scan_result, color_mask_tie, color_mask_skin, color_mask_raod, max_rec_size, resize_size)
        lpoint_out = (lpoint[0], lpoint[1])
        rpoint_out = (rpoint[0], rpoint[1])

        # 条件分岐で状態判別
        if tiestep==0:
            if x_min > 50 and x_max < 95:
                tiestep = 1
                step_counter = 0
        elif tiestep==1:  
            if x_min > 50 and x_max > 110 :
                tiestep = 2
                cross = 0
                step_counter = 0
        elif tiestep==2:
            if x_min < 50 and x_max < 90: 
                tiestep = 3
                cross = 0
                step_counter = 0
        elif tiestep==3:
            if x_min > 50 and x_max > 110:
                tiestep = 4
                step_counter = 0
        elif tiestep==4:
            if x_min > 65 and x_max < 100:
                tiestep = 5
                step_counter = 0
        elif tiestep==5:
            if x_min > 65 and x_max > 110:
                tiestep = 6
                step_counter = 0
        elif tiestep==6:
            if template_matting() > 2000000:
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
        #cv2.imshow('frame_original', frame_original)
        #cv2.imshow('frame_resized', frame_resized)
        #cv2.imshow('colorsensor_center', colorsensor_center)
        #cv2.imshow('color_mask',color_mask_tie)
   
        # 結果の画像を保存
        #cv2.imwrite(filename_resized, frame_resized)
        filename_original = f"{folder_path}/original_{counter}.jpg"  
        cv2.imwrite(filename_original, frame_original)
        #cv2.imwrite(filename_mask_tie_original, color_mask_tie_original)
    if tiestep==0:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if rpoint_out[0]>0:
            draw_arrow(frame_original, current_time,lpoint_out, (rpoint_out[0]+200,rpoint_out[1]))
        image_path = f"img/img{tiestep+2}.jpg"  # 挿入したい画像のパス
        frame_original = embed_image(
            frame_original, image_path, position=(frame_original.shape[1] - 250, 50), overlay_size=(180, 230),
            text="NEXT STEP!!", text_position=(frame_original.shape[1] - 280, 40), text_color=(0, 0, 0), text_size=1.3, text_thickness=3
        )
    elif tiestep==1:
        if rpoint[0]+50<lpoint[0] and cross == 0:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            draw_arrow(frame_original, current_time,lpoint_out, (rpoint_out[0]+200,rpoint_out[1]))  
        else:
            cross = 1
        image_path = f"img/img{tiestep+1}.jpg"  # 挿入したい画像のパス
        frame_original = embed_image(
            frame_original, image_path, position=(frame_original.shape[1] - 250, 50), overlay_size=(180, 230),
            text="NEXT STEP!!", text_position=(frame_original.shape[1] - 280, 40), text_color=(0, 0, 0), text_size=1.3, text_thickness=3
        )
    elif tiestep==2:
        if lpoint[0]+50<rpoint[0] and cross == 0:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            draw_arrow(frame_original, current_time, rpoint_out, (lpoint_out[0]-100,lpoint_out[1]))
        else:
            cross = 1
        image_path = f"img/img{tiestep+1}.jpg"  # 挿入したい画像のパス
        frame_original = embed_image(
            frame_original, image_path, position=(frame_original.shape[1] - 250, 50), overlay_size=(180, 230),
            text="NEXT STEP!!", text_position=(frame_original.shape[1] - 280, 40), text_color=(0, 0, 0), text_size=1.3, text_thickness=3
        )
    elif tiestep==3:
        if lpoint[0]+50<rpoint[0] and cross == 0:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            draw_arrow(frame_original, current_time, lpoint_out, (rpoint_out[0]+100,rpoint_out[1]))
        else:
            cross = 1
        image_path = f"img/img{tiestep+1}.jpg"  # 挿入したい画像のパス
        frame_original = embed_image(
            frame_original, image_path, position=(frame_original.shape[1] - 250, 50), overlay_size=(180, 230),
            text="NEXT STEP!!", text_position=(frame_original.shape[1] - 280, 40), text_color=(0, 0, 0), text_size=1.3, text_thickness=3
        )
    elif tiestep==4:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if crosspoint[0]>50 and crosspoint[0]<rpoint_out[0] and crosspoint[1]<rpoint_out[1] :
            draw_arrow_straight(frame_original, current_time, rpoint_out, crosspoint)
            step5_crosspoint = crosspoint
        image_path = f"img/img{tiestep+1}.jpg"  # 挿入したい画像のパス
        frame_original = embed_image(
            frame_original, image_path, position=(frame_original.shape[1] - 250, 50), overlay_size=(180, 230),
            text="NEXT STEP!!", text_position=(frame_original.shape[1] - 280, 40), text_color=(0, 0, 0), text_size=1.3, text_thickness=3
        )
    elif tiestep==5:
        draw_arrow_straight(frame_original, current_time, (step5_crosspoint[0]-50, step5_crosspoint[1]-50), step5_crosspoint)
        image_path = f"img/img{tiestep+1}.jpg"  # 挿入したい画像のパス
        frame_original = embed_image(
            frame_original, image_path, position=(frame_original.shape[1] - 250, 50), overlay_size=(180, 230),
            text="NEXT STEP!!", text_position=(frame_original.shape[1] - 280, 40), text_color=(0, 0, 0), text_size=1.3, text_thickness=3
        )
    elif tiestep==6:
        image_path = f"img/img{tiestep+1}.jpg"  # 挿入したい画像のパス
        frame_original = embed_image(
            frame_original, image_path, position=(frame_original.shape[1] - 250, 50), overlay_size=(180, 230),
            text="NEXT STEP!!", text_position=(frame_original.shape[1] - 280, 40), text_color=(0, 0, 0), text_size=1.3, text_thickness=3
        )
    else:
        cv2.putText(frame_original,"unknown",(40,80),0,0.8,(255,0,0),2,cv2.FILLED)  
    
    cv2.imshow('frame_original', frame_original)
    
    # 'q' キーを押すと終了
    if cv2.waitKey(1) == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()