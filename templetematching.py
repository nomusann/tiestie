import cv2
import numpy as np
import matplotlib.pyplot as plt

#対象画像の読み込み
img = cv2.imread('sample3.jpg')
#テンプレート画像の読み込み
templ = cv2.imread('temp2.jpg')

#cv2.TM_CCOEFF_NORMEDで正規化相互相関演算を行い、結果をresultに格納
result = cv2.matchTemplate(img,                  #対象画像
                           templ,                #テンプレート画像
                           cv2.TM_CCOEFF_NORMED  #類似度の計算方法
                           )

#類似度の閾値
threshold =0.88
#類似度が閾値以上の座標を取得
match_y, match_x = np.where(result >= threshold)

#テンプレート画像のサイズ
w = templ.shape[1]
h = templ.shape[0]

#対象画像をコピー
dst = img.copy()

#マッチした箇所に赤枠を描画
#赤枠の右下の座標は左上の座標（x,y)にテンプレート画像の幅、高さ(w,h）を足す。
for x,y in zip(match_x, match_y):
    cv2.rectangle(dst,        #対象画像
                  (x,y),      #マッチした箇所の左上座標
                  (x+w, y+h), #マッチした箇所の右下座標
                  (0,0,225),  #線の色
                  2           #線の太さ
                  )          


plt.imshow(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
plt.show()
# 結果画像を保存
cv2.imwrite('result.jpg', dst)