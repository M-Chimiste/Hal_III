import numpy as np
import cv2

for i in range(2,399):
    data = np.load(f"game_data/{i}.npy")
    #print(data)
    cv2.imshow("", cv2.resize(data, (0,0), fx=20, fy=20))
    cv2.waitKey(30)