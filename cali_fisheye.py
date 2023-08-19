import cv2
import numpy as np
import os

# Create the output file
fout = open("calibration_result.txt", "w")

# Set the parameters
imageFolderPath = "image_path/images/"
print("开始提取角点………………")
image_count = 14
board_size = (6, 9)
corners_Seq = []
image_Seq = []
successImageNum = 0
count = 0
f = 0

# Loop through image files
for entry in os.scandir(imageFolderPath):
    imageFilePath = entry.path
    print("Processing:", imageFilePath)
    image = cv2.imread(imageFilePath)
    if image is None:
        print("Failed to read image:", imageFilePath)
        continue
    
    # Extract corners
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patternfound, corners = cv2.findChessboardCorners(imageGray, board_size, None)
    if not patternfound:
        print("can not find chessboard corners!")
        continue
    
    # Refine corners
    corners = cv2.cornerSubPix(imageGray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1))
    
    # Draw corners
    imageTemp = image.copy()
    for corner in corners:
        cv2.circle(imageTemp, tuple(corner[0]), 10, (0, 0, 255), 2, 8, 0)
    f += 1
    imageFileName = f"image_path/image_2/{f}_corner.jpg"
    cv2.imwrite(imageFileName, imageTemp)

    count += len(corners)
    successImageNum += 1
    corners_Seq.append(corners)
    image_Seq.append(image)

print("角点提取完成！")

# Calibration
print("开始鱼眼相机标定………………")
square_size = 20  # 单位：毫米
object_Points = []
img_points = []
obj_p = np.zeros((1, board_size[0]*board_size[1], 3), np.float32)
obj_p[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
for _ in range(successImageNum):
    object_Points.append(object_Points)

object_Points = np.array(object_Points, dtype=np.float32)
# corners_Seq = np.array(corners_Seq, dtype=np.float32)

intrinsic_matrix = np.zeros((3, 3))
distortion_coeffs = np.zeros(4)
rotation_vectors = []
translation_vectors = []
flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
image_size = image_Seq[0].shape[1::-1]

# 鱼眼相机标定

rms, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            object_Points,
            corners_Seq,
            image_size,
            intrinsic_matrix,
            distortion_coeffs,
            rotation_vectors,
            translation_vectors,
            flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

Camera_intrinsic = {"mtx": mtx, "dist": dist}

print("相机内参矩阵和畸变系数：")
print(Camera_intrinsic)
print("鱼眼相机标定完成！")

# 鱼眼图像矫正
print("开始图像畸变矫正………………")
for i, image in enumerate(image_Seq):
    undistorted_image = cv2.fisheye.undistortImage(image, mtx, dist)
    undistorted_file_name = f"image_path/undistorted_{i + 1}.jpg"
    cv2.imwrite(undistorted_file_name, undistorted_image)
print("图像畸变矫正完成！")

print("Done")
