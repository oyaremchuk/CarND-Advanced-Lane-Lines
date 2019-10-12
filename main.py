import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from help import *


def get_distortion_coef():
    try:
        mtx = np.load('mtx.npy')
        dist = np.load('dist.npy')
    except:
        nx = 9
        ny = 6
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        images = glob.glob('camera_cal/*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.save('mtx', mtx)
        np.save('dist', dist)
    return mtx, dist


def get_warp_coef(w, h):
    try:
        M_warp = np.load('M_warp.npy')
    except:
        src_w_top = int(round(w * 0.475))
        src_w_bottom = int(round(w * 0.20))
        src_h_top = int(round(h * 0.650))
        src_h_bottom = int(round(h * 1.00))

        dst_w_top = src_w_top
        dst_w_bottom = dst_w_top
        dst_h_top = int(0)
        dst_h_bottom = int(h)

        src_vertices = np.array([(src_w_top, src_h_top), (w - src_w_top, src_h_top),
                                 (w - src_w_bottom, src_h_bottom), (src_w_bottom, src_h_bottom)], dtype='float32')
        dst_vertices = np.array([(dst_w_top, dst_h_top), (w - dst_w_top, dst_h_top),
                                 (w - dst_w_bottom, dst_h_bottom), (dst_w_bottom, dst_h_bottom)], dtype='float32')

        M_warp = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
        np.save('M_warp', M_warp)
    return M_warp


def find_lane_by_color(image, is_debug=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 175])
    upper_white = np.array([255, 80, 255])
    lower_yello = np.array([15, 50, 50])
    upper_yello = np.array([40, 255, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    yello_mask = cv2.inRange(hsv, lower_yello, upper_yello)

    wy_mask = cv2.bitwise_or(white_mask, yello_mask)

    kernel_dilate = np.ones((9, 9), np.uint8)
    dilated_wy_mask = cv2.dilate(wy_mask, kernel_dilate, iterations=1)

    image_wy = cv2.bitwise_and(image, image, mask=wy_mask)
    return  image_wy


def find_lane_advanced(image):
    return image


def main():
    idv = 2
    videos = glob.glob('*.mp4')
    print('Opening: {}'.format(videos[idv]))
    cap = cv2.VideoCapture(videos[idv])

    if (cap.isOpened() == False):
        print('Error opening video stream or file: {}'.format(videos[idv]))
        return

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mtx, dist = get_distortion_coef()
    M_warp = get_warp_coef(w, h)

    delay = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print('{}/{}'.format(pos, length))

            undist = cv2.undistort(frame, mtx, dist, None, mtx)
            warp = cv2.warpPerspective(undist, M_warp, (w, h), flags=cv2.INTER_LANCZOS4)
            warp = gaussian_blur(warp, 3)

            cv2.imshow('warp', warp)
            lane = find_lane_advanced(warp)
            cv2.imshow('lane', lane)

            key = cv2.waitKey(delay)
            if key == ord('p'):
                delay = 5
            if key == ord('s'):
                delay = 0
            if key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                delay = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
