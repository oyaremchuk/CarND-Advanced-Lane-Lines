import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks
import glob
from help import *


def get_distortion_coef(is_new=False):
    try:
        mtx = np.load('mtx.npy')
        dist = np.load('dist.npy')
    finally:
        if ~is_new:
            return mtx, dist
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


def get_warp_coef(w, h, is_new=False):
    try:
        m_warp = np.load('m_warp.npy')
    except:
        pass
    finally:
        if not is_new:
            return m_warp
        src_w_top = int(round(w * 0.475))
        src_w_bottom = int(round(w * 0.20))
        src_h_top = int(round(h * 0.650))
        src_h_bottom = int(round(h * 0.95))

        dst_w_top = int(round(w * 0.475))
        dst_w_bottom = int(round(w * 0.450))
        dst_h_top = int(0)
        dst_h_bottom = int(h)

        src_vertices = np.array([(src_w_top, src_h_top), (w - src_w_top, src_h_top),
                                 (w - src_w_bottom, src_h_bottom), (src_w_bottom, src_h_bottom)], dtype='float32')
        dst_vertices = np.array([(dst_w_top, dst_h_top), (w - dst_w_top, dst_h_top),
                                 (w - dst_w_bottom, dst_h_bottom), (dst_w_bottom, dst_h_bottom)], dtype='float32')

        m_warp = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
        np.save('m_warp', m_warp)
    return m_warp


def find_lane_mask_by_hsv_color(hsv, is_morpho_apply=True):
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 55, 255])
    lower_yello = np.array([15, 50, 100])
    upper_yello = np.array([45, 255, 255])

    white = cv2.inRange(hsv, lower_white, upper_white)
    yello = cv2.inRange(hsv, lower_yello, upper_yello)

    wy = cv2.bitwise_or(white, yello)
    binary_output = np.zeros_like(wy)
    binary_output[wy > 0] = 1
    if is_morpho_apply:
        binary_output = apply_morpho(binary_output)
    return binary_output


def find_mag_mask_by_thresh(data, sobel_kernel=3, m_thresh=(0, 255), is_morpho_apply=True):
    sobelx = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= m_thresh[0]) & (gradmag <= m_thresh[1])] = 1
    if is_morpho_apply:
        binary_output = apply_morpho(binary_output)
    return binary_output


def apply_morpho(data):
    kernel = np.ones((5, 5), np.uint8)
    result = data
    result = cv2.dilate(data, kernel, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


class Lane:
    def __init__(self, w, h, is_left, window_w=20, window_h=10):
        self.is_left = is_left
        self.w = w
        self.h = h
        self.m_w = int(w // 2)
        self.m_h = int(h // 2)
        self.window_w = window_w
        self.window_h = window_h
        self.line_poly = None

    def _find_start_peak(self, data, threshold=5):
        histogram = np.sum(data, axis=0)
        histogram[histogram < threshold] = 0
        peaks, _ = find_peaks(histogram)

        if peaks.size != 0:
            if self.is_left:
                return peaks[-1]
            else:
                return peaks[0]
        return None

    def _find_peak_w(self, data, threshold=5):
        histogram = np.sum(data, axis=0)
        histogram[histogram < threshold] = 0
        peaks, _ = find_peaks(histogram)
        left = peaks[peaks <= self.window_w]
        right = peaks[peaks > self.window_w]

        if left.size and right.size:
            if self.window_w - left[-1] > right[0] - self.window_w:
                peak = right[0]
            else:
                peak = left[-1]
        if left.size:
            peak = left[-1]
        elif right.size:
            peak = right[0]
        else:
            return None
        return peak

    def find(self, data):
        if self.is_left:
            center_w = self._find_start_peak(data[self.m_h:, :self.m_w], 50)
        else:
            center_w = self._find_start_peak(data[self.m_h:, self.m_w:], 50)

        if center_w is None:
            return

        if not self.is_left:
            center_w += self.m_w

        center_h = self.h - self.window_h
        line_x_coordinates = []
        line_y_coordinates = []

        for i in range(0, int(self.h / self.window_h)):
            part = data[center_h-self.window_h:center_h+self.window_h, center_w-self.window_w:center_w+self.window_w]
            offset_w = self._find_peak_w(part)
            center_h -= self.window_h
            if offset_w is not None:
                offset_w -= self.window_w
                center_w = center_w + offset_w
            line_x_coordinates.append(center_w)
            line_y_coordinates.append(center_h)
        if line_x_coordinates is not None:
            line_poly = np.polyfit(line_y_coordinates, line_x_coordinates, 2)
            self.line_poly = np.poly1d(line_poly)


def main():
    idv = 0
    frame_offset = 0
    videos = glob.glob('*.mp4')
    print('Opening: {}'.format(videos[idv]))
    cap = cv2.VideoCapture(videos[idv])

    if not cap.isOpened():
        print('Error opening video stream or file: {}'.format(videos[idv]))
        return

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output_videos/{}.avi'.format(videos[idv]), fourcc, 20.0, (w, h))
    mtx, dist = get_distortion_coef()
    m_warp = get_warp_coef(w, h)

    window_w = 30
    window_h = 15
    left = Lane(w, h, window_w=window_w, window_h=window_h, is_left=True)
    right = Lane(w, h, window_w=window_w, window_h=window_h, is_left=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
    delay = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print('{}/{}'.format(pos, length))

            undist = cv2.undistort(frame, mtx, dist, None, mtx)
            undist = gaussian_blur(undist, 3)
            hsv = cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)

            color_mask = find_lane_mask_by_hsv_color(hsv)
            mag_mask = find_mag_mask_by_thresh(hsv[:, :, 2], 15, (10, 255))
            lane_or_mag_mask = cv2.bitwise_or(color_mask, mag_mask)
            lane_and_mag_mask = cv2.bitwise_and(color_mask, mag_mask)

            warp_lane_or_mag_mask = cv2.warpPerspective(lane_or_mag_mask, m_warp, (w, h))
            warp_lane_and_mag_mask = cv2.warpPerspective(lane_and_mag_mask, m_warp, (w, h))
            cv2.imshow('warp_lane_and_mag_mask', warp_lane_and_mag_mask*255)

            left.find(warp_lane_and_mag_mask)
            right.find(warp_lane_and_mag_mask)

            lane_img = np.zeros((h, w, 3), dtype=np.uint8)
            if left.line_poly is not None and left.line_poly is not None:
                yp = np.linspace(0, h-1, h)
                xpl = np.rint(left.line_poly(yp))
                xpr = np.rint(right.line_poly(yp))
                left_pts = np.vstack((xpl, yp)).astype(np.int32).T
                right_pts = np.vstack((xpr, yp)).astype(np.int32).T
                cv2.fillConvexPoly(lane_img, np.concatenate((left_pts, right_pts[::-1])), [0, 255, 255])
                cv2.polylines(lane_img, [left_pts], False, (255, 0, 0), 5)
                cv2.polylines(lane_img, [right_pts], False, (0, 0, 255), 5)
            lane_img = cv2.warpPerspective(lane_img, m_warp, (w, h), flags=cv2.WARP_INVERSE_MAP)
            lane_img = cv2.addWeighted(undist, 1, lane_img, 0.3, 0)
            cv2.imshow('out', lane_img)
            out.write(lane_img)
            key = cv2.waitKey(delay)
            if key == ord('p'):
                delay = 5
            if key == ord('s'):
                delay = 0
            if key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
            if key == ord('d'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)-2)
                delay = 0
        else:
            cap.release()
            out.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
