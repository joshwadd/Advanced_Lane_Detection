import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from lane_detection import LaneDetection













def main():
    NX = 9
    NY = 6
    h = 720
    w = 1280
    nwindows = 9
    margin = 100
    tol =50

    CALIBRATION_IMAGES = glob.glob('../camera_cal/*')

    lane_detector = LaneDetection(h, w, CALIBRATION_IMAGES, NX, NY, nwindows, margin, tol)

    video_output = '../output_videos/project_video.mp4'
    clip1 = VideoFileClip("../project_video.mp4")

    white_clip = clip1.fl_image(lane_detector.process_frame)
    white_clip.write_videofile(video_output, audio=False)







if __name__ == '__main__':
    main()
