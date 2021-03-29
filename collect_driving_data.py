import cv2
import numpy as np
import os
import serial
import csv

class Camera:
    def __init__(self):
        self.cap = None
        self.fWidth  = 1280
        self.fHight = 720
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.out = None
        self.frame_count = None

    def get_frame(self):
        self.ret, self.im = self.cap.read()
        if self.ret == True:
            # cameara is upsidedown in car so flip
            # flip both axis
            self.im = cv2.flip(self.im, -1)
            self.out.write(self.im)
            self.frame_count += 1
        return self.ret, self.im, self.frame_count

    def new_vid(self, cam_num, vid_save):
        self.cap = cv2.VideoCapture(cam_num)
        self.cap.set(4, self.fHight)
        self.cap.set(3, self.fWidth)
        self.out = cv2.VideoWriter(
                f'./driving_data/videos/{vid_save}.mp4',
                self.fourcc,
                20.0,
                (self.fWidth, self.fHight))
        self.frame_count = 0

    def clean_up(self):
        self.cap.release()
        self.out.release()

class SteeringAngle:
    '''
    The Aurdino sorts out the midpoint here we just need
    to inform the controller of the min and max points.
    This is done throgh truning the Steering wheel to full
    locks each side.
    '''
    def __init__(self):
        # Set up serial comuncation
        self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

        # get center val ie 0 degs is straiget
        input('turn stering to full left lock')
        self.ser.write(b'2')
        input('turn stering to full right lock')
        self.ser.write(b'3')

    def get_angle(self):
        self.ser.write(b'1')
        # remove the newline simbal for printing
        x = int(self.ser.readline().decode())
        # convert from raw value to degress
        # divide by 1200 becase interupt working on change in V
        deg = (x/1200)*360
        print(deg)
        return deg

if __name__=="__main__":
    sa = SteeringAngle()
    cam = Camera()
    key = None
    while key != ord('d'):
        input('press enter to begin')
        vid_count = os.listdir("./driving_data/videos")
        cam.new_vid(2, len(vid_count))
        csv_file = open(f'./driving_data/csv_files/{len(vid_count)}.csv', 'w', newline='')
        ret = True
        while ret:
            ret, f, f_num = cam.get_frame()
            if ret == True:
                d = sa.get_angle()
                csv_file.write(f'{f_num},{d}\n')
                cv2.imshow("Current Frame", f)
                key = cv2.waitKey(30)
                if key == ord('q') or key == ord('d'):
                    break
        # close camera
        cam.clean_up()
        csv_file.close()
