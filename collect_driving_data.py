import cv2
import numpy as np
import os
import serial
import csv

class Camera:
    def __init__(self, cam_num):
        self.fWidth  = 1280
        self.fHight = 720
        self.cap = cv2.VideoCapture(cam_num)
        self.cap.set(4, self.fHight)
        self.cap.set(3, self.fWidth)
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.out = None
        self.frame_count = None

    def get_frame(self):
        self.ret, self.im = self.cap.read()
        if self.ret == True:
            # cameara is upsidedown in car so flip
            # flip both axis
            self.im = cv2.flip(self.im, -1)
        return self.ret, self.im

    def save_image(self):
        self.out.write(self.im)
        self.frame_count += 1
        return self.frame_count

    def new_vid(self, vid_save):
        self.out = cv2.VideoWriter(
                f'./driving_data/videos/{vid_save}.mp4',
                self.fourcc,
                20.0,
                (self.fWidth, self.fHight))
        self.frame_count = 0

    def end_vid(self):
        self.out.release()

    def clean_up(self):
        self.cap.release()
        try:
            self.out.release()
        except Exception as e:
            print(e)
            print('Recording alread colsed')

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
        x = int(self.ser.readline().decode())
        # convert from raw value to degress
        # divide by 1200 becase interupt working on change in V
        deg = (x/1200)*360
        return deg

def add_text(im, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    col = (0,0,255)
    org = (0, im.shape[0]-30)
    thinkness = 5
    font_scale = 2
    return cv2.putText(im, text, org, font, font_scale, col, thinkness)

def data_gather_loop(cam, angle):
    print('New vid started')
    vid_count = os.listdir("./driving_data/videos")
    cam.new_vid(len(vid_count))
    csv_file = open(f'./driving_data/csv_files/{len(vid_count)}.csv', 'w', newline='')
    ret = True
    while ret:
        ret, frame = cam.get_frame()
        if ret == True:
            d = angle.get_angle()
            frame_num = cam.save_image()
            csv_file.write(f'{frame_num},{d}\n')
            frame =  add_text(frame, 'Recodring')
            cv2.imshow("Current Frame", frame)
            key = cv2.waitKey(10)
            if key == ord('q') or key == ord('d'):
                cam.end_vid()
                csv_file.close()
                break
    return key
    
if __name__=="__main__":
    angle = SteeringAngle()
    cam = Camera(2)
    key = None
    while key != ord('d'):
        if key != ord('\r'):
            ret, frame = cam.get_frame()
            if ret:
                frame =  add_text(frame, 'Not Recording')
                cv2.imshow('Current Frame', frame)
                key = cv2.waitKey(10)
        else:
            key = data_gather_loop(cam, angle)

    # close camera
    cam.clean_up()
