import serial
import time

# Set up serial comuncation
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

# get center val ie 0 degs is straiget
input('turn stering to full left lock')
ser.write(b'2')
input('turn stering to full right lock')
ser.write(b'3')

while 1:
    ser.write(b'1')
    # remove the newline simbal for printing
    x = int(ser.readline().decode())
    # convert from raw value to degress
    # divide by 1200 becase interupt working on change in V
    deg = (x/1200)*360
    print(deg)
