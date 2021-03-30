#!/bin/bash

arduino-cli compile --fqbn arduino:avr:uno ../steering_angle
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:uno ../steering_angle
