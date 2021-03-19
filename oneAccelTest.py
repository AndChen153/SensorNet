#!/usr/bin/python
import smbus
import math
import time
 
# Register
power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c
 
def read_byte(reg):
    return bus.read_byte_data(address, reg)
 
def read_word(reg):
    h = bus.read_byte_data(address, reg)
    l = bus.read_byte_data(address, reg+1)
    value = (h << 8) + l
    return value
 
def read_word_2c(reg):
    val = read_word(reg)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

def dist(a,b):
    return math.sqrt((a*a)+(b*b))
 
def get_y_rotation(x,y,z):
    radians = math.atan2(x, dist(y,z))
    return -math.degrees(radians)
 
def get_x_rotation(x,y,z):
    radians = math.atan2(y, dist(x,z))
    return math.degrees(radians)

def find_num(label):
    for i in range (1,4):
    highest = 0
    for filename in listdir('./data{0}'.format(i)):
        highest += 1
    return highest
 
bus = smbus.SMBus(1) # bus = smbus.SMBus(0)
address = 0x68       # via i2cdetect

cycle = 0
while True:
    label = raw_input("activity label? (runs 15 times)")
    low = find_num(label)
    for i in range (low,low+15):
        start = raw_input("start?")
        print("/data{1}/data{0}.csv printed".format(i, label))
        f = open("./data{1}/data{0}.csv".format(cycle, label), "w")
        for i in range(1000):
            # Activate to be able to address the module
            bus.write_byte_data(address, power_mgmt_1, 0)
            
            acceleration_xout = read_word_2c(0x3b)
            acceleration_yout = read_word_2c(0x3d)
            acceleration_zout = read_word_2c(0x3f)
            
            acceleration_xout_scaled = acceleration_xout / 16384.0
            acceleration_yout_scaled = acceleration_yout / 16384.0
            acceleration_zout_scaled = acceleration_zout / 16384.0
            
            x = '{0},{1},{2},{3},{4} \n'.format(i,acceleration_xout, acceleration_yout, acceleration_zout, int(label))
            f.write(x)

            time.sleep(0.0001)

            
        f.close()
