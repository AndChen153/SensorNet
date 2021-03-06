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

def read_byte2(reg):
    return bus2.read_byte_data(address2, reg)
 
def read_word2(reg):
    h = bus2.read_byte_data(address2, reg)
    l = bus2.read_byte_data(address2, reg+1)
    value = (h << 8) + l
    return value
 
def read_word_2c2(reg):
    val = read_word2(reg)
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
 
bus = smbus.SMBus(1) # bus = smbus.SMBus(0)
address = 0x68       # via i2cdetect
bus2 = smbus.SMBus(1) # bus = smbus.SMBus(0)
address2 = 0x69       # via i2cdetect


while True:
    # Activate to be able to address the module
    bus.write_byte_data(address, power_mgmt_1, 0)

    print "Gyro"
    print "--------"
    
    gryo_xout = read_word_2c(0x43)
    gryo_yout = read_word_2c(0x45)
    gryo_zout = read_word_2c(0x47)
    
    print "gryo_xout: ", ("%5d" % gryo_xout), " scaled: ", (gryo_xout / 131)
    print "gryo_yout: ", ("%5d" % gryo_yout), " scaled: ", (gryo_yout / 131)
    print "gryo_zout: ", ("%5d" % gryo_zout), " scaled: ", (gryo_zout / 131)
    
    print
    print "Accelerometer"
    print "---------------------"
    
    acceleration_xout = read_word_2c(0x3b)
    acceleration_yout = read_word_2c(0x3d)
    acceleration_zout = read_word_2c(0x3f)
    
    acceleration_xout_scaled = acceleration_xout / 16384.0
    acceleration_yout_scaled = acceleration_yout / 16384.0
    acceleration_zout_scaled = acceleration_zout / 16384.0
    
    print "acceleration_xout: ", ("%6d" % acceleration_xout), " Scaled: ", acceleration_xout_scaled
    print "acceleration_yout: ", ("%6d" % acceleration_yout), " Scaled: ", acceleration_yout_scaled
    print "acceleration_zout: ", ("%6d" % acceleration_zout), " Scaled: ", acceleration_zout_scaled
    
    print "X Rotation: " , get_x_rotation(acceleration_xout_scaled, acceleration_yout_scaled, acceleration_zout_scaled)
    print "Y Rotation: " , get_y_rotation(acceleration_xout_scaled, acceleration_yout_scaled, acceleration_zout_scaled)



    bus2.write_byte_data(address2, power_mgmt_1, 0)

    print "Gyro"
    print "--------"
    
    gryo_xout = read_word_2c2(0x43)
    gryo_yout = read_word_2c2(0x45)
    gryo_zout = read_word_2c2(0x47)
    
    print "gryo_xout2: ", ("%5d" % gryo_xout), " scaled: ", (gryo_xout / 131)
    print "gryo_yout2: ", ("%5d" % gryo_yout), " scaled: ", (gryo_yout / 131)
    print "gryo_zout2: ", ("%5d" % gryo_zout), " scaled: ", (gryo_zout / 131)
    
    print
    print "Accelerometer"
    print "---------------------"
    
    acceleration_xout = read_word_2c2(0x3b)
    acceleration_yout = read_word_2c2(0x3d)
    acceleration_zout = read_word_2c2(0x3f)
    
    acceleration_xout_scaled = acceleration_xout / 16384.0
    acceleration_yout_scaled = acceleration_yout / 16384.0
    acceleration_zout_scaled = acceleration_zout / 16384.0
    
    print "acceleration_xout2: ", ("%6d" % acceleration_xout), " Scaled: ", acceleration_xout_scaled
    print "acceleration_yout2: ", ("%6d" % acceleration_yout), " Scaled: ", acceleration_yout_scaled
    print "acceleration_zout2: ", ("%6d" % acceleration_zout), " Scaled: ", acceleration_zout_scaled
    
    print "X Rotation2: " , get_x_rotation(acceleration_xout_scaled, acceleration_yout_scaled, acceleration_zout_scaled)
    print "Y Rotation2: " , get_y_rotation(acceleration_xout_scaled, acceleration_yout_scaled, acceleration_zout_scaled)

    time.sleep(5)
