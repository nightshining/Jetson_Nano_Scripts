import time
import smbus2

class PCA9685:
    def __init__(self, address=0x40, bus=1):
        self.bus = smbus2.SMBus(bus)
        self.address = address
        self.MODE1 = 0x00
        self.PRESCALE = 0xFE
        self.LED0_ON_L = 0x06
        
        # Wake up
        self.bus.write_byte_data(self.address, self.MODE1, 0x00)
        # Set PWM frequency to 50Hz
        prescale = int(25000000.0 / (4096 * 50.0) - 1)
        self.bus.write_byte_data(self.address, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.address, self.MODE1, 0x80)
        time.sleep(0.005)

    def set_pwm(self, channel, on, off):
        base_addr = self.LED0_ON_L + 4 * channel
        self.bus.write_word_data(self.address, base_addr, on)
        self.bus.write_word_data(self.address, base_addr + 2, off)

    def set_angle(self, channel, angle):
        # Convert angle to PWM value
        pulse = int(150 + (angle * (600 - 150) / 180))
        self.set_pwm(channel, 0, pulse)

def test_servos():
    try:
        pwm = PCA9685()
        
        for channel in range(16):
            print("\nTesting Channel {}".format(channel))
            
            print("Moving to 0 degrees")
            pwm.set_angle(channel, 0)
            time.sleep(1)
            
            print("Moving to 90 degrees")
            pwm.set_angle(channel, 90)
            time.sleep(1)
            
            print("Moving to 180 degrees")
            pwm.set_angle(channel, 180)
            time.sleep(1)
            
            print("Returning to 90 degrees")
            pwm.set_angle(channel, 90)
            time.sleep(1)
    
    except Exception as e:
        print("Error: {}".format(e))

if __name__ == '__main__':
    test_servos()
