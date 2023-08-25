import fnmatch
import serial
import csv

def auto_detect_serial_unix(preferred_list=['*']):
    '''try to auto-detect serial ports on win32'''
    import glob
    glist = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    ret = []

    # try preferred ones first
    for d in glist:
        for preferred in preferred_list:
            if fnmatch.fnmatch(d, preferred):
                ret.append(d)
    if len(ret) > 0:
        return ret
    # now the rest
    for d in glist:
        ret.append(d)
    return ret

def main():
    available_ports = auto_detect_serial_unix()
    port = serial.Serial(available_ports[0], baudrate= 115200,timeout=0.1)
    ## baudrate is transfering ratio
    ## timeout is to set a read timeout value in seconds
    with open('period_value.csv', 'a+') as fff:
        while(True):
            rawdata = port.readline()
            data = rawdata.decode()
            rows = csv.writer(fff)
            rows.writerow(data)
            print(data)
            #print(data[0:3])
        return 0

if __name__ == '__main__':
    main()
