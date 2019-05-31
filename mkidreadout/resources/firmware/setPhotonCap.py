
import argparse

from mkidreadout.channelizer.Roach2Controls import Roach2Controls

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Turn photon cap on/off')
    parser.add_argument('roaches', nargs='+', type=str, help='List of ROACH numbers (last 3 digits of IP)')
    parser.add_argument('-s', '--start', action='store_true', help='Start photon cap')
    parser.add_argument('-x', '--stop', action='store_true', help='Stop photon cap')
    parser.add_argument('-i', '--host-ip', type=str, default='10.0.0.52', help='IP of server')
    parser.add_argument('-p', '--port', type=int, default=50000, help='Destination port')
    clOptions = parser.parse_args()
    roachList = []
    specDictList = []
    plotSnaps = True
    startAtten = 40

    if clOptions.start == clOptions.stop:
        raise Exception('Must specify --start or --stop')

    for roachNum in clOptions.roaches:
        ip = '10.0.0.'+roachNum
        roach = Roach2Controls(ip)
        roach.connect()
        if clOptions.start:
            roach.startSendingPhotons(clOptions.host_ip, clOptions.port)
        else:
            roach.stopSendingPhotons()
