import time, logging, os, sys
from mkidcore import notify
from datetime import datetime

MAIL_RECIPIENTS = ['ben', 'jeb']
LOG_DIR = r'C:\data\compressor_log'

MOTOR_CURRENT_ALARM_THRESHOLD = 1
STALE_TIME_SECONDS = 60

EMAIL_HOLDOFF_MIN = 10
MONITOR_INTERVAL_SEC = 15


def find_log(path):
    timestamp = None
    out = None
    for p, _, files in os.walk(path):
        try:
            date = datetime.strptime(os.path.basename(p), '%Y%m%d')
        except ValueError:
            continue
        if not timestamp or date > timestamp:
            if not files:
                continue
            timestamp = date
            out = [p, files]

    try:
        out[1] = sorted([f for f in out[1] if 'CPTLog' in f])[-1]
    except (TypeError, IndexError):
        log.debug('Unable to find log in {}'.format(path))
        return ''
    ret = os.path.join(*out)
    log.info('Found log "{}"'.format(ret))
    return ret


def parse(line):
    # #"Log File for cm_virt_panel	 created on 2020/08/05 10:00:08"
    # #"Timestamp"	"Low side PSI"	"High side PSI"	"In water temp F"	"Out water temp F"	"He gas temp F"	"Oil temp F"	"Motor Amps"
    # 2020/08/05 10:00:08	093.3	273.0	040.3	073.0	146.3	077.9	22.0
    data = line.split()
    sampletime = data[0] + ' ' + data[1]
    sampletime = datetime.strptime(sampletime.decode('utf-8'), '%Y/%m/%d %H:%M:%S')
    current = float(data[-1])
    log.debug("{}: {} A ('{}')".format(sampletime,current,line))
    return sampletime, current


sender = 'MEC@physics.ucsb.edu'

if __name__ == '__main__':
    logging.basicConfig()
    log = logging.getLogger('Compressor Monitor')
    log.setLevel('DEBUG')
    logging.getLogger('mkidcore').setLevel('DEBUG')

    notify.notify(MAIL_RECIPIENTS, 'Compressor Monitor Starting', sender=sender, sms=False, email=True)
    try:

        logfile = find_log(LOG_DIR)

        try:
            fp = open(logfile, 'r')
        except IOError:
            log.critical('Unable to open logfile {}'.format(logfile))
            notify.notify(MAIL_RECIPIENTS, 'Log Missing. Exiting.', sender=sender, sms=False, email=True)
            sys.exit(1)

        while True:
            try:
                lastline = fp.readlines()[-1]
                sampletime, current = parse(lastline)
                if (datetime.now() - sampletime).total_seconds() > STALE_TIME_SECONDS:
                    log.info('Stale timestamp, checking for new logfile')
                    newlog = find_log(LOG_DIR)
                    if newlog == logfile:
                        log.info('No new logfile, notifying.')
                        notify.notify(MAIL_RECIPIENTS, 'Compressor log stale. Most recent entry {}'.format(sampletime), 
                                      sender=sender, sms=False, email=True, holdoff_min=EMAIL_HOLDOFF_MIN)
                    else:
                        log.info('Switching to new log: {}'.format(newlog))
                        fp.close()
                        logfile = newlog
                        fp = open(logfile, 'r')
                elif current < MOTOR_CURRENT_ALARM_THRESHOLD:
                    log.critical('Current Fault: {}'.format(current))
                    notify.notify(MAIL_RECIPIENTS, 'Compressor current of {:.1f} A at {}'.format(current, sampletime),
                                  subject='MEC Compressor Error', sender=sender, sms=True, email=True,
                                  holdoff_min=EMAIL_HOLDOFF_MIN)
            except ValueError:
                log.warning('Failed to parse "{}" from log: {}'.format(logfile))
                notify.notify(MAIL_RECIPIENTS, 'Compressor Log Corrupt', sender=sender, sms=False, email=True,
                              holdoff_min=EMAIL_HOLDOFF_MIN)
            except IOError:
                log.error('Failed to read from log: {}'.format(logfile))
                notify.notify(MAIL_RECIPIENTS, 'Compressor Log Read Error', sender=sender, sms=False, email=True,
                              holdoff_min=EMAIL_HOLDOFF_MIN)
            time.sleep(MONITOR_INTERVAL_SEC)

    except Exception as e:
        log.critical('Compressor monitor died: {}\n'.format(e), exc_info=True)
        notify.notify(MAIL_RECIPIENTS, 'Compressor monitor exiting due to exception.', sender=sender, sms=False, email=True)
    