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

    return os.path.join(out)


def parse(line):
    # #"Log File for cm_virt_panel	 created on 2020/08/05 10:00:08"
    # #"Timestamp"	"Low side PSI"	"High side PSI"	"In water temp F"	"Out water temp F"	"He gas temp F"	"Oil temp F"	"Motor Amps"
    # 2020/08/05 10:00:08	093.3	273.0	040.3	073.0	146.3	077.9	22.0
    data = line.split()
    sampletime = data[0] + ' ' + data[1]
    sampletime = datetime.strptime(sampletime.decode('utf-8'), '%Y/%m/%d %H:%M:%S')
    current = float(data[-1])
    return sampletime, current


if __name__ == '__main__':
    try:
        logging.basicConfig()
        log = logging.getLogger('Compressor Monitor')

        logfile = find_log(LOG_DIR)

        try:
            fp = open(logfile, 'r')
        except IOError:
            log.critical('Unable to open logfile {}'.format(logfile))
            notify.notify(MAIL_RECIPIENTS, 'Log Missing', sender='MEC Compressor', sms=False, email=True,
                          holdoff_min=EMAIL_HOLDOFF_MIN)
            sys.exit(1)

        while True:
            try:
                lastline = fp.readlines()[-1]
                sampletime, current = parse(lastline)
                log.info('{} ')
                if (datetime.now() - sampletime).total_seconds() > STALE_TIME_SECONDS:
                    log.info('State timestamp, checking for new logfile')
                    newlog = find_log(LOG_DIR)
                    if newlog == logfile:
                        log.info('No new logfile, notifying.')
                        notify.notify(MAIL_RECIPIENTS, 'Log Stale', sender='MEC Compressor', sms=False, email=True,
                                      holdoff_min=EMAIL_HOLDOFF_MIN)
                    else:
                        log.debug('Switching to new log: {}'.format(newlog))
                        fp.close()
                        logfile = newlog
                        fp = open(logfile, 'r')

                if current < MOTOR_CURRENT_ALARM_THRESHOLD:
                    notify.notify(MAIL_RECIPIENTS, 'Current Fault: {} A'.format(current),
                                  subject='MEC Compressor Error', sender='MEC Compressor', sms=True, email=True,
                                  holdoff_min=EMAIL_HOLDOFF_MIN)
            except ValueError:
                log.debug('Failed to parse "{}" from log: {}'.format(logfile))
                notify.notify(MAIL_RECIPIENTS, 'Log Corrupt', sender='MEC Compressor', sms=False, email=True,
                              holdoff_min=EMAIL_HOLDOFF_MIN)
            except IOError:
                log.debug('Failed to read from log: {}'.format(logfile))
                notify.notify(MAIL_RECIPIENTS, 'Log Read Error', sender='MEC Compressor', sms=False, email=True,
                              holdoff_min=EMAIL_HOLDOFF_MIN)
            time.sleep(MONITOR_INTERVAL_SEC)

    except Exception as e:
        log.critical('Compressor monitor died: {}\n'.format(e), exc_info=True)
