import time

CUR_TIME = None


def tok(info=''):
    global CUR_TIME
    if not CUR_TIME:
        CUR_TIME = time.time()
        return
    t = time.time()
    delta = t-CUR_TIME
    print('{}{:e}s'.format(info, delta))
    CUR_TIME = t

