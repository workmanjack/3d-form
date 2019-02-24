import requests
import datetime
import logging
import os


SRC_ROOT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.join(SRC_ROOT, '..')
LOGS_DIR = os.path.join(SRC_ROOT, 'logs')


def get_logger(logname='root', verbosity=1):
    """
    Good intro to logging in python:
    https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
    
    Inspiration for this function:
    https://stackoverflow.com/questions/7621897/python-logging-module-globally
    """
    
    logger = logging.getLogger(logname)

    level = logging.INFO
    if verbosity is not None and int(verbosity) > 0:
        level = logging.DEBUG

    logger.setLevel(logging.DEBUG)  # we adjust on console and file later
    # create file handler which logs even debug messages (and make sure logs dir exists)
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    fh = logging.FileHandler(datetime.datetime.now().strftime(os.path.join(LOGS_DIR, '%Y-%m-%d_%H-%M__{}.log'.format(logname))), 'w', 'utf-8')
    fh.setLevel(logging.DEBUG)  # always log everything to file
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)  # only log to console what the user wants
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def api_json(url):
    """
    Returns json data from the provided url

    Args:
        url: str

    Returns:
        json as dict or None
    """
    print('api_query: {}'.format(url))
    resp = requests.get(url=url)
    data = None
    if resp.status_code != 200:
        print('failed to retrieve data from {0}'.format(url))
        print('status_code={0}'.format(resp.status_code))
    else:
        data = resp.json()
    return data


def dataframe_pctile_slice(df, col, pctile):
    return df[df[col] < df[col].quantile(pctile)]
