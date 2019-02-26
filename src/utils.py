import requests
import datetime
import logging.config
import os


SRC_ROOT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.join(SRC_ROOT, '..')
LOGS_DIR = os.path.join(SRC_ROOT, 'logs')

        
def get_logger(logname='root', verbosity=1):
    """
    Good intro to logging in python:
    https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
    
    Python Logging Config Dict Format:
    https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
    
    How to use dictConfig:
    https://stackoverflow.com/questions/7507825/where-is-a-complete-example-of-logging-config-dictconfigs
    """
    #logfile = datetime.datetime.now().strftime(os.path.join(LOGS_DIR, '%Y-%m-%d_%H-%M__{}.log'.format(logname)))
    logfile = datetime.datetime.now().strftime(os.path.join(LOGS_DIR, '{}.log'.format(logname)))
    
    DEFAULT_LOGGING = {
        'version': 1,
        "disable_existing_loggers": False,
        'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s - %(levelname)s - %(module)s: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                #'formatter': 'detailed',
                #'level': 'INFO',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': logfile,
                'mode': 'w',
                'formatter': 'detailed',
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        },
    }

    logging.config.dictConfig(DEFAULT_LOGGING)
    logging.info('Logging to {}'.format(logfile))
    return


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


def read_json_data(path):
    """
    Reads json data from specified path
    
    Args:
        path: str, path to json file
        
    Returns:
        dict or None depending on if file exists
    """
    data = None
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    return data

def dataframe_pctile_slice(df, col, pctile):
    return df[df[col] < df[col].quantile(pctile)]
