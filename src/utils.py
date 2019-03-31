import logging.config
import numpy as np
import collections
import subprocess
import requests
import datetime
import psutil
import time
import json
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
    logfile = datetime.datetime.now().strftime(os.path.join(LOGS_DIR, '%Y-%m-%d_%H-%M__{}.log'.format(logname)))
    #logfile = datetime.datetime.now().strftime(os.path.join(LOGS_DIR, '{}.log'.format(logname)))
    
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
                'stream': 'ext://sys.stdout'
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


def read_json_data(path, verbose=False):
    """
    Reads json data from specified path
    
    Args:
        path: str, path to json file
        verbose: bool, if True, will print debug output
        
    Returns:
        dict or None depending on if file exists
    """
    data = None
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    elif verbose:
        print('{} does not exist'.format(path))
    return data

def dataframe_pctile_slice(df, col, pctile):
    return df[df[col] < df[col].quantile(pctile)]


def kill_tensorboard():
    cmd = ['pgrep', 'tensorboard']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()
    rc = p.returncode
    print('{} yielded -> {}'.format(cmd, output))
    if output:
        tb_ids = output.split(b'\n')
        for tb_id in tb_ids:
            if len(tb_id) > 0:
                subprocess.run(['kill', str(int(tb_id))])
                print('killed {}!'.format(tb_id))
    return


def elapsed_time(start):
    return (time.time() - start) / 60


def compare_dicts(a, b, root='root', tabs=0):
    """
    doesn't quite work as hoped
    """
    keys_in_a_not_b = set(a.keys()).difference(set(b.keys()))
    keys_in_b_not_a = set(b.keys()).difference(set(a.keys()))
    same = set(a.keys()).intersection(set(b.keys()))
    tabsp = '\t' * tabs
    tabsp1 = tabsp + '\t'
    for k in keys_in_a_not_b:
        print(tabsp + '[{}] In A, Not B:'.format(root))           
        print(tabsp1 + '{}: {}'.format(k, a[k]))
    for k in keys_in_b_not_a:
        print(tabsp + '[{}] In B, Not A:'.format(root))           
        print(tabsp1 + '{}: {}'.format(k, b[k]))
    for k in same:
        aval = a[k]
        bval = b[k]
        if aval == bval:
            print(tabsp + '[{}] Same:'.format(root))
            print(tabsp1 + '{}: {}'.format(k, aval))
        elif isinstance(aval, dict):
            # check assumption that if aval is dict then bval is dict
            assert isinstance(bval, dict)
            compare_dicts(aval, bval, root=k, tabs=tabs+1)
        else:
            print(tabsp + '[{}] Different:'.format(root))
            print(tabsp1 + '{}: {}'.format(k, aval))
    return


def flatten_dict(d, parent_key='', sep='/'):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def retrieve_ckpt_dir_from_legacy_sacred_cfg(root):
    """
    Just in case you need to retrieve a ckpt_dir from an old sacred config.json save
    
    Args:
        root: str, path to dir with sacred experiments
        
    Returns:
        cfg_path: str, path to sacred config
        ckpt_dir: str, path to dir housing model ckpts of config
    """
    cfg_path = None
    ckpt_dir = None
    found = False
    for exp_dir in os.listdir(root):
        if exp_dir == '_sources':
            continue
        #print('-------')
        #print(exp_dir)
        cfg_path = os.path.join(root, exp_dir + '/config.json')
        with open(cfg_path, 'r') as json_file:
            cfg = json.load(json_file)
        ckpt_dir = cfg.get('cfg', 'banana').get('model', 'apple').get('ckpt_dir', 'orange')
        #print(ckpt_dir)
        if '2019-03-12_16-36-25' in ckpt_dir:
            print('found it!')
            found = True
            break
    if found:
        return cfg_path, ckpt_dir

    
def memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    return memoryUse


def np_recon_loss(x, y):
    """
    Performs the same recon loss calc done in voxel_vae but with numpy
    instead of tensorflow

    Args:
        x: np.array, original input object
        y: np.array, reconstructed output object
        
    Returns:
        float, reconstruction loss as computed in voxel_vae
    """
    clipped_input = np.clip(x, 1e-7, 1.0 - 1e-7)
    clipped_output = np.clip(y, 1e-7, 1.0 - 1e-7)
    bce = -(98.0 * clipped_input * np.log(clipped_output) + 2.0 * (1.0 - clipped_input) * np.log(1.0 - clipped_output)) / 100.0
    # Voxel-Wise Reconstruction Loss 
    # Note that the output values are clipped to prevent the BCE from evaluating log(0).
    recon_loss = np.mean(bce, 1)
    mean_recon = np.mean(recon_loss)
    return mean_recon
            