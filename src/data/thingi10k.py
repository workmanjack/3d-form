import urllib.request
import requests
import shutil
import json
import csv
import os


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


def thingi10k_api_data(obj_id):
    """
    Returns api data of specified thingi10k object

    Args:
        obj_id: int or str
    """
    url = 'https://ten-thousand-models.appspot.com/api/v1/model/{}'.format(obj_id)
    return api_json(url)


def thingi10k_image(obj_id, dest):
    """
    Returns an image file for the specified thingi10k object

    https://stackoverflow.com/questions/3042757/downloading-a-picture-via-urllib-and-python
    https://docs.python.org/3.0/library/urllib.request.html#urllib.request.urlretrieve

    Args:
        obj_id: int or str
        dest: str, path to save image file to

    Returns:
        str, path at which image file was saved
    """
    url = 'https://storage.googleapis.com/thingi10k/renderings/{}.png'.format(obj_id)
    print('img_query: {}'.format(url))
    filename, headers = urllib.request.urlretrieve(url)
    shutil.move(filename, dest)
    return dest


def make_thingi10k_index(data_dir, index_path):
    """
    Constructs a csv index of thingi10k objects
    """
    rows = list()
    header = ['file', 'name', 'num_vertices']

    mesh_dir = os.path.join(data_dir, 'external/Thingi10k/raw_meshes')
    raw_dir = os.path.join(data_dir, 'raw')
    proc_dir = os.path.join(data_dir, 'processed')

    fields = list()
    data = list()

    count = 0

    for path in os.listdir(mesh_dir):

        # quick check that we are reading something that we care about
        if not path.endswith('.stl'):
            print('not an stl file: {}'.format(path))
            continue

        # get file ids
        obj_id = os.path.splitext(path)[0]

        # get api data
        file_data = thingi10k_api_data(obj_id)
        stl_name = '{}.json'.format(obj_id)

        # gather object images
        img_name = '{}.png'.format(obj_id)
        dest = os.path.join(raw_dir, img_name)
        if os.path.exists(dest):
            print('{} already exists'.format(dest))
        else:
            dest = thingi10k_image(obj_id, dest)

        # write json
        file_data['stl_file'] = stl_name
        file_data['img_file'] = img_name
        dest = os.path.join(raw_dir, stl_name)
        with open(dest, 'w') as outfile:
            json.dump(file_data, outfile)

        # save index row
        fields = list(set(fields).union(set(file_data.keys())))
        data.append(file_data)
        count += 1

    # write index
    # index_path = '_output/thingi10k.csv'
    with open(index_path, 'w', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fields)

        writer.writeheader()
        for i, obj in enumerate(data):
            # https://stackoverflow.com/questions/1285911/how-do-i-check-that-multiple-keys-are-in-a-dict-in-a-single-passs
            # make sure that this obj has all the right fields
            if set(fields) <= set(obj):
                for f in fields:
                    if not obj.get(f, None):
                        obj[f] = None

            writer.writerow(obj)

    print('{} objects processed'.format(count))
    print('Index written to {}'.format(index_path))