import requests
import os


PROJECT_ROOT = os.path.join(os.path.realpath(os.path.dirname(__file__)), '..')

             
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
