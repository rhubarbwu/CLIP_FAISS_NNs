import urllib.request, json


def build_vocabulary(url):
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())

    keys = set()

    def unpack_keys(data, main=None):
        for (key, value) in data.items():
            keys.add(key.replace(" ", "_"))
            unpack_keys(value, main=key)

    unpack_keys(data)
    return sorted(keys)
