import execjs  # pip install PyExecJS

import requests
import json
import execjs
import numpy as np
import math
import logging
import itertools

logging.basicConfig(level=logging.INFO)

url = "https://webapi.bos.xyz/models/170642112/routes/170652840/edges/point?devcode=e10e59bf0ee97213ca7104977877bd1a"

DISTANCE = "distance"
TO = 'to'
X = 'x'
Y = 'y'
Z = 'z'


class Cap:
    def __init__(self, position, target, up):
        self.position = position
        self.target = target
        self.up = up


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def all_directions(point_from, point_to):
    from_x = point_from[X]
    from_y = point_from[Y]
    from_z = point_from[Z]
    to_x = point_to[X]
    to_y = point_to[Y]
    to_z = point_to[Z]
    dis = np.linalg.norm(np.array([to_x, to_y, to_z]) - np.array([from_x, from_y, from_z]))
    xys = []
    zs = [int(to_z - dis), int(to_z), int(to_z + dis)]
    for i in range(8):
        theta = i * math.pi / 4
        x = int((to_x - from_x) * math.cos(theta) - (to_y - from_y) * math.sin(theta) + from_x)
        y = int((to_y - from_y) * math.cos(theta) + (to_x - from_x) * math.sin(theta) + from_y)
        xys.append((x, y))
    xyzs = []
    for i in xys:
        for j in zs:
            xyzs.append(Cap(Point(int(from_x), int(from_y), int(from_z)).__dict__, Point(i[0], i[1], j).__dict__,
                            Point(0, 0, 1).__dict__).__dict__)
    return xyzs


def get_data():
    response = requests.get(url)
    text = response.text
    json_obj = json.loads(text)
    bos_edges = json_obj['data']['bosEdges']
    return bos_edges


def filtered_points(bos_edges):
    init = .0
    capture_points = []
    for edge in bos_edges:
        distance = float(edge[DISTANCE])
        if distance + init > 1000:
            capture_points.append(edge[TO])
            init = .0
        else:
            init = init + distance
    return capture_points


if __name__ == "__main__":
    bos_edges = get_data()
    filtered = filtered_points(bos_edges)
    logging.info(len(filtered))
    caps = []
    for i in range(len(filtered) - 1):
        cap = all_directions(filtered[i], filtered[i + 1])
        for j in cap:
            caps.append(j)
    logging.info((len(filtered) - 1) * 24)
    logging.info(len(caps))
    logging.info(caps[0])
