import execjs  # pip install PyExecJS

import requests
import json
import execjs
import numpy as np
import math
import logging
import threading
from config.config import *
import itertools
from parse.pic_parser import Base64Parser

logging.basicConfig(level=LEVEL)


class Cap:
    """
    It is used for Json data format:
    {
    position:{x:1,y:2,z:3},
    target:{x:1,y:2,z:3},
    up:{x:0,y:0,z:1}
    }
    """

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
    """
    Given two 3-D points, the first is the "from" point, the second is the "to" point.
    Compute 24 direction around the "from" point.
    In the XOY plane, 8 directions divide the plane into 8 regions equally.
    In the XOZ or YOZ plane, the angle between z axis x or y axis is -pi/4, 0, and pi/4 respectively.
    :param point_from: Around this point, compute 24 directions.
    :param point_to: the distance between "from" and "to" points is used as the direction end point.
    :return: 24 points in 24 directions.
    """
    from_x = point_from[X]
    from_y = point_from[Y]
    from_z = point_from[Z] + HEIGHT
    to_x = point_to[X]
    to_y = point_to[Y]
    to_z = point_to[Z] + HEIGHT
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
    """
    Get data from web.
    :return: Json data needed.
    """
    response = requests.get(URL)
    text = response.text
    json_obj = json.loads(text)
    return json_obj[JSON_DATA][JSON_BOS_EDGES]


def filtered_points(bos_edges, sample_distance=SAMPLE_DISTANCE):
    """
    Among all the points, filter out points that the distance between them
    is greater than the sample_distance.
    :param bos_edges: Json data containing all the distances and points.
    :param sample_distance: the filter condition that the distance
    between two points must be greater than
    :return: All the points meeting the distance condition.
    """
    init = .0
    capture_points = []
    for edge in bos_edges:
        distance = float(edge[DISTANCE])
        if distance + init > sample_distance:
            capture_points.append(edge[TO])
            init = .0
        else:
            init = init + distance
    return capture_points


def flatten(filtered):
    """
    Flatten List[List] into List.
    :param filtered: The object needed to be flatten.
    :return: Flatten list.
    """
    caps = []
    for i in range(len(filtered) - 1):
        cap = all_directions(filtered[i], filtered[i + 1])
        for j in cap:
            caps.append(j)
    return caps


def get_batch_pics(points, js_path=JS_PATH, js_function=JS_FUNC, pic_dir=PIC_DIR):
    """
    Convert a list of json point data into pictures and store them.
    :param points: Point to capture a picture.
    :param pic_dir: Path to store pictures.
    :param js_function: JavaScript function name.
    :param js_path: javascript path.
    :return: None
    """
    file = open(js_path)
    line = file.readline()
    html_str = ''
    while line:
        html_str = html_str + line
        line = file.readline()
    logging.debug(html_str)
    ctx = execjs.compile(html_str)
    logging.debug(ctx)
    logging.debug(js_path)
    logging.debug(js_function)
    batch_list = ctx.call(js_function, points)
    # print(batch_list)
    # Base64Parser().parse_json_list(batch_list, pic_dir)


def get_pics(points, batch=PARSE_BATCH_SIZE):
    """
    Divide the point list into several list and run them in parallel.
    :param points: A list of all points.
    :param batch: A sub list of the points.
    :return: None
    """
    num = len(points) // batch
    for i in range(num - 1):
        threading.Thread(target=get_batch_pics(points[num: num + 1]))
    threading.Thread(target=get_batch_pics(points[num: len(points)]))


def save(list_json, json_file):
    with open(json_file, 'w') as file:
        file.write(list_json)


if __name__ == "__main__":
    bos_edges = get_data()
    filtered = filtered_points(bos_edges)
    logging.debug(len(filtered))
    caps = flatten(filtered)
    logging.debug((len(filtered) - 1) * 24)
    logging.debug(len(caps))
    logging.debug(caps[0])
    # get_pics(filtered)
    json_list = json.dumps(caps)
    save(json_list, 'caps.json')
