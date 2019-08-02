import requests
import json

url = "https://webapi.bos.xyz/models/170642112/routes/170652840/edges/point?devcode=e10e59bf0ee97213ca7104977877bd1a"

DISTANCE = "distance"
TO = 'to'
response = requests.get(url)
text = response.text
json_obj = json.loads(text)

bos_edges = json_obj['data']['bosEdges']
init = .0

capture_points = []
for edge in bos_edges:
    distance = float(edge[DISTANCE])
    if distance + init > 1000:
        capture_points.append(edge[TO])
        init = .0
    else:
        init = init + distance
print(capture_points)