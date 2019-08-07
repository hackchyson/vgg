import execjs
import webbrowser

url = "www.baidu.com"
a = webbrowser.open(url)
print(a)

def test(data):
    print(data)


file = open("/home/hack/PycharmProjects/vgg/parse/index2.html")
line = file.readline()
htmlstr = ''
while line:
    htmlstr = htmlstr + line
    line = file.readline()

ctx = execjs.compile(htmlstr)
result = ctx.call('start')
print(result)
