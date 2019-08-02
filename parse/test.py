import execjs


file = open("/home/hack/PycharmProjects/vgg/parse/hello.js")
line=file.readline()
htmlstr=''
while line:
    htmlstr = htmlstr+line
    line=file.readline()

ctx=execjs.compile(htmlstr)
result = ctx.call('helloWord',"bala")
print(result)