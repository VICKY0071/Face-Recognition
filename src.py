from urllib import request

a = request.urlopen("https://www.google.com")

print(a.read())