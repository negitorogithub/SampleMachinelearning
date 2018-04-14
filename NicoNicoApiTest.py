import requests

infoRequest = requests.get("http://ext.nicovideo.jp/api/getthumbinfo/sm3033822")
print(infoRequest)