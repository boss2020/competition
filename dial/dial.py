import requests

url = 'http://10.3.8.211/login'
dial_url = 'http://10.3.8.211/dial'

data = {
  'user': '',
  'pass': '',
  'line': 'CUC-BRAS'
}

if __name__ == "__main__":
    s = requests.Session()
    r_login = s.post(url, data=data)
    while True:
        r_dial = s.get(dial_url)
        if r_dial.json()['code']== 0:
            break
#进入北邮登录界面模拟登录，登录成功之后对应的
#'code'的相应属性为0
