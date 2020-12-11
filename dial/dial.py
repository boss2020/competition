import requests

url = 'http://10.3.8.211/login'
dial_url = 'http://10.3.8.211/dial'

data = {
  'user': '2018110737',
  'pass': 'kuaile9696',
  'line': 'CUC-BRAS'
}

if __name__ == "__main__":
    s = requests.Session()
    r_login = s.post(url, data=data)
    while True:
        r_dial = s.get(dial_url)
        if r_dial.json()['code']== 0:
            break
