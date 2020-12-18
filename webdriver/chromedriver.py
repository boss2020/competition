from  selenium  import  webdriver
chrome_driver='C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'
driver = webdriver.Chrome(executable_path = chrome_driver)
driver.get("http://www.baidu.com")
