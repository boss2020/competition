import  requests
import  json
import  time
import  openpyxl
#模拟浏览器发送请求并获取响应结果
#第一句：导入requests模块
#第二局：发送请求
def  get_comments(productId,page):
    url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={0}&score=0&sortType=5&page={1}&pageSize=10&isShadowSku=0&fold=1'.format(productId,page)
    #使用python字符串的格式,{0}对应着第一个字符串,{1}对应着第二个字符串
    #{0}对应第一个productId,{1}对应着第二个page内容
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}
    reap = requests.get(url,headers = headers)
    #数据还没完事，因为数据不是json格式，需要转为json格式
    #print(reap.text)    #响应结果显示输出
    s = reap.text.replace('fetchJSON_comment98(','')
    s = s.replace(');','')
    #将获取数据开头的fetchJSON_comment98(以及数据结尾处的);去除掉
    #将str类型的数据转为json格式的数据
    json_data = json.loads(s)
    return  json_data
#获取最大页数
def  get_maxpage(productId):
    dic_data = get_comments(productId,'0')
    #调用刚才写的函数，向服务器发送请求，获取字典数据
    return  dic_data['maxPage']
    #仔细观察属性，发现'maxPage'露在外面
#提取数据
def  get_info(productId):
    max_page = get_maxpage(productId)
    #调用函数获取商品的最大评论页数
    lst = []  #用于存储提取到的商品数据
    for  page  in  range(1,max_page+1):  #循环执行次数
        #获取每页的商品评论
        comments = get_comments(productId,page)
        comm_lst= comments['comments']
        #根据key获取value，根据comments获取评论的列表
        #遍历评论列表，分别获取每条评论中的内容，颜色，鞋码
        for  item  in  comm_lst:    #每条评论又分别是一个字典，再继续根据key获取值
            content = item['content']   #获取评论中的内容
            color = item['productColor']    #获取评论中的颜色
            size = item['productSize']      #获取评论中的鞋码
            lst.append([content,color,size])    #将每条评论的信息添加到列表当中
        time.sleep(3)   #延迟时间，防止程序执行速度太快，被封IP
    save(lst)   #调用自己编写的函数，将列表中的数据进行存储

#用于将爬取到的数据存储到excel之中
def  save(lst):   
    wk = openpyxl.Workbook()    #创建工作簿对象
    #一个.xlsx文件，称为一个工作簿，一个工作簿中有三个工作表
    sheet = wk.active   #获取活动表
    #遍历列表，将列表中的数据添加到工作表中，列表中的一条数据，在excel中是一行
    for  item  in  lst:
        sheet.append(item)
    #保存到磁盘上
    wk.save('销售数据.xlsx')
if __name__ == '__main__':
    productId = '70690115165'
    #print(get_maxpage(productId))
    get_info(productId)
    #速度比较慢，因为总共爬了10页，而且每一页隔3秒，大致需要一分钟`
