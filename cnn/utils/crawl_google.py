from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib3
import time
import base64


searchterm = '트럭'
#url = "https://www.google.co.in/search?q=" + searchterm + "&source=lnms&tbm=isch"
url ="https://google.com/search?q=Truck&sxsrf=ALeKk03QI-7gDtC0ko70XC6VAhLZqZmTBQ:1628238693611&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjWp8Tn_ZvyAhWKfXAKHQVUDd8Q_AUoAXoECAEQAw"

result_dir = 'crawl'
result_dir = os.path.expanduser(result_dir)
result_dir = os.path.join(result_dir, searchterm)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

browser = webdriver.Chrome("chromedriver.exe")
browser.get(url)
header = {
    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36"
    }

counter = 0
succounter = 0

for _ in range(1000):
    browser.execute_script("window.scrollBy(0,10000)")
    isShowBottom = browser.find_element_by_xpath('//*[@id="yDmH0d"]/div[2]').is_displayed()  # 마지막 바닥 요소
    isShowMoreButton = browser.find_element_by_class_name("mye4qd").is_displayed() # 버튼 요소

    if isShowMoreButton:
        time.sleep(1)
        browser.find_element_by_class_name('mye4qd').click()

    if isShowBottom and not isShowMoreButton:
        # 마지막 바닥과 버튼 요소가 보이지 않으므로 검색이 끝난것으로 판단하고 종료
        break

retries = urllib3.Retry(connect=5, read=2, redirect=5)
http = urllib3.PoolManager(retries= retries)

for x in browser.find_elements_by_xpath("//div[@class='islrc']/div[@class='isv-r PNCib MSM1fd BUooTd']\
                                        /a[@class='wXeWr islib nfEiy mM5pbd']"):
    counter = counter + 1
    print("Total Count:", counter)
    print("Succsessful Count:", succounter)

    try:
        x.click()
        browser.implicitly_wait(1)

        xpath = '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img'
        img_src = browser.find_element_by_xpath(xpath).get_attribute("src")

        try:
            if img_src[:4] == "data":
                data_type, base64_str = img_src.split(",")
                img_type = data_type.split(";")[0].split("/")[-1]

                img_data = base64.b64decode(base64_str)
            else:
                req = http.request('GET', img_src, timeout=3)  # , headers={'User-Agent': header})
                img_data = req.data
        except:
            print("Nope", img_src)

        save_path = os.path.join(result_dir, searchterm + "_" + str(counter) + "." + img_type)
        with open(save_path, "wb") as f:
            f.write(img_data)

        succounter += 1
    except Exception as e:
        print("can't get img")
        print(e)

print(succounter, "pictures succesfully downloaded")
browser.close()