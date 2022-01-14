from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib3
import time
import base64


searchterm = 'SUV'
url = "http://auto.danawa.com/search/?q=" + searchterm

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

retries = urllib3.Retry(connect=5, read=2, redirect=5)
http = urllib3.PoolManager(retries= retries)

for index in range(len(browser.find_elements_by_class_name("newcarPage"))):
    print("Total Count:", counter)
    print("Succsessful Count:", succounter)

    for x in browser.find_elements_by_xpath("//a[@class='image']/img"):
        counter = counter + 1
        try:
            img_src = x.get_attribute("src")

            try:
                req = http.request('GET', img_src, timeout=3)  # , headers={'User-Agent': header})
                img_type = img_src.split("/")[-1].split(".")[-1]
                img_data = req.data
            except:
                print("Nope", img_src)
                continue

            save_path = os.path.join(result_dir, searchterm + "_" + str(succounter) + "." + img_type)
            with open(save_path, "wb") as f:
                f.write(img_data)

            succounter += 1
        except Exception as e:
            print("can't get img")
            print(e)

    next_page = browser.find_elements_by_xpath('//*[@id="autodanawa_gridC"]/div/div[3]/div[2]/div/div/a')[index]
    next_page.click()

print(succounter, "pictures succesfully downloaded")
browser.close()

