import os
from icrawler.builtin import GoogleImageCrawler, BaiduImageCrawler, BingImageCrawler


def checked_dir_and_get_path(dir_name):
    dir_path = os.path.expanduser("crawl")
    result_dir = os.path.join(dir_path, dir_name)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    return result_dir

# keyword = "띄워%20쓰기"
keyword = "SUV"
save_path = checked_dir_and_get_path(keyword)

filters = dict(
    type="photo",
    size='large',
    # color='white',
    # license='commercial,modify',
    # date=((2000, 1, 1), (2019, 8, 27))
)

filters = dict(
    type="photo",
    size='large',
    # license='commercial,modify',
    date="pastyear")
bing_crawler = BingImageCrawler(downloader_threads=4,
                                storage={'root_dir': bing_path})
bing_crawler.crawl(keyword=keyword, filters=filters, offset=0, max_num=10)


# baidu_crawler = BaiduImageCrawler(storage={'root_dir': save_path})
# baidu_crawler.crawl(keyword=keyword, offset=0, max_num=100, min_size=(200,200), max_size=None)