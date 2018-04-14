from bs4 import BeautifulSoup
import requests
import time
import string
import re
from itertools import chain


# 文字コードの場所一覧
# アルファベット小文字→(97, 123)
# アルファベット大文字→(65, 91)
# 半角数字→(48, 58)
# ひらがな→(12353, 12436)
# カタカナ→(12449, 12532+1) ((+1の場所にはヴが入ってる))
# 全角数字→(65296, 65306)
# 平仮名と大文字アルファベットすべてと"その他の文字"のリストを返す
def all_song_tag_list() -> []:
    all_hiragana_list = [chr(i) for i in range(12353, 12436)]
    all_alphabet_list = list(string.ascii_uppercase)
    other_tag_list = ["その他の文字"]
    return all_hiragana_list + all_alphabet_list + other_tag_list


# Wikiの"タグキーワード「 」を含むページ一覧"のページを渡す
def find_song_urls_in_wiki(soup: BeautifulSoup) -> []:
    cmd_tags = soup.find(class_="cmd_tag")
    li_tags = cmd_tags.find_all("li")
    song_urls_in_wiki = []
    for li_tag in li_tags:
        if (".html" and "//www5.atwiki.jp/hmiku/pages/") in str(li_tag):
            song_urls_in_wiki.append(li_tag.a.get("href"))
    return song_urls_in_wiki


# Wikiの"タグキーワード「 」を含むページ一覧"のページを渡す
def find_tag_page_urls_in_wiki_fast(soup: BeautifulSoup) -> []:
    cmd_tag = soup.find(class_="cmd_tag")
    url_tags = cmd_tag.find_all("a", text=re.compile("^[0-9]|^[0-9][0-9]|^[0-9][0-9][0-9]"))
    song_urls_in_wiki = [url_tag.get("href") for url_tag in url_tags]
    song_urls_in_wiki.append(song_urls_in_wiki[0])
    song_urls_in_wiki = sorted(set(song_urls_in_wiki), key=song_urls_in_wiki.index)
    return song_urls_in_wiki


# 平仮名かアルファベット(小文字、大文字両方可)か"その他の文字"
def find_tag_page_urls_by_character(character: str) -> []:
    tag_page_urls = []
    for page_index in range(0, 10000):

        resp = requests.get("https://www5.atwiki.jp/hmiku/tag/" + character, {"p": str(page_index)})
        soup = BeautifulSoup(resp.text, "lxml")
        is_song_url_exist = False
        if find_song_urls_in_wiki(soup):
            tag_page_urls.append(resp.url)
            print(resp.url)
            is_song_url_exist = True
        if is_song_url_exist is False:
            break
    tag_page_urls = list(set(tag_page_urls))
    time.sleep(0.1)
    return tag_page_urls


def find_song_page_urls_all() -> []:
    all_song_urls = []
    for page_index in range(0, 10):

        resp = requests.get("https://www5.atwiki.jp/hmiku/tag/%E6%9B%B2", {"p": str(page_index)})
        soup = BeautifulSoup(resp.text, "lxml")

        page_song_urls = find_song_urls_in_wiki(soup)
        if page_song_urls:
            all_song_urls.extend(page_song_urls)
        else:
            break
    time.sleep(0.1)
    return all_song_urls


class Song:
    def __init__(self):
        self.title = ""
        self.nikonikoLinks = []
        self.nikonikoCount = -1
        self.nikonikoThumbnailLink = ""
        self.youtubeLinks = []
        self.youtubeCount = -1
        self.youtubeThumbnailLink = ""
        self.authors = []

    # 曲説明のページを渡す
    def apply_info_by_page(self, soup: BeautifulSoup) -> None:
        self.title = soup.find("title").text.replace(" - 初音ミク Wiki - アットウィキ", "")

        youtube_tag = (soup.find(class_="test"))
        if youtube_tag:
            self.youtubeLinks.append(youtube_tag.get("href"))
        else:
            pass

        # "iframe"はニコニコ埋め込みのタグ
        iframe_tags = soup.find_all("iframe")
        for iframe_tag in iframe_tags:
            nikoniko_link_tag = iframe_tag.find("a")
            if nikoniko_link_tag:
                self.nikonikoLinks.append(nikoniko_link_tag.get("href"))
            else:
                pass


print(find_song_page_urls_all())


"""
print(find_tag_page_urls_in_wiki_fast(
    BeautifulSoup(requests.get("https://www5.atwiki.jp/hmiku/tag/%E3%81%8B").text, "lxml")))

print(all_song_tag_list())

allCharacterList2Search = "あ"
allTagsPageUrls = [find_tag_page_urls_by_character(character) for character in allCharacterList2Search]
allTagsPageUrls = list(chain.from_iterable(allTagsPageUrls))
print(allTagsPageUrls)
# allMikuWikiLinks = [find_song_url_in_wiki(BeautifulSoup(resp, "lxml")) for resp in ]


mikuWikiTagSoups = [BeautifulSoup(requests.get(tagPageUrl).text, "lxml") for tagPageUrl in allTagsPageUrls]
mikuWikiUrls = [find_song_urls_in_wiki(mikuWikiTagSoup) for mikuWikiTagSoup in mikuWikiTagSoups]
mikuWikiUrls = list(chain.from_iterable(mikuWikiUrls))

songList = []
for mikuWikiUrl in mikuWikiUrls:
    song = Song()
    time.sleep(0.1)
    response = requests.get("https:"+mikuWikiUrl)
    beautifulSoup = BeautifulSoup(response.text, "lxml")
    song.apply_info_by_page(beautifulSoup)
    songList.append(song)
    for nikoniko_link in song.nikonikoLinks:
        print(song.title + "  のニコニコURLは  " + nikoniko_link)
    for youtube_link in song.youtubeLinks:
        print(song.title + "  のYoutubeURLは  " + youtube_link)
"""