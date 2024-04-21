from typing import Any, Dict

import scrapy
from scrapy import Spider
from scrapy.crawler import CrawlerProcess

from src.utils import BASE_PATH


def custom_settings_helper(name: str) -> Dict[str, Any]:
    path = BASE_PATH / "data/pipeline/01_scrape" / f"{name}.csv"
    return {
        "DOWNLOAD_DELAY": 0.35,
        "ROBOTSTXT_OBEY": False,
        "FEEDS": {path: {"format": "csv", "overwrite": True}},
    }


class BlogSpider(Spider):
    name = "blog"
    custom_settings = custom_settings_helper("blog")
    url = "https://discuss.ardupilot.org"

    def start_requests(self):
        yield scrapy.Request(
            url=f"{self.url}/categories_and_latest", callback=self.parse_categories
        )

    def parse_categories(self, response):
        for category in response.json()["category_list"]["categories"]:
            topic_url = (
                lambda cid: f"{self.url}/c/{category['slug']}/{cid}/l/latest.json?ascending=false&page=0"
            )
            topic_request = lambda cid: scrapy.Request(
                topic_url(cid), self.parse_topics
            )
            yield from map(topic_request, category["subcategory_ids"])
            if not category["has_children"]:
                yield topic_request(category["id"])

    def parse_topics(self, response):
        topic_list = response.json()["topic_list"]
        post_request = lambda t: scrapy.Request(
            f"{self.url}/t/{t['id']}/0.json", self.parse
        )
        yield from map(post_request, topic_list["topics"])
        if "more_topics_url" in topic_list:
            base_topic_url = " ".join(response.url.split("&")[:-1])
            page_number = int(response.url.split("=")[-1])
            yield scrapy.Request(
                f"{base_topic_url}&page={page_number + 1}", self.parse_topics
            )

    def parse(self, response):
        posts = response.json()["post_stream"]["posts"]
        for post in posts:
            yield {
                "topic_id": post["topic_id"],
                "post_id": post["id"],
                "post_number": post["post_number"],
                "reply_to_post_number": post["reply_to_post_number"],
                "content": post["cooked"],
            }

        if posts[-1]["post_number"] < response.json()["highest_post_number"]:
            url = "/".join(response.url.split("/")[:-1])
            next_post = f"{url}/{posts[-1]['post_number'] + 6}.json"
            print(next_post)
            yield scrapy.Request(next_post)


class DiscordSpider(Spider):
    name = "discord"
    custom_settings = custom_settings_helper("discord")
    url = "https://discord.com/api/v9"
    headers = {
        "authorization": "XXX"
    }
    guild_id = "674039678562861068"
    limit = 100

    def start_requests(self):
        url = f"{self.url}/guilds/{self.guild_id}/channels"
        yield scrapy.Request(url=url, callback=self.parse_guild, headers=self.headers)

    def parse_guild(self, request):
        for channel_id in map(lambda channel: channel["id"], request.json()):
            url = f"{self.url}/channels/{channel_id}/messages?limit={self.limit}"
            yield scrapy.Request(url=url, callback=self.parse, headers=self.headers)

    def parse(self, request):
        channel_id = request.url.split("/")[-2]
        messages = request.json()

        for message in messages:
            yield {
                "channel_id": channel_id,
                "message_id": message["id"],
                "content": message["content"],
            }

        if len(messages) == self.limit:
            before_url = f"{self.url}/channels/{channel_id}/messages?limit={self.limit}&before={messages[-1]['id']}"
            yield scrapy.Request(
                url=before_url, callback=self.parse, headers=self.headers
            )


class DocumentationSpider(Spider):
    name = "docs"
    custom_settings = custom_settings_helper("docs")

    def start_requests(self):
        url = "https://ardupilot.org/ardupilot/"
        yield scrapy.Request(url=url, callback=self.parse_documentations)
        yield scrapy.Request(url=url, callback=self.parse)

    def parse_documentations(self, response):
        for href in response.xpath('//li[@class="toctree-l1"]/a/@href').extract():
            if href.startswith("https://ardupilot.org"):
                yield scrapy.Request(url=href, callback=self.parse)

    def parse(self, response):
        yield {
            "url": response.url,
            "content": response.xpath('//div[@role="main"]').get(),
        }

        next_page = response.css("a.btn.btn-neutral.float-right::attr(href)").get()
        if next_page is not None:
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse)


def main():
    process = CrawlerProcess()
    process.crawl(DocumentationSpider)
    process.crawl(DiscordSpider)
    process.crawl(BlogSpider)
    process.start()


if __name__ == "__main__":
    main()
