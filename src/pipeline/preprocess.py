from itertools import chain

import pandas as pd
import spacy
from scrapy import Selector
from tqdm import tqdm

from src.utils import load_data, save_data


def load_and_preprocess_scraping_data(source, unique_column):
    df = load_data(f"pipeline/01_scrape/{source}.csv")
    df.rename(columns={unique_column: "id"}, inplace=True)
    df = df[["id", "content"]]
    df.dropna(inplace=True)
    df["source"] = source
    return df


def extract_content(df):
    def extract_content_helper(row):
        selector = Selector(text=row["content"])
        p = selector.xpath("//p//text()").getall()
        li = selector.xpath("//li//text()").getall()
        return list(filter(lambda item: len(item) > 1, chain(p, li)))

    df["content"] = df.progress_apply(extract_content_helper, axis=1)
    df = df.explode(column="content")
    df.dropna(inplace=True)
    return df


def combine_to_source(*dfs):
    source = pd.concat(dfs)
    source["id"] = source.groupby("id").grouper.group_info[0]
    source.sort_values("id", inplace=True)
    source.reset_index(drop=True, inplace=True)
    return source


def extract_sentences(df):
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser", "lemmatizer"])
    nlp.add_pipe("sentencizer")

    def preprocess_sentence_helper(content):
        nlp_pipe = nlp.pipe(content, batch_size=50, n_process=6)
        return [list(doc.sents) for doc in tqdm(nlp_pipe, total=len(content))]

    df["content"] = df["content"].str.encode("ascii", "ignore").str.decode("ascii")
    df["content"] = df["content"].str.replace(r"\s+", " ", regex=True)
    df["content"] = preprocess_sentence_helper(df["content"])
    df = df.explode(column="content")
    df.rename(columns={"content": "sentence"}, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    return df


def filter_sentences_grammar(df):
    def grammar_check(sent):
        has_verb = any(map(lambda t: t.pos_ == "VERB", sent))
        has_noun = any(map(lambda t: t.pos_ == "NOUN", sent))
        has_propn = any(map(lambda t: t.pos_ == "PROPN", sent))
        return has_verb and (has_noun or has_propn) and len(sent) > 5

    df = df[df["sentence"].apply(grammar_check)].reset_index(drop=True)
    df["sentence"] = df["sentence"].apply(lambda sent: sent.text)
    return df


def drop_duplicate_mentions(df):
    df.drop_duplicates(subset=["sentence", "id"], inplace=True)
    df.drop(columns=["id"], inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def drop(df, pattern):
    return df[~df["sentence"].str.contains(pattern, regex=True)].reset_index(drop=True)


def replace(df, pattern, replace_string):
    df["sentence"] = df["sentence"].str.replace(pattern, replace_string, regex=True)
    return df


def strip(df, pattern):
    df["sentence"] = df["sentence"].str.strip(pattern)
    return df


def lstrip(df, pattern):
    df["sentence"] = df["sentence"].str.lstrip(pattern)
    return df


def rstrip(df, pattern):
    df["sentence"] = df["sentence"].str.rstrip(pattern)
    return df


def filter_and_drop_regex(source):
    # restore information, handle @
    # after this cell there are not @ in the source

    df = replace(
        source,
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "url",
    )
    df = replace(df, r"<@!?\d{18}>", "user")
    df = replace(df, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email")
    df = replace(df, r"@([^\s+])", r"\1")
    df = replace(df, "@", "at")
    df = drop(df, r" [a-z0-9]{40} ")

    # remove fragments for {}, [], ()

    df = replace(df, r"\[.+=.+\]", "")
    df = replace(
        df,
        r"\[/code|\[/?quote\]?|/?quote\]|\[?[ou]l\]|\[?li\]|\[/[ou]l\]?|\[/li\]?|\[/",
        "",
    )
    df = replace(df, r"systemd\[1\]: ", "")
    df = replace(df, r"[()]", "")
    df = drop(df, r"[\{\}\[\]]")

    # Filter log rows or programming code handle : , ;
    # search for file 'C:\
    df = drop(df, r"Test: |eg: |INFO | DEBUG |WARN |ERROR ")

    df = replace(df, r"\.\.+", " ")
    df = replace(df, r"\.?!!+", "!")
    df = replace(df, r"\.?\?\?+", "?")
    df = replace(df, r" \?", "?")
    df = replace(df, r" \!", "!")
    df = replace(df, r",,+", ",")
    df = replace(df, r"\-\-+", " ")
    df = replace(df, r" : ", ": ")
    df = replace(df, r" , ", ", ")
    df = replace(df, r" ; ", "; ")
    df = replace(df, r"[^\w]\'|\'[^\w]", " ")
    df = strip(df, r"[,;:\'\-]")

    df = replace(df, r" #+ ", " ")
    df = drop(df, r"#")

    df = drop(df, r"\$")
    # # % -> valid character for percentage
    df = replace(df, " % ", " ")

    df = strip(df, r"\^")
    df = replace(df, r"\^\^+", "")
    df = drop(df, r"\^")

    df = replace(df, r" & ", " and ")
    df = drop(df, r"&")

    df = strip(df, r"\*")
    df = replace(df, r"\*\*\*+", "")
    df = drop(df, r"\*")

    df = drop(df, r"\\")

    df = replace(df, r"[^\/]*\/\/", "")
    df = replace(df, r"and\/or", "and")
    df = replace(df, r" \/ ", "or")
    df = replace(
        df, r"(gps|GPS)\/(compass|Compass)|(compass|Compass)\/(gps|GPS)", "gps compass"
    )
    df = replace(df, r" w\/ ", "with")
    df = replace(df, r" w\/o ", "without")
    df = drop(df, r"[^ ]+\/[^ ]*\/[^ ]+| \/[^ ] | [^ ]\/ | \/ ")

    df = replace(df, r"`", "")
    df = replace(df, r'"', "'")

    df = drop(df, r"[<>]")
    df = replace(df, r"~", "")
    df = drop(df, r"=")
    df = drop(df, r"\|")

    df = lstrip(df, r"[%,:;_\/\-+\' ?!.0-9]")
    df = rstrip(df, r"[%,:;_\/\-+\.\' ]")
    df = replace(df, r"\s+", " ")

    df = df[~(df.sentence.str.count(r"[*@#$%^&*~\-_<>\{\}\[\]`]") > 3)].reset_index(
        drop=True
    )
    df = df[df.sentence.str.split().apply(len) > 4].reset_index(drop=True)
    df["sentence"] = df.progress_apply(
        lambda row: row["sentence"]
        if row["sentence"][-1] in ".?!"
        else f"{row['sentence']}.",
        axis=1,
    )
    df["sentence"] = df["sentence"].str.capitalize()

    return df


def main():
    tqdm.pandas()

    docs = load_and_preprocess_scraping_data("docs", "url")
    blog = load_and_preprocess_scraping_data("blog", "topic_id")
    discord = load_and_preprocess_scraping_data("discord", "channel_id")

    blog = extract_content(blog)
    docs = extract_content(docs)

    source = combine_to_source(blog, docs, discord)
    source = extract_sentences(source)
    source = filter_sentences_grammar(source)
    source = drop_duplicate_mentions(source)
    source = filter_and_drop_regex(source)

    source = source.reset_index().rename(columns={"index": "id"})
    save_data(source, "pipeline/02_preprocess/source.csv")


if __name__ == "__main__":
    main()
