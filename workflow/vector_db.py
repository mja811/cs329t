import json
import os
from dotenv import load_dotenv

from config import OUTPUT_DIR, DATA_DIR

load_dotenv()

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
import datetime
import textwrap

METADATA_KEYS = [
                "post_id",
                "title",
                "flair",
                "created_utc",
                "url",
                "downs",
                "ups",
                "score",
            ]


class VectorDB:
    def __init__(self, json_doc_fpath, persist_dir="chroma_db") -> None:
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        if self._db_exists():
            print(f"Loading existing Chroma DB from '{self.persist_dir}'...")
            self.db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=OpenAIEmbeddings(),
            )
        else:
            print(f"Creating new Chroma DB at '{self.persist_dir}'...")
            self._format_documents(json_doc_fpath)
            self._embed()

    def _db_exists(self):
        # Check if the directory exists and contains chroma.sqlite3
        if not os.path.isdir(self.persist_dir):
            return False
        contents = os.listdir(self.persist_dir)
        return len(contents) > 0 and any('chroma.sqlite3' in f or 'sqlite3' in f for f in contents)

    def _format_documents(self, json_doc_fpath):
        documents = []
        df = pd.read_json(json_doc_fpath)
        
        # Drop duplicates if necessary
        df = df.drop_duplicates(subset=["post_id", "title", "url", "selftext"])
        
        for _, row in df.iterrows():
            selftext = row.get("selftext")
            
            # Skip if selftext is empty, null, or not a valid string
            if pd.isna(selftext) or not str(selftext).strip():
                continue
                
            # Create document with selftext as content
            chapter_doc = Document(page_content=str(selftext))
            
            # Add metadata
            for key in METADATA_KEYS:
                if key in row:
                    chapter_doc.metadata[key] = row[key]
                    
            documents.append(chapter_doc)
        
        print(f"Formatted {len(documents)} documents")
        self.documents = documents

    def _embed(self):
        if not self.documents:
            raise ValueError("No documents to embed")
            
        self.db = Chroma.from_documents(
            documents=self.documents,
            embedding=OpenAIEmbeddings(),
            persist_directory=self.persist_dir,
        )
        print("Chroma DB created and persisted successfully.")

    def query(self, query_text, k=1):
        """Query the vector database for similar documents"""
        docs = self.db.similarity_search(query_text, k=k)
        return docs


def create_post_json(doc):
    post_json = {}
    post_json["selftext"] = doc.page_content
    for k in METADATA_KEYS:
        post_json[k] = doc.metadata.get(k, "N/A")
    return post_json


def run_vectordb_node(post_json, log_dir, k=5):
    vdb = VectorDB(json_doc_fpath=DATA_DIR / "aita_posts.json")
    post_id = post_json["post_id"]
    query = post_json["selftext"]
    save_filepath = log_dir / f"vdb_{post_id}_{k}.jsonl"
    data = []

    if os.path.exists(save_filepath):
        print("Reading cached results from '{}'...".format(save_filepath))
        with open(save_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # skip empty lines
                    data.append(json.loads(line))
        return data

    results = vdb.query(query, k)

    output_fpath = log_dir / f"query_results_{post_id}.txt"
    with open(output_fpath, "w", encoding="utf-8") as f, open(save_filepath, "w", encoding="utf-8") as f2:
        f.write("=== QUERY ===\n")
        query = textwrap.fill(query, width=80)
        f.write(query + "\n\n")
        f.write("=== RESULTS ===\n")

        for i, doc in enumerate(results, 1):
            post_json = create_post_json(doc)
            data.append(post_json)
            f2.write(json.dumps(post_json) + "\n")

            title = doc.metadata.get("title", "N/A")
            score = doc.metadata.get("score", "N/A")
            post_id = doc.metadata.get("post_id", "N/A")
            url = doc.metadata.get("url", "N/A")

            f.write(f"\n--- Result {i} ---\n")
            f.write(f"Title: {title}\n")
            f.write(f"Score: {score}\n")
            f.write(f"Post ID: {post_id}\n")
            f.write(f"URL: {url}\n\n")

            content = doc.page_content
            wrapped_content = textwrap.fill(content, width=80)
            f.write(wrapped_content)

            if len(doc.page_content) > 1000:
                f.write("\n\n[...content truncated...]")

            f.write("\n\n")
            f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"Query and results saved to {output_fpath}")

    return data


if __name__ == "__main__":
    post_json = {
        "post_id": "abc",
        "title": "AITA for wanting to refuse to help my unemployed brother apply for college cause I am worried how it will affect me?",
        "selftext": "I am (20M) an international Actuarial Science student currently in my third year of study of a 7 year  program. Recently I received a call from my father telling me that he wants my 33 year old brother to join me at school doing the program I am doing as a First year and asking if it was possible for him to apply with his degree and results and if I could help with that.   Some backstory on my brother is that he did go to school for Business Management in my native country but has been struggling to get a job. He wanted to do a masters but that didn't go through, (don't what to bash my brother here but even with this degree he was on and off and would decide when he would go to school or not, this caused him to spend nearly twice as long the duration). After he finished he couldn't find a job and somewhere along the line he gave up I guess.   He has been home doing nothing for like 7 years now. He was being encouraged by my other relatives to find a job, maybe in a different field or least do something in the meantime. Multiple programmes, courses etc popped up during those years but he rejected them saying he wants to do something else.   Back to me now, So after my first year abroad I noticed how my bills and tuition fees where putting a strain on my parents when I would go back on holiday, this continued until one day I overheard them saying they had taken a collateral loan on their car in order to make the deadline for my tuitions. I asked what was going on and they tried to downplay saying that they would always do this when they paid my fees and they had nearly payed back the chain loans. It was always a stressful when due dates would I arrive and I had to call them over and over asking for money.   I felt bad for them and decided I needed to find a cheaper school as this situation was untenable. I spent close to year looking for a cheaper school in the same country till I found it. So I ended transferring from that better school to this one so that at least my parents can breathe and save up for their pension as they quite older. The situation became better but far from ideal, the major stress comes from things like rent and groceries as money is still scarce.   Now fast forward to today, NOW TELL ME WHY DO THEY SUDDENLY THINK THAT THEY CAN NOW DOUBLE THE AMOUNT THEY CAN SPEND BY SENDING MY BROTHER HERE. I thought I had done the right thing by proactively seeking cheaper schools and accommodation so that things are no longer so tight and stressful to the wire. Now if they go through with this idea it will be worse than before and I don't know if I can handle it.   I know I sound so heartless and selfish to my own blood but, I know if they go through with this my living standards will drastically worsen. I already receive no pocket money or anything like that, the only cash I have is from the small saving I have from transport costs when I decide to walk or wait for the evening buses.",
        "flair": "Not the A-hole",
        "created_utc": 1761061210.0,
        "url": "https://www.reddit.com/r/AmItheAsshole/comments/1ocgbt8/aita_for_wanting_to_refuse_to_help_my_unemployed/",
        "downs": 0,
        "ups": 39,
        "score": 39
    }
    res = run_vectordb_node(post_json, OUTPUT_DIR / "vectordb_results/")
    print(res)