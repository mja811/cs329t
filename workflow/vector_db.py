import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
import datetime


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
            for key in [
                "post_id",
                "title",
                "flair",
                "created_utc",
                "url",
                "downs",
                "ups",
                "score",
            ]:
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


if __name__ == "__main__":
    vdb = VectorDB(json_doc_fpath="../data/aita_posts.json")
    
    query = """My boyfriend and I went to get pedicures together, something we rarely do and I thought would be a nice, 
    low-key couples activity. He finished before me, and I still had about 25–30 minutes left. Instead of waiting and 
    relaxing, he suddenly said it felt too hot inside and announced that he was going to walk home to "get some exercise," 
    since his doctor told him to move more. For context, it wasn't hot outside at all, it was around 70° and really pleasant. 
    He kept asking if I was okay with him leaving, which made it feel even stranger, like he was waiting for permission to 
    do something he already knew I'd find odd. I told him it was his choice, but I didn't really understand why he couldn't 
    just wait. He ended up walking home, which took about 23 minutes. The whole thing felt off, though, mostly because that 
    just so happened to line up exactly with the time his Discord group (which includes one particular female friend he always 
    seems eager to talk to) usually gets online. I just found it inconsiderate. We went together, it was supposed to be 
    something shared, and he couldn't stay 25 more minutes until I was done?"""

    results = vdb.query(query, k=5)

    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_fpath = f"/results/query_results_{timestamp}.txt"

    with open(output_fpath, "w", encoding="utf-8") as f:
        f.write("=== QUERY ===\n")
        f.write(query + "\n\n")
        f.write("=== RESULTS ===\n")

        for i, doc in enumerate(results, 1):
            title = doc.metadata.get("title", "N/A")
            score = doc.metadata.get("score", "N/A")
            post_id = doc.metadata.get("post_id", "N/A")
            url = doc.metadata.get("url", "N/A")

            f.write(f"\n--- Result {i} ---\n")
            f.write(f"Title: {title}\n")
            f.write(f"Score: {score}\n")
            f.write(f"Post ID: {post_id}\n")
            f.write(f"URL: {url}\n\n")
            import textwrap
            content = doc.page_content
            wrapped_content = textwrap.fill(content, width=80)
            f.write(wrapped_content)
            
            if len(doc.page_content) > 1000:
                f.write("\n\n[...content truncated...]")
            
            f.write("\n\n")
            f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"Query and results saved to {output_fpath}")