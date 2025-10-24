import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
import datetime
import textwrap


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

    query="""Me (21F) and my friends, who I’ll call “Muriel” (22F) and “Lisbon” (21F), took an Uber to go out this weekend.  As soon as we got in the car, our Uber driver (late 30s) launched into a tirade about how his life is basically falling apart. For half an hour, he used us as a captive audience to rant about his marital and career failures. He bragged that he was making $200K in tech previously, but got laid off, and now his wife wants to leave him. Then he starting kicking off about his wife and “how could she abandon him for no reason in his time of need after 15 years of marriage” and saying she’s going to take their kids. He said he’s only going to be an Uber driver for a short time to network and get back into tech, and because of “interesting conversations like these.”  The conversation turned to us girls, and he started interrogating us about our majors and how old we are. He said he studied computer science back in the day. Muriel told him she’s majoring in history, and he remarked “ok, so just put the fries in the bag.” I said I’m majoring in women, gender, and sexuality. He interrupts me saying “another worthless major.” So that was the end of civility.  I said there’s no such thing as a worthless major, as the point is to learn critical thinking, and my major is actually one of the most important ones, since the oppression of women is one of society’s greatest problems that must be solved. And I said “maybe if you had taken a course in women, gender, and sexuality in college, your wife wouldn’t be leaving you.”  He said I don’t know anything about the real world, and I’ll never get a job outside of fast food. I told him the fact that I already got a six-figure return offer from a fast-growing tech startup, which I secured no problem through networking with my sorority alums. Muriel interjected that she also has a return offer with a top investment bank to which our college is a feeder.   I commented that it seems like the only person who doesn’t know anything about the world is him, and maybe he’d have a decent job still if he didn’t pick a useless major like CS.  We reached the event venue at this point, and he told us to “get the hell out of his car.” I sarcastically wished him luck with his divorce.  Lisbon, who’s not confrontational and was quiet most of the ride, said Muriel and I were “so aggro for no reason” and embarrassed her because we can’t take a joke. And that now her Uber rating will go down. In my opinion, she’s a bit of a STEM supremacist as well and likely sympathized with him as a fellow CS major. I think she’s also somewhat bitter that she never got a return offer from her summer internship in tech, so she’s perfectly fine with her humanities friends being disrespected to feel better about herself.  The whole exchange lowkey ruined our night out."""
    
    results = vdb.query(query, k=5)

    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_fpath = f"results/query_results_{timestamp}.txt"

    with open(output_fpath, "w", encoding="utf-8") as f:
        f.write("=== QUERY ===\n")
        query = textwrap.fill(query, width=80)
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
            
            content = doc.page_content
            wrapped_content = textwrap.fill(content, width=80)
            f.write(wrapped_content)
            
            if len(doc.page_content) > 1000:
                f.write("\n\n[...content truncated...]")
            
            f.write("\n\n")
            f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"Query and results saved to {output_fpath}")