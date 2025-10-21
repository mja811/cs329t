import requests
import json
import time
import os
from datetime import datetime

# Where to save data
OUTPUT_FILE = "aita_posts.json"

def fetch_posts(subreddit="AmItheAsshole", limit=100, after=None):
    """Fetch one page of new posts from Reddit."""
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    if after:
        url += f"&after={after}"

    response = requests.get(url, headers={"User-agent": "AITA Minimal Collector"})
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return [], None

    data = response.json().get("data", {})
    posts = data.get("children", [])
    after = data.get("after", None)

    return posts, after


def collect_posts(max_posts=500):
    """Collect posts and save them into a JSON file."""
    total = 0
    after = None

    # Load existing data if file already exists
    if os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            all_posts = json.load(f)
    else:
        all_posts = []

    while total < max_posts:
        posts, after = fetch_posts(after=after)
        if not posts:
            print("No more posts found or rate limited.")
            break

        for p in posts:
            d = p["data"]
            flair = d.get("link_flair_text")
            if flair is None:  # Skip posts without verdicts
                continue

            post_data = {
                "post_id": d["id"],
                "title": d.get("title", "").replace("\n", " ").strip(),
                "selftext": d.get("selftext", "").replace("\n", " ").strip(),
                "flair": flair,
                "created_utc": d.get("created_utc"),
                "url": f"https://www.reddit.com{d.get('permalink')}",
                "downs": d.get("downs"),
                "ups": d.get("ups"),
                "score": d.get("score"),
            }

            all_posts.append(post_data)
            total += 1

            if total >= max_posts:
                break

        print(f"Collected {total} posts so far...")

        if not after:
            break

        time.sleep(2)  # Avoid Reddit rate limiting

    # Save all posts to JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)

    print("Done! Posts saved to", OUTPUT_FILE)


if __name__ == "__main__":
    collect_posts(max_posts=1000)
