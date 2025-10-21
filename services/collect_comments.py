import csv
import requests
import time
import os

INPUT_FILE = "aita_posts.csv"
OUTPUT_FILE = "aita_comments.csv"

FIELDS = ["post_id", "comment_id", "author", "body", "score", "created_utc", "ups", "downs"]

# Ensure output CSV has headers
if not os.path.isfile(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()


def fetch_comments(post_id):
    """Fetch top-level comments for a given post ID."""
    url = f"https://www.reddit.com/r/AmItheAsshole/comments/{post_id}.json"
    headers = {"User-Agent": "AITA-CommentCollector/1.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error {response.status_code} on {post_id}")
        return []

    try:
        data = response.json()
        if len(data) < 2:
            return []

        comments_data = data[1]["data"]["children"]
        comments = []

        for c in comments_data:
            if c["kind"] != "t1":
                continue

            d = c["data"]
            author = d.get("author")
            # Skip moderators, bots, or removed/deleted comments
            if not author or author in ["Judgement_Bot_AITA", "AutoModerator", "[deleted]"]:
                continue

            body = d.get("body", "").strip()
            if not body:
                continue

            comments.append({
                "post_id": post_id,
                "comment_id": d.get("id"),
                "author": author,
                "body": body.replace("\n", " "),
                "score": d.get("score"),
                "created_utc": d.get("created_utc"),
                "ups": d.get("ups"),
                "downs": d.get("downs"),
            })

        return comments

    except Exception as e:
        print(f"Failed to parse comments for {post_id}: {e}")
        return []


def collect_all_comments():
    """Iterate over posts in the CSV and collect comments."""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        posts = list(reader)

    print(f"Found {len(posts)} posts to collect comments for.\n")

    total_comments = 0

    for i, post in enumerate(posts, 1):
        post_id = post["post_id"]
        print(f"[{i}/{len(posts)}] Fetching comments for post {post_id}...")

        comments = fetch_comments(post_id)

        if comments:
            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writerows(comments)

            total_comments += len(comments)
            print(f"  Saved {len(comments)} comments ({total_comments} total)")
        else:
            print("  No valid comments found")

        # Sleep to respect Reddit rate limits
        time.sleep(2)

    print("\nDone! Comments saved to", OUTPUT_FILE)


if __name__ == "__main__":
    collect_all_comments()
