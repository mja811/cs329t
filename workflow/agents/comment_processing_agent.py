import json
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import pandas as pd

from config import PROCESSED_COMMENTS_DIR, DATA_DIR

# https://www.reddit.com/r/AITAH/comments/1ecylvt/what_do_each_of_the_acronyms_here_mean/
AITA_RESULT_ACRONYMS = {
    "YTA" : "You're the Asshole",
    # "YWBTA" : "You Would be the Asshole",
    "NTA" :	"Not the Asshole",
    # "YWNBTA" : "You Would Not be the Asshole",
    "ESH" : "Everyone Sucks here",
    "NAH" :	"No Assholes here",
    "INFO" : "Not Enough Info",  # remove in text search bc too vague I think for exact search
}
ALL_ACRONYMS_LIST_STR = "\n".join([k + ": " + v for k,v in AITA_RESULT_ACRONYMS.items()])


def get_csv_file_path(post_id):
    return os.path.join(PROCESSED_COMMENTS_DIR, f"{post_id}.csv")


def get_results(df):
    result = {}
    for acr in AITA_RESULT_ACRONYMS:
        comments = list(df[df["result"] == acr]["body"])
        if len(comments) > 0:
            result[acr] = comments
    return result


def create_comment_prompt(post_dict, post_comments_df):
    post_title = post_dict["title"]
    post_text = post_dict["selftext"]
    categories = "\n".join([k + ": " + v for k,v in AITA_RESULT_ACRONYMS.items()])

    prompt = f"""Your job as an agent is to classify the comments for an Am I The Asshole Post into categories, based on the comment text in relation to the post content. For each comment, classify the comment as one of the following categories:
{categories}

Here is a simplified example:
START EXAMPLE

Post Title: AITA for eating in class?
Post Description: I get really hungry during my afternoon classes. I need to snack to pay attention. However, I have been told that you cannot eat in class. AITA if I take out a snack?

Comments table:
|    | post_id   | comment_id   | author               | body                                                                                                                                                                                                                                                                                                                             |   score |   created_utc |   ups |   downs |
|---:|:----------|:-------------|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|--------------:|------:|--------:|
|  0 | 1odtjvy   | nkwil5o      | OrdinaryMajestic4686 | NTA. You need to eat to pay attention to class.                                                                                                                  |      41 |   1.76119e+09 |    41 |       0 |
|  1 | 1odtjvy   | nkwithi      | Buho45               | Is this a loud snack? Is it disruptive?                                                                                                                          |       3 |   1.76119e+09 |     3 |       0 |
|  2 | 1odtjvy   | nkyg2mn      | myeyesarelistening   | This is disrespectful, you should make sure you eat before class so that you have full attention and are not distracted by food.                                 |       2 |   1.76123e+09 |     2 |       0 |
|  3 | 1odtjvy   | nkzrl7b      | Katerh               | You need to follow the rules, sometimes it is a building or health hazard and may have severe consequences even though it doesn't seem like a big deal.          |       3 |   1.76124e+09 |     3 |       0 |

Result: [NTA, INFO, YTA, YTA]

END OF EXAMPLE
---

First, read the following Am I The Asshole post:
Post Title: {post_title}
Post Text: {post_text}

Here is the comments table which contains all comments in the "body" column:
{post_comments_df.to_markdown()}

For each row in the comments table, choose the single best matching acronym (from {list(AITA_RESULT_ACRONYMS.keys())}).
Return a list of the acronyms in the same order as they are presented in the table.
Double check that there is a result for each row, so that the final results list should be the same length as the table rows.
Respond **only** with the list of acronyms, no extra text.
"""
    return HumanMessage(content=prompt)

def process_single_post_comments(post_dict, log_dir=None, use_saved_results=True, agent_str="gpt-4o-mini"):
    post_id = post_dict["post_id"]

    # Check if post has already been processed. If so, load saved results.
    result_fp = get_csv_file_path(post_id)
    if use_saved_results and os.path.exists(result_fp):
        df = pd.read_csv(result_fp)
        result_df = df
    else:
        df = pd.read_csv(DATA_DIR / "aita_comments.csv")
        post_comments_df = df[df["post_id"] == post_dict["post_id"]]
        if post_comments_df.empty:
            return {}
        post_comments_df.reset_index(drop=True, inplace=True)
        # post_comments_df.index = post_comments_df.index + 1

        prompt = create_comment_prompt(post_dict, post_comments_df)
        llm = ChatOpenAI(model=agent_str)
        llm_reply = llm.invoke([prompt])
        print(llm_reply)

        cleaned_results = []
        try:
            content_str = (
                llm_reply.content
                if isinstance(llm_reply.content, str)
                else str(llm_reply.content)
            )
            parsed = content_str.split("[")[-1].split("]")[0]
            results = parsed.split(",")
            for r in results:
                rstrip = r.strip()
                for a in AITA_RESULT_ACRONYMS.keys():
                    if a in rstrip:
                        cleaned_results.append(a)
        except Exception as exc:
            raise ValueError(
                f"Invalid response:\n{llm_reply.content}"
            ) from exc

        assert len(cleaned_results) == len(post_comments_df), f"Response length mismatch: {len(cleaned_results)} vs. {len(post_comments_df)}; {cleaned_results}"
        post_comments_df["result"] = cleaned_results
        post_comments_df.to_csv(result_fp, index=False)
        result_df = post_comments_df

    results = get_results(result_df)
    results.update(post_dict)

    if log_dir:
        with open(log_dir / f"{post_id}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


def run_comment_processing_node(posts_list, log_dir=None, use_saved_results=True, agent_str="gpt-4o-mini"):
    sorted_comments_data = []
    for post in posts_list:
        try:
            sorted_comments_data.append(process_single_post_comments(post, log_dir, use_saved_results, agent_str))
        except Exception as e:
            print(e)
    return sorted_comments_data


if __name__ == '__main__':
    for post in [
        {
            "post_id": "1odtjvy",
            "title": "AITA for not really caring all that much that my mom sent my aunt an angry text on my behalf?",
            "selftext": "Hi, On Father's Day of this year,  my aunt and uncle had my dad [70] and I [30M] over for food and such. During this gathering, I gave my dad a $25 gift card to Lowe's for Father's Day. I was self conscious about this gift as it wasn't much money and I could conceivably afford more, but I had already spent a lot of money that weekend on gifts for a couple other people and this was what I decided was right for me at the time. And frankly, he wasn't a very good father, so he should have been happy he got anything. There isn't a card in the store I could buy him and mean it. Too much has happened. Taylor Tomlinson has a good bit that conveys my exact feelings.  Soon after I gave it to him, he starts making fun of it, saying that he spent like $80 on the food he brought. My aunt chimed in condescendingly, \"what can you buy with $25, [my name]?\" I sat there and took it, trying to let it go. The rest of the gathering passed without incident.   As I drove home, I got more and more angry. This was a PERFECT example of why I don't like my father. It exemplified so many things. I was staying with my mom at the time, and when I got home she could tell I was upset. I eventually told her what happened, including that my aunt had chimed in. I broke down and texted my dad and called him a fucking asshole. My mom has seen my struggles with my father and she was even more mad than I was. She eventually said she was going to text my aunt and give her a piece of her mind. I knew this would only make things worse, but 30 years of rage was boiling over in me and I was too pissed off to care. She texted my aunt \"You have highly disappointed me that you belittled [my name] with the gift he gave his father.\" She said my aunt would do the same thing if my mom had insulted one of her kids. My adult sister agreed. I also tend to agree.  The following afternoon I got a text from my aunt saying that she was sorry and she thought we were just joking around. I accepted her apology and told her she and I were fine but my father and I have ceased getting along.   Soon after I sent my father a letter addressing what my therapist and I consider to be years of psychological abuse from him, telling him that my relationship with him will not move forward until he reads it.   Things have been pretty cold with my aunt and uncle since all this happened. I saw them for the second time since Father's Day at a family party (my dad wasn't there) and my uncle called me out for going and crying to my mommy about what his wife said. I retorted that I didn't ask my mom to do that, but I don't really blame her for doing it. He said it was an overreaction and that we had all been drinking and that I was out of line. AITA?",
            "flair": "Not the A-hole",
            "created_utc": 1761192213.0,
            "url": "https://www.reddit.com/r/AmItheAsshole/comments/1odtjvy/aita_for_not_really_caring_all_that_much_that_my/",
            "downs": 0,
            "ups": 38,
            "score": 38
        },
        {
            "post_id": "1odtj1x",
            "title": "AITA for not asking my BIL first?",
            "selftext": "I don’t have kids so I wanted to gauge how I turn out in this story. I have a nephew that is 2yo and 2 days ago was the first time I was asked by my sister to babysit. (I had been asking to babysit since he was born). I was super excited and moved my work schedule so I could do it. Note: when anyone is asked to babysit, her Husband is always home. He works at home and they have someone babysit so he can focus on working.  The babysitting was for a couple hours in the morning, everything seemed to go well. I had checked with my sister beforehand to ask if I could take my nephew to the park a half block away. Her husband said to just let him know when I was taking him. I did. I was also asked to babysit yesterday, again everything went fine. Fast forward to today, I was asked to babysit again. (My BIL was home) I texted him to let him know I was taking my nephew for a walk.   So this is where AITA comes in. He called me a couple minutes later asking where I was at, clearly angry. He said they really prefer if people ask permission before taking their son anywhere because “you just freak out if your kid is gone”. I said I could definitely bring him back no problem. He said no it’s fine, I’m just saying. (Annoyed/angry the whole time) When I got back he didn’t come upstairs through the whole time even up to when I left their house. I also told my sister about the situation and she said its better if I phrase things as a question.. Again I don’t think I’m in the wrong because I did text beforehand and the park is literally just a half block away in an enclosed neighborhood. But I also don’t have kids so I don’t know if it’s normal? AITA?",
            "flair": "Not the A-hole",
            "created_utc": 1761192132.0,
            "url": "https://www.reddit.com/r/AmItheAsshole/comments/1odtj1x/aita_for_not_asking_my_bil_first/",
            "downs": 0,
            "ups": 11,
            "score": 11
        },
        {
            "post_id": "1odsh8m",
            "title": "AITA for not wearing pants when guests are over?",
            "selftext": "I want to start by saying this isn't a serious argument between me (32F) and my boyfriend (33M). It's something we were laughing about, but something he also genuinely believes, and he pushed me to make the post because he thinks he's right and that people will agree with him. Lol.   Right now, I'm wearing what you could say are \"booty shorts\" and a t-shirt. My dad's friend is over to do some work on my house. My boyfriend asks why I am wearing shorts and not pants, that we're clearly different people because he would wear pants if he had guests over. I said, what I'm wearing is normal, this is normal. He disagrees. So, AITA for not wearing pants when other people visit?",
            "flair": "No A-holes here",
            "created_utc": 1761188795.0,
            "url": "https://www.reddit.com/r/AmItheAsshole/comments/1odsh8m/aita_for_not_wearing_pants_when_guests_are_over/",
            "downs": 0,
            "ups": 0,
            "score": 0
        },
        {
            "post_id": "1odsch2",
            "title": "AITA because I don't text a lot ???",
            "selftext": "So my friend of 10 years has stopped talking to me at the moment and I need to know if it's mainly my fault. She's always been abit awkward with me- for example got annoyed when I wouldn't lend her money for dr*gs etc.   Anyway-I've suffered with bad depression and struggle to leave the house the last year since my dad died- something I'm working on gradually though to get to my old self!   My texting isn't great at all as I don't focus on being on my phone I try to occupy myself with other things so the last few months I haven't been great at texting.  My friend and I haven't spoken loads the last few months but I'll message her every few days and say hi how are you, just to make sure she's okay.  A few weeks ago I sent her a few messages trying to make conversation and she ignored me. A few days later I sent her a photo of my newborn niece to which she ignored. Then yesterday I messaged her to say hey is everything okay and she responded saying that she doesn't see the point in messaging me anymore because all I do is ask how she is and then she won't hear from me till the day after/2 days later.  Personally I don't see an issue with talking every few days but I want to understand where she's coming from also. She told me I don't leave the house to see her or try to talk to her properly so what's the point in trying.  Personally I think this is inconsiderate as she knows I've been struggling- but then am I not seeing things from her point of view ? I told her I don't see an issue with texting every few days to see how she is but she told me that it's draining trying to explain why it annoys her- so what now🙃",
            "flair": "No A-holes here",
            "created_utc": 1761188396.0,
            "url": "https://www.reddit.com/r/AmItheAsshole/comments/1odsch2/aita_because_i_dont_text_a_lot/",
            "downs": 0,
            "ups": 2,
            "score": 2
        }
        ]:
        try:
            result = run_comment_processing_node(post)
            print(result)
        except Exception as e:
            print(e)
