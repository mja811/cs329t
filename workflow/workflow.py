from config import RUN_LOGS
from vector_db import *
from agents.comment_processing_agent import run_comment_processing_node


def run_workflow(run_name, post_query_json):
    log_dir = RUN_LOGS / run_name
    comments_dir = log_dir / "comments/"
    os.makedirs(comments_dir, exist_ok=True)

    vdb_results = run_vectordb_node(post_query_json, log_dir)

    sorted_comments_data = []
    for post in vdb_results:
        try:
            sorted_comments_data.append(run_comment_processing_node(post, comments_dir))
        except Exception as e:
            print(e)

    print(sorted_comments_data)

if __name__ == '__main__':
    run_name = "test"
    post_query_json = {
        "post_id": "1ocgbt8",
        "title": "AITA for wanting to refuse to help my unemployed brother apply for college cause I am worried how it will affect me?",
        "selftext": "I am (20M) an international Actuarial Science student currently in my third year of study of a 7 year  program. Recently I received a call from my father telling me that he wants my 33 year old brother to join me at school doing the program I am doing as a First year and asking if it was possible for him to apply with his degree and results and if I could help with that.   Some backstory on my brother is that he did go to school for Business Management in my native country but has been struggling to get a job. He wanted to do a masters but that didn't go through, (don't what to bash my brother here but even with this degree he was on and off and would decide when he would go to school or not, this caused him to spend nearly twice as long the duration). After he finished he couldn't find a job and somewhere along the line he gave up I guess.   He has been home doing nothing for like 7 years now. He was being encouraged by my other relatives to find a job, maybe in a different field or least do something in the meantime. Multiple programmes, courses etc popped up during those years but he rejected them saying he wants to do something else.   Back to me now, So after my first year abroad I noticed how my bills and tuition fees where putting a strain on my parents when I would go back on holiday, this continued until one day I overheard them saying they had taken a collateral loan on their car in order to make the deadline for my tuitions. I asked what was going on and they tried to downplay saying that they would always do this when they paid my fees and they had nearly payed back the chain loans. It was always a stressful when due dates would I arrive and I had to call them over and over asking for money.   I felt bad for them and decided I needed to find a cheaper school as this situation was untenable. I spent close to year looking for a cheaper school in the same country till I found it. So I ended transferring from that better school to this one so that at least my parents can breathe and save up for their pension as they quite older. The situation became better but far from ideal, the major stress comes from things like rent and groceries as money is still scarce.   Now fast forward to today, NOW TELL ME WHY DO THEY SUDDENLY THINK THAT THEY CAN NOW DOUBLE THE AMOUNT THEY CAN SPEND BY SENDING MY BROTHER HERE. I thought I had done the right thing by proactively seeking cheaper schools and accommodation so that things are no longer so tight and stressful to the wire. Now if they go through with this idea it will be worse than before and I don't know if I can handle it.   I know I sound so heartless and selfish to my own blood but, I know if they go through with this my living standards will drastically worsen. I already receive no pocket money or anything like that, the only cash I have is from the small saving I have from transport costs when I decide to walk or wait for the evening buses.",
        "flair": "Not the A-hole",
        "created_utc": 1761061210.0,
        "url": "https://www.reddit.com/r/AmItheAsshole/comments/1ocgbt8/aita_for_wanting_to_refuse_to_help_my_unemployed/",
        "downs": 0,
        "ups": 39,
        "score": 39
    }
    run_workflow(run_name, post_query_json)
