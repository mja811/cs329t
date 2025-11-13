from config import RUN_LOGS, log_to_file
from vector_db import *
from agents.comment_processing_agent import run_comment_processing_node
from agents.advice_agent import run_advice_node
from agents.debate_agent import run_debate_agent_node


def run_workflow(run_name, post_query_json):
    log_dir = RUN_LOGS / run_name
    comments_dir = log_dir / "comments/"
    os.makedirs(comments_dir, exist_ok=True)
    debate_dir = log_dir / "debate/"
    os.makedirs(debate_dir, exist_ok=True)

    vdb_results = run_vectordb_node(post_query_json, log_dir)
    sorted_comments_data = run_comment_processing_node(vdb_results, comments_dir)
    # sorted_comments_data = []
    mod_summary, verdict, pct = run_debate_agent_node(sorted_comments_data, debate_dir, post_query_json)
    advice = run_advice_node(sorted_comments_data, verdict, mod_summary)
    print(advice)

    fp = debate_dir / f"debate_transcript_{post_query_json['post_id']}.txt"
    log_to_file(fp, advice)

if __name__ == '__main__':
    run_name = "test_full"
    # post_query_json = {
    #     "post_id": "1ocgbt8_2",
    #     "title": "AITA for wanting to refuse to help my unemployed brother apply for college cause I am worried how it will affect me?",
    #     "selftext": "I am (20M) an international Actuarial Science student currently in my third year of study of a 7 year  program. Recently I received a call from my father telling me that he wants my 33 year old brother to join me at school doing the program I am doing as a First year and asking if it was possible for him to apply with his degree and results and if I could help with that.   Some backstory on my brother is that he did go to school for Business Management in my native country but has been struggling to get a job. He wanted to do a masters but that didn't go through, (don't what to bash my brother here but even with this degree he was on and off and would decide when he would go to school or not, this caused him to spend nearly twice as long the duration). After he finished he couldn't find a job and somewhere along the line he gave up I guess.   He has been home doing nothing for like 7 years now. He was being encouraged by my other relatives to find a job, maybe in a different field or least do something in the meantime. Multiple programmes, courses etc popped up during those years but he rejected them saying he wants to do something else.   Back to me now, So after my first year abroad I noticed how my bills and tuition fees where putting a strain on my parents when I would go back on holiday, this continued until one day I overheard them saying they had taken a collateral loan on their car in order to make the deadline for my tuitions. I asked what was going on and they tried to downplay saying that they would always do this when they paid my fees and they had nearly payed back the chain loans. It was always a stressful when due dates would I arrive and I had to call them over and over asking for money.   I felt bad for them and decided I needed to find a cheaper school as this situation was untenable. I spent close to year looking for a cheaper school in the same country till I found it. So I ended transferring from that better school to this one so that at least my parents can breathe and save up for their pension as they quite older. The situation became better but far from ideal, the major stress comes from things like rent and groceries as money is still scarce.   Now fast forward to today, NOW TELL ME WHY DO THEY SUDDENLY THINK THAT THEY CAN NOW DOUBLE THE AMOUNT THEY CAN SPEND BY SENDING MY BROTHER HERE. I thought I had done the right thing by proactively seeking cheaper schools and accommodation so that things are no longer so tight and stressful to the wire. Now if they go through with this idea it will be worse than before and I don't know if I can handle it.   I know I sound so heartless and selfish to my own blood but, I know if they go through with this my living standards will drastically worsen. I already receive no pocket money or anything like that, the only cash I have is from the small saving I have from transport costs when I decide to walk or wait for the evening buses.",
    #     "flair": "Not the A-hole",
    #     "created_utc": 1761061210.0,
    #     "url": "https://www.reddit.com/r/AmItheAsshole/comments/1ocgbt8/aita_for_wanting_to_refuse_to_help_my_unemployed/",
    #     "downs": 0,
    #     "ups": 39,
    #     "score": 39
    # }
    # post_query_json = {
    #     "post_id": "1oce9q9",
    #     "title": "AITA for confronting my parents about money",
    #     "selftext": "My parents make a generous contribution of £100 pounds a month to my child trust fund (I am turning 18 the coming summer) and have done this for several years.  About 2 years ago, my grandma paid £1000 to all her grandchildren (some adults, some not), since her situation meant that she may not be able to make this contribution in the future. Since I was 15 or 16 at the time, my parents took responsibility for the money. Recently, I asked my mum about this money, and when she would pay it into my trust fund.  She said that I was ignoring her generous contribution and that I should treat it as my grandma paying for 10 months worth of my parents contributions. I used the analogy of a child receiving regular pocket money and then upon receiving some money for Christmas, the parents took it, since the child is all ready paid pocket money.. which didn’t seem right to me.  On the other hand, I do recognise that my parents all ready make a very generous contribution. Should I have confronted her about this, and should I again?  EDIT: Let me add some more context.  Roughly around the time of this gift of £1000 pounds, my parents set up premium bonds accounts (essentially government backed savings that are separate to the trust fund) for me, my twin sister, and themselves. My sister’s accounts and mine each had, and still have, the minimum of £25 in. I asked my parents 2 years ago at the time of the gift, to transfer the £1000 to my own premium bonds account, but they said that they would just put it all in a pot in their account (along with their own money). My mum argued it would earn more interest..? I assumed that upon turning 18, £1000 (plus the interest it gathered) would be transferred to me, but apparently not.",
    #     "flair": "Not the A-hole",
    #     "created_utc": 1761056429.0,
    #     "url": "https://www.reddit.com/r/AmItheAsshole/comments/1oce9q9/aita_for_confronting_my_parents_about_money/",
    #     "downs": 0,
    #     "ups": 96,
    #     "score": 96
    # }
    # post_query_json = {
    #     "post_id": "abc2",
    #     "selftext": "My mom has this unshakeable belief that I have no friends, and I would rather be cooped up in my room doing school work and hobbies. She thinks that if I join a sorority, I’ll make some lasting friendships and it’ll solve all of my “problems.” She was in a sorority herself, so her logic is “since I had a good experience, Red will too.”  Originally, I wasn’t really bothered by her pressuring. My college does deferred rush, meaning that the rushing for sororities happens in the spring and not the fall. That way, we get time to go to sorority events and get to know the houses. I thought, “ok, I’ll hear her out and try the events to see if I like it.”  I ultimately ended up feeling that the experience was not for me, and I have expressed this numerous times to my mom. Every time I express this, she thinks up some excuse to dispel my argument like “you have a preconceived notion about the girls in it” or “you just haven’t done enough.” It doesn’t matter how I think or feel, she must find a way to discount it.  It’s gotten to the point where just because I won’t commit to a sorority, I am “making her depressed.” I have experienced so many arguments, yelling, and tears and just “this is hurting me!” It’s become all about herself. Doing well in classes? It doesn’t matter; I’m not doing enough for sororities. I joined this cool club? A club is nothing; sororities are better. If I go home she wants to strike up a conversation about sororities, nothing else. It feels like all of my value here in college comes down to this one thing. It’s making me feel trapped and it’s degrading on my mental health.  She’s even gone the extra mile to share my Instagram with people I don’t know, and give my phone number to another person, whom I also don’t know. I’m not on social media a lot, so this made me very uncomfortable.  I had a professor notice the shift in my mood, so she asked me what was troubling me and I explained this to her. Everyone, including her, that I have explained my situation to has said something along the lines of “it’s not for everyone, it’s ok if you don’t want to do it.” Even my dad encourages me to do what I want. It is only her.  I’ve reached my limit, and I’m at the point where, come this spring, I’m considering not even rushing, not just because I don’t like it, but out of spite. If she wants to make me feel bad about myself because I won’t join a sorority, fine;  I’ll make sure she knows that type of behavior will not get me to do it. It saddens me because what could’ve been this fun cool thing now feels like a burden to me. I go to a sorority event and I just feel this deep sadness; it sucks. If I cave and actually join a sorority, I’m just letting her win, and it encourages her to behave like this again when she can’t get me to do something she wants.  I want to make a note: I’m sure she does this from a place of love, it’s just hurting me.  AITA for doing this out of spite?"
    # }
    # post_query_json = {
    #     "post_id": "1occux6",
    #     "title": "AITA for having my friends toddler take her first steps while her parents weren't there?",
    #     "selftext": "So I'm friends with this guy that I've known since kindergarten. He got married some years ago and now him and his wife have a little baby girl and she's currently at toddler age. I come over often enough that I'm used to seeing the little bugger and she's great. I don't have any other little kids from family or anything in my life so it's been great seeing a little human grow.   Every once in a blue moon they ask me to watch her for a few hours while my friend and his wife get some date time or whatver they need to do. So I was hanging out with her and I know that they've been trying to get her to take her first steps. I read about a trick where if you make them hold something then they will walk without holding onto a surface. So I gave her a toy and filmed it and it worked!   I sent the video to them and my friend didn't care, he was just happy but his wife was pissed! She was mad at me that they weren't there for that big moment. I don't have the type of relationsbio with her where I can have a deep 1 on 1 with her but I talked to my friend about it and he's not upset with me but his wife still is. She thinks I robbed her if a key moment. I did het it on film but I get what she's saying.   I had no malicious intent but was I the AH for getting the toddler to take those first steps? I hi estoy didn't think the trick would work but it did.",
    #     "flair": "Asshole",
    #     "created_utc": 1761053039.0,
    #     "url": "https://www.reddit.com/r/AmItheAsshole/comments/1occux6/aita_for_having_my_friends_toddler_take_her_first/",
    #     "downs": 0,
    #     "ups": 3558,
    #     "score": 3558
    # }
    # post_query_json = {
    #     "post_id": "1odlag6",
    #     "title": "AITA for telling competitive gamer I beat him ?",
    #     "selftext": "I’m just curious cause it bothered me. I liked this one creator. He’s a competitive gamer and he’s really good at this certain team game. I’m a gamer myself but I don’t consider myself good at this game. Well I got him in my game I was so excited because wow he’s a pretty big creator and I’m playing against him. I beat him twice. I went to his discord and sent him a message along the lines of “ his I saw you in my game ! And I beat you twice gg!“ and I left a little funny emoji. I didn’t mean any harm I just thought it was cool that I seen him and I actually won against him. But he just gave me a rude response like “ well win me 1v1, and the only reason you won is because I wasn’t playing with my friends, when I said I didn’t mean any harm he said something like “ well don’t come up in here saying that unless you wanna beat me 1v1 or something. I want playing with my friends either but still I thought he was super rude over this game or was I the asshole for my message ?",
    #     "flair": "Everyone Sucks",
    #     "created_utc": 1761168945.0,
    #     "url": "https://www.reddit.com/r/AmItheAsshole/comments/1odlag6/aita_for_telling_competitive_gamer_i_beat_him/",
    #     "downs": 0,
    #     "ups": 0,
    #     "score": 0
    # }
    # post_query_json = {
    #     "post_id": "1obbxjj",
    #     "title": "AITA for \"not telling\" my friend stuff?",
    #     "selftext": "For context, I'm currently studying abroad in university and I'm on my second year. I met this person within the first few days of the semester and we got close ever since. We're from the same major, which kinda helped with getting closer since we were in the same class.  This semester, we got some new friends. These people are more of the party type, but my friend barely drinks. (We'll call them Oliver to make it easier.) We've invited Oliver to join us for drinks but they refused because: (1) They're not gonna drink. (2) They want to call their partner at nighttime. (3) They don't want to waste money, even for food.   Everything seemed fine up till we (new friend gruop)started hanging out more. Me and Oliver would talk a lot about what went down between people back then (last semester,) mostly funny incidents that happened when we were intoxicated. But this time around was different. We had a new friend group, not just random people that changes every weekend. There were secrets involved, group chats, and (planned) trips.  Now that there's a group chat without Oliver, I can't help but to feel bad, as if I've never invited them to these gatherings. Oliver has also made it clear that even if it is free, they'll never join us because of their partner. Not because their partner is restricting them, but because they mainly call at night. This is another issue that I have with Oliver. (I'll call Oliver's partner Jake.)   Everytime we go out for food, Jake would call. It's not just a quick call, it can last up to 30 minutes and they'd do that repeatedly. I've tried telling Oliver that it makes me feel like they're not valuing my time since we both have busy schedules. Oliver would apologize but wouldn't do anything about it. I tried asking my friend on what I should do in this situation, but they told me that I've done enough and that Oliver should acknowledge what I said properly. It hasn't always been this way, Oliver used to be such a good friend before they got into this said relationship.   Back to the issue, Oliver has never once wanted to join us, yet they always ask me about the tea that happened while we were drunk. I didn't tell them anything. I'd feel horrible for telling them other people's secrets. After telling them that I won't be doing that anymkre they said, \"Oh so that's how it is now.\" I'm not sure whether that's in a good/bad context. I tried telling them that if they join our gatherings, then maybe we'll tell Oliver about our drunk stories. Not sure if it'll actually happen, but based on their reaction, it seemed like a no.   But now my concern is that Oliver will find out about the other plans and feel offended that they're not 'invited.' I feel like a terrible friend",
    #     "flair": "No A-holes here",
    #     "created_utc": 1760938981.0,
    #     "url": "https://www.reddit.com/r/AmItheAsshole/comments/1obbxjj/aita_for_not_telling_my_friend_stuff/",
    #     "downs": 0,
    #     "ups": 2,
    #     "score": 2
    # }
    post_query_json = {
    "post_id": "1o8t39z",
    "title": "AITA for letting my daughter stay with me and my wife after she found out that her mom lied to her",
    "selftext": "My ex and I have a 12 year old daughter, Olivia, with autism level 2. My ex has primary custody and I have Olivia on Wednesdays and every other weekend.   A few months ago my ex told me she was feeling a little burnt out so she wanted us to keep Olivia from Wednesday to Sunday so she could go on a trip with some friends. Olivia is very attached to her mom, so my ex told Olivia that it was a work trip and she had to go.   Last week one of my ex’s friends was at the house with my ex and Olivia and their vacation somehow came up. Olivia figured out that the vacation was the “work trip” that her mom told her about and freaked out because her mom lied to her and her mom doesn’t lie.   She hid in her room for the rest of the day, then called me and asked me to come get her. She’s been with us ever since.   She’s really upset about this. She cries all the time because she wants her mom but she doesn’t know what else her mom lied about and she doesn’t trust her.   Apparently Olivia’s aide called my ex because she’s been having a hard time in school so now my ex wife is demanding that I send Olivia home so they can get back to her routines and start working with her therapist to help her get over this but I told her Olivia will go back when she’s ready. Now she’s threatening to call the police and/or take me back to court over “custodial interference” even though we’ve always had a very friendly co parenting relationship.   My wife thinks we should send her back to avoid drama but I think it should be Olivia’s choice. AITA for refusing to send her back after she found out her mom lied to her?",
    "flair": "Asshole",
    "created_utc": 1760679401.0,
    "url": "https://www.reddit.com/r/AmItheAsshole/comments/1o8t39z/aita_for_letting_my_daughter_stay_with_me_and_my/",
    "downs": 0,
    "ups": 7633,
    "score": 7633
    }
    run_workflow(run_name, post_query_json)
