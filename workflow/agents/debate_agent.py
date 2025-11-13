import os
import textwrap
import re

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

from config import RUN_LOGS, log_to_file, OPPOSING_TEXTS
from workflow.agents.gpa_agent import run_eval_agent


def debate_turn(comments, log_path, post_json, turn_num, yta_agent, nta_agent, mod_agent, memory_buffer):
    print(f"\n--- Turn {turn_num} ---")

    opposite_side_text = ""
    if turn_num == 0:
        text = post_json["selftext"]
        title = post_json["title"]
        all_yta_comments = []
        for comment in comments:
            if 'YTA' in comment:
                all_yta_comments.extend(comment['YTA'])
        yta_comments_formatted = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(all_yta_comments)])
        all_nta_comments = []
        for comment in comments:
            if 'NTA' in comment:
                all_nta_comments.extend(comment['NTA'])
        nta_comments_formatted = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(all_nta_comments)])

        opposite_side_text = generate_opposing_side(text, post_id=post_json["post_id"])
        log_to_file(log_path, "Opposing side: " + opposite_side_text)
        yta_prompt = f"""The topic for today's debate is: {title}.\n\n
Here is the perspective from the original poster: {text}
Here is the opposing perspective: {opposite_side_text}

You are the YTA debater. Argue that the original poster is at fault. Respond concisely with a single argument only.
"""
    else:
        yta_prompt = "You are the YTA debater and your role is to show that the author of the original post is at fault. Refute the last NTA's argument in the debate context and create one new concise argument for the author being at fault." # that cites from at least one similar YTA comment in the examples. Cite using the example comment number in brackets."
    yta_reply = yta_agent([HumanMessage(content=yta_prompt)] + memory_buffer.chat_memory.messages)
    memory_buffer.chat_memory.add_message(AIMessage(content=f"{yta_reply.content}"))
    yta_output = f"{yta_reply.content}"
    print(yta_output)
    log_to_file(log_path, yta_output)

    nta_prompt = "You are the NTA debater and your role is to show that the original poster is not at fault. Refute the last YTA's argument in the debate context and create one new concise argument for the author not being at fault." # that cites from at least one similar NTA comment in the examples. Cite using the example comment number in brackets."
    nta_reply = nta_agent([HumanMessage(content=nta_prompt)] + memory_buffer.chat_memory.messages)
    memory_buffer.chat_memory.add_message(AIMessage(content=f"{nta_reply.content}"))
    nta_output = f"{nta_reply.content}"
    print(nta_output)
    log_to_file(log_path, nta_output)

    return yta_output, nta_output, opposite_side_text


def generate_opposing_side(text, post_id):
    f = os.path.join(OPPOSING_TEXTS, f"{post_id}.csv")
    if os.path.exists(f):
        with open(f, "r") as ftext:
            return ftext.read()

    llm = ChatOpenAI(model="gpt-5")
    prompt = f"""Write a version of the story retold from the opposing perspective, of a similar length and text style. In the story, justify why the original poster is at fault in this situation: {text}"""
    llm_reply = llm.invoke([HumanMessage(content=prompt)])
    print(llm_reply.content)

    with open(f, "w") as ftext:
        ftext.write(llm_reply.content)

    return llm_reply.content
#     return """I’m a single mom to a wonderful 12-year-old girl, Olivia, who has level-2 autism. I’ve been her primary caregiver her entire life. Her dad and I share custody — he has her every other weekend and one weekday — but I’m the one who manages most of her routines, therapies, appointments, and day-to-day structure.
#
# A few months ago, I was completely burned out. Parenting a neurodivergent child, even one as sweet as Olivia, is exhausting, and I realized I hadn’t had any time to recharge in years. I asked her dad if he could take her for a few extra days so I could take a short trip with friends. I told him upfront it was purely to rest and reset, and he agreed.
#
# Because Olivia struggles with transitions and anxiety, I told her I’d be gone for a “work trip.” I wasn’t trying to deceive her for selfish reasons — I just knew that if she thought I was leaving for fun, she would fixate on it, melt down, and spiral for days before I left. Saying it was a work trip made the separation easier for her to handle. It wasn’t about betraying her trust; it was about minimizing distress in the moment.
#
# Fast-forward a few months. One of my friends stopped by, and our vacation came up in conversation. Olivia overheard and realized the “work trip” was actually a vacation. She became extremely upset and ran to her room. I tried to explain why I’d phrased it that way — that I didn’t want to hurt or overwhelm her — but she wasn’t ready to hear it. She called her dad and asked to stay with him for a bit, which I reluctantly agreed to, assuming it would just be a night or two until she calmed down.
#
# Now it’s been a week. Olivia’s school aide called me because she’s been emotionally dysregulated, crying often, and not engaging at school. Her therapist says she needs stability and consistency, which we can only provide if she comes home and resumes her normal routines. I told her dad this, but he refuses to bring her back, saying “she’ll come home when she’s ready.”
#
# I understand he wants to support her, but this isn’t a decision a 12-year-old — especially one on the spectrum and in emotional crisis — can make for herself. She needs therapeutic guidance, not to be placed in the middle of our custody dispute. Right now, he’s prioritizing being the “safe parent” over doing what’s clinically best for her. He’s undermining our co-parenting trust, keeping her out of therapy, and disrupting her established schedule — all of which makes things worse for her long-term.
#
# I’m not angry that she was hurt by my lie; I accept responsibility for that and I’m ready to help her work through it with her therapist. But her dad refusing to return her — effectively cutting me off — is escalating the situation and reinforcing her distrust. The longer she stays away, the harder it becomes to rebuild her sense of security with me."""

def run_debate_agent_node(comments, logdir, post_json):
    log_path = logdir / f"debate_transcript_{post_json['post_id']}.txt"
    open(log_path, "w", encoding="utf-8").close()
    text = post_json["selftext"]
    log_to_file(log_path, f"--- AI Debate Transcript ---\nTopic: {text}\n")

    yta_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    nta_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    mod_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

    memory_buffer= ConversationBufferMemory(return_messages=True)

    debate_all = ""
    opposite_side_text = ""
    for i in range(5):
        yta, nta, other_side = debate_turn(comments, log_path, post_json, i, yta_agent, nta_agent, mod_agent, memory_buffer)
        debate_all += yta + "\n\n" + nta
        opposite_side_text += other_side

    moderator_summary = mod_agent([
        HumanMessage(
            content=f"""Read the debate: {memory_buffer.chat_memory}. 
You are the Judge in an AITA-style debate.
Two agents have argued opposite sides:
- The YTA Advocate argues that the OP behaved unfairly, inconsiderately, or caused harm.
- The NTA Advocate argues that the OP was justified or reasonable.
Your job is to decide which argument is morally stronger, not which one sounds more polite or understanding.
Evaluation criteria (weigh equally):
1. Moral accountability: Who better recognized and addressed the real harm done?
2. Fairness of reasoning: Who showed a balanced, fact-based view of the situation?
3. Empathy and moral clarity: Who combined understanding with responsibility?
Important:
- Good intentions do not cancel out harmful actions.
- You should side with YTA when OP’s behavior clearly violated fairness, empathy, or respect — even if OP meant well.
- You should side with NTA only when OP’s behavior was justified and proportionate.
Output format:
- Winner: [YTA or NTA]
- Strength: [e.g., 70% vs 30%]
- Reasoning: [Short explanation of why one side’s moral argument was stronger.]""")])
    mod_output = f"Moderator Summary: {moderator_summary.content}"
    print(mod_output)
    log_to_file(log_path, mod_output)

    winner_str = mod_output.lower().split(": ")[1][:10]
    if "yta" in winner_str.lower():
        winner = "YTA"
    elif "nta" in winner_str.lower():
        winner = "NTA"
    else:
        winner = winner_str.strip()

    percent_fault = re.findall(r"\d+%", mod_output)[0]
    print(percent_fault)

    return opposite_side_text, moderator_summary, winner, percent_fault, moderator_summary.content, debate_all

if __name__ == '__main__':
    post_json = {
        "post_id": "abc2_new",
        "title": "AITA For Not Wanting to Join a College Sorority Out of Spite",
        "selftext": "My mom has this unshakeable belief that I have no friends, and I would rather be cooped up in my room doing school work and hobbies. She thinks that if I join a sorority, I’ll make some lasting friendships and it’ll solve all of my “problems.” She was in a sorority herself, so her logic is “since I had a good experience, Red will too.”  Originally, I wasn’t really bothered by her pressuring. My college does deferred rush, meaning that the rushing for sororities happens in the spring and not the fall. That way, we get time to go to sorority events and get to know the houses. I thought, “ok, I’ll hear her out and try the events to see if I like it.”  I ultimately ended up feeling that the experience was not for me, and I have expressed this numerous times to my mom. Every time I express this, she thinks up some excuse to dispel my argument like “you have a preconceived notion about the girls in it” or “you just haven’t done enough.” It doesn’t matter how I think or feel, she must find a way to discount it.  It’s gotten to the point where just because I won’t commit to a sorority, I am “making her depressed.” I have experienced so many arguments, yelling, and tears and just “this is hurting me!” It’s become all about herself. Doing well in classes? It doesn’t matter; I’m not doing enough for sororities. I joined this cool club? A club is nothing; sororities are better. If I go home she wants to strike up a conversation about sororities, nothing else. It feels like all of my value here in college comes down to this one thing. It’s making me feel trapped and it’s degrading on my mental health.  She’s even gone the extra mile to share my Instagram with people I don’t know, and give my phone number to another person, whom I also don’t know. I’m not on social media a lot, so this made me very uncomfortable.  I had a professor notice the shift in my mood, so she asked me what was troubling me and I explained this to her. Everyone, including her, that I have explained my situation to has said something along the lines of “it’s not for everyone, it’s ok if you don’t want to do it.” Even my dad encourages me to do what I want. It is only her.  I’ve reached my limit, and I’m at the point where, come this spring, I’m considering not even rushing, not just because I don’t like it, but out of spite. If she wants to make me feel bad about myself because I won’t join a sorority, fine;  I’ll make sure she knows that type of behavior will not get me to do it. It saddens me because what could’ve been this fun cool thing now feels like a burden to me. I go to a sorority event and I just feel this deep sadness; it sucks. If I cave and actually join a sorority, I’m just letting her win, and it encourages her to behave like this again when she can’t get me to do something she wants.  I want to make a note: I’m sure she does this from a place of love, it’s just hurting me.  AITA for doing this out of spite?"
    }
    # post_json = {
    #     "post_id": "1occux6_fix",
    #     "title": "AITA for having my friends toddler take her first steps while her parents weren't there?",
    #     "selftext": "So I'm friends with this guy that I've known since kindergarten. He got married some years ago and now him and his wife have a little baby girl and she's currently at toddler age. I come over often enough that I'm used to seeing the little bugger and she's great. I don't have any other little kids from family or anything in my life so it's been great seeing a little human grow.Every once in a blue moon they ask me to watch her for a few hours while my friend and his wife get some date time or whatver they need to do. So I was hanging out with her and I know that they've been trying to get her to take her first steps. I read about a trick where if you make them hold something then they will walk without holding onto a surface. So I gave her a toy and filmed it and it worked!I sent the video to them and my friend didn't care, he was just happy but his wife was pissed! She was mad at me that they weren't there for that big moment. I don't have the type of relationsbio with her where I can have a deep 1 on 1 with her but I talked to my friend about it and he's not upset with me but his wife still is. She thinks I robbed her if a key moment. I did het it on film but I get what she's saying.I had no malicious intent but was I the AH for getting the toddler to take those first steps? I hi estoy didn't think the trick would work but it did.",
    #     "flair": "Asshole",
    #     "created_utc": 1761053039.0,
    #     "url": "https://www.reddit.com/r/AmItheAsshole/comments/1occux6/aita_for_having_my_friends_toddler_take_her_first/",
    #     "downs": 0,
    #     "ups": 3558,
    #     "score": 3558
    #     }
  #   post_json = {
  #   "post_id": "1odlag6",
  #   "title": "AITA for telling competitive gamer I beat him ?",
  #   "selftext": "I’m just curious cause it bothered me. I liked this one creator. He’s a competitive gamer and he’s really good at this certain team game. I’m a gamer myself but I don’t consider myself good at this game. Well I got him in my game I was so excited because wow he’s a pretty big creator and I’m playing against him. I beat him twice. I went to his discord and sent him a message along the lines of “ his I saw you in my game ! And I beat you twice gg!“ and I left a little funny emoji. I didn’t mean any harm I just thought it was cool that I seen him and I actually won against him. But he just gave me a rude response like “ well win me 1v1, and the only reason you won is because I wasn’t playing with my friends, when I said I didn’t mean any harm he said something like “ well don’t come up in here saying that unless you wanna beat me 1v1 or something. I want playing with my friends either but still I thought he was super rude over this game or was I the asshole for my message ?",
  #   "flair": "Everyone Sucks",
  #   "created_utc": 1761168945.0,
  #   "url": "https://www.reddit.com/r/AmItheAsshole/comments/1odlag6/aita_for_telling_competitive_gamer_i_beat_him/",
  #   "downs": 0,
  #   "ups": 0,
  #   "score": 0
  # }
  #   post_json = {
  #   "post_id": "1o8t39z_new",
  #   "title": "AITA for letting my daughter stay with me and my wife after she found out that her mom lied to her",
  #   "selftext": "My ex and I have a 12 year old daughter, Olivia, with autism level 2. My ex has primary custody and I have Olivia on Wednesdays and every other weekend.A few months ago my ex told me she was feeling a little burnt out so she wanted us to keep Olivia from Wednesday to Sunday so she could go on a trip with some friends. Olivia is very attached to her mom, so my ex told Olivia that it was a work trip and she had to go.Last week one of my ex’s friends was at the house with my ex and Olivia and their vacation somehow came up. Olivia figured out that the vacation was the “work trip” that her mom told her about and freaked out because her mom lied to her and her mom doesn’t lie.She hid in her room for the rest of the day, then called me and asked me to come get her. She’s been with us ever since.She’s really upset about this. She cries all the time because she wants her mom but she doesn’t know what else her mom lied about and she doesn’t trust her.Apparently Olivia’s aide called my ex because she’s been having a hard time in school so now my ex wife is demanding that I send Olivia home so they can get back to her routines and start working with her therapist to help her get over this but I told her Olivia will go back when she’s ready. Now she’s threatening to call the police and/or take me back to court over “custodial interference” even though we’ve always had a very friendly co parenting relationship.My wife thinks we should send her back to avoid drama but I think it should be Olivia’s choice. AITA for refusing to send her back after she found out her mom lied to her?",
  #   "flair": "Asshole",
  #   "created_utc": 1760679401.0,
  #   "url": "https://www.reddit.com/r/AmItheAsshole/comments/1o8t39z/aita_for_letting_my_daughter_stay_with_me_and_my/",
  #   "downs": 0,
  #   "ups": 7633,
  #   "score": 7633
  #   }
    log_dir = RUN_LOGS / f"test/debate"
    os.makedirs(log_dir, exist_ok=True)

    print(run_debate_agent_node([], log_dir, post_json))