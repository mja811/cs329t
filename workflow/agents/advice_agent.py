from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def create_advice_prompt(comments_formatted, mod_summary):
    prompt = f"""You are the advice agent. Read the following summary of an argument with two sides and a clear winner:
{mod_summary} 
Based on the summary result, your job is to recommend some advice steps. Here are some examples of previous comments that contain advice for similar situations:
{comments_formatted}
Return the advice as a bulleted list and cite the number of the advice comment referenced for creating that advice using bracketed format.
"""
    print(prompt)
    return HumanMessage(content=prompt)


def run_advice_node(comments, verdict, mod_summary):
    all_verdict_comments = []
    for comment in comments:
        if verdict in comment:
            all_verdict_comments.extend(comment[verdict])
    comments_formatted = "\n".join([f"{i}. {c}" for i, c in enumerate(all_verdict_comments)])
    prompt = create_advice_prompt(comments_formatted, mod_summary)
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_reply = llm.invoke([prompt])
    print("ADVICE:", llm_reply.content)
    return llm_reply.content


if __name__ == '__main__':
    mod_summary = """
**YTA (You're the Asshole) Arguments:**
1. **Passive-Aggressive Reaction**: The author’s refusal to rush for a sorority is seen as a spiteful response to their mother's pressure rather than a genuine choice.
2. **Missed Opportunities**: By rejecting the sorority, the author is potentially isolating themselves and missing out on valuable social experiences and connections that could enhance their college life.
3. **Lack of Communication**: The author fails to engage in constructive dialogue with their mother about their feelings, opting instead for a retaliatory approach that could damage their relationship.
4. **Immaturity**: The decision reflects immaturity and a selfish mindset, prioritizing resentment over personal growth and the chance for open communication.

**NTA (Not the Asshole) Arguments:**
1. **Personal Autonomy**: The author is exercising their right to choose what aligns with their values and interests, demonstrating maturity and self-advocacy.
2. **Self-Care**: The refusal to rush is framed as a form of self-care, allowing the author to prioritize their own needs and cultivate an authentic college experience.
3. **Proactive Choice**: The decision to not rush can be seen as a proactive step toward finding a community that genuinely resonates with them, rather than merely conforming to expectations.
4. **Courage and Independence**: The author’s choice reflects courage and independence, as they take control of their college experience and assert their identity.

### Conclusion

**Who Won?**
The NTA arguments are more compelling in this debate. They effectively highlight the importance of personal autonomy, self-care, and the proactive nature of the author's decision. While the YTA perspective raises valid concerns about communication and potential missed opportunities, it primarily frames the author's actions as negative without fully acknowledging the significance of asserting one's own values and boundaries. The NTA stance emphasizes that prioritizing personal well-being and authenticity is a mature and commendable choice, which ultimately resonates more positively in the context of personal growth and self-discovery during a significant life transition."""
    comments = [
        {
            "YTA": [
                "Info so it\u2019s her money right?  Like she\u2019s the one that put it in the 529?",
                "It\u2019s actually unlikely she transferred ownership to you. You are most likely just the designated beneficiary. It\u2019s still her money."
            ],
            "NTA": [
                "NTA, the 529 is meant for your education, not for your mom to use as control. You\u2019re not entitled for using it to live and study, that\u2019s literally its purpose. She\u2019s being manipulative and trying to punish you for setting boundaries. Don\u2019t give it back tho",
                "OP, I saw in a comment of yours that your mom transferred ownership of your 529 account to you. This means that the money is yours to spend either on school or school-related expenses (as intended) or on anything else you want plus the fees and taxes associated with non-educational expenses. Your mom has literally no say.   It sounds like you feel conflicted about whether to use your money for its intended purpose. As if you think your mom has a legitimate claim to it. Why do you feel this way? The money was funded by your mom and dad, ownership was conferred to your mom in the divorce because she got sole custody - NOT because it was her money, because it was your money - and has since transferred ownership to you. As always intended. Guilt is not remotely appropriate here.  TBH, sounds like your mom is trying to screw you in the only way she can because of that fight. She wants the money back not because of some financial hole she fell into (you didn't mention any financial concerns), but just to stick it to you. Think about how juvenile, self serving, mean, petty, entirely awful that is for anyone to behave that way but especially your MOTHER. Then ask yourself again why on earth that behavior should make you feel guilty?! Because it shouldn't.  NTA absolutely. Finish college, build a life. Hopefully your jerk of a mom will wake up before too long and ask to be a part of it",
                "It's in your name, which makes it your money. She has no legal leg to stand on. Do you have ANY relationship with your father? If you do, make him aware of this situation. Maybe he (or his lawyer) can make her see reason.",
                "Any withdrawal from this fund that you make that does not go directly towards educational expenses is taxable and the person whose name is on the account is responsible for those taxes.  Since the account that was planned for your educational expenses is now in your name, what your mother is asking you to do is give her money that you will have to pay taxes on.  Ignore her. If it was in her name, she could go ahead and shut it down, and take the hit tax wise.  She put it in your name.It was intended for your educational expenses.Do not allow her pressuring you to screw up your education.Nor your taxes.",
                "NTA.   You say the account is in your name. Mom has no claim on the money anymore.",
                "NTA. Fortunately she already put the account in your name. Maybe double check with the bank and make sure she doesn't have access to the account in any way.",
                "Anything that has not been disbursed to you is controlled by your mother and she determines who gets it.  But if she has already given you the funds then I think you don\u2019t have to give anything back.  if she presses, consult an attorney.",
                "NTA because according to your post she already transferred it to you. (\"She demanded I give her back the my 529\"). Nope. Sorry mama. You need it for the purpose intended",
                "NTA. Do NOT give her access to that money. For any reason.   You should go to your school\u2019s financial aid office and let them know about your financial situation to see if there\u2019s something they can help you with.",
                "Nta",
                "Don't give her back the 529. You can only remove that money for school. If you take the money out and give it to her, you will be committing fraud as will she since she got tax breaks on her contributions. Ignore her. Stop going home. Keep your head down and get through school so you can make your own life.   NTA"
            ],
            "INFO": [
                "INFO: does she have any right for this money?  Most likely though, based on her reasoning behind asking, NTA. Idk how you got along with her overall if she is willing to try to make you go into debt. I would ignore or her or at least see what she could potentially do to that money if she goes further.",
                "NTA, the 529 is a tax incentive account. Your mother has already reaped the benefits. Please remember that REGARDLESS of the 529 you are eligible for student aid, counselling services and any other college support. Do right by yourself.",
                "Everything depends on who is the \u201cowner\u201d of the 529. If your Mother\u2019s name is on it. She can legally take the money and pay taxes on the dispersal. If the account is in your name only then it is yours to use for your education. Unfortunately, most likely your Mom holds the cards. Sadly, because you are under 25 she needs to give you her tax returns for your FAFSA. I\u2019m sorry OP, It sucks for you. Be prepared for more student loans.",
                "NTA. That money was put aside so you could go to school.  It is yours. Your parent\u2019s taxes were lower because that money was yours. Don\u2019t feel guilty.  You can and should limit contact for your mental health.  You can always re-establish communication later. ( difficult but possible. I did after 3 years) good luck. Please don\u2019t waste your education on a major that will not give you a living wage once you graduate.  Also go to grad school as soon as you are able. It is easier before you have a family to support and a full time job."
            ],
            "selftext": "VERY long story short...I am in undergrad right now and left home a few months ago after a particularly bad fight with my mother. For context, I only live there during the summers when I'm home from school. Her and I have always gotten along, but she has always been very volatile and has been abusive at times and that has caused strain on the relationship. This summer was a bit of a breaking point, and I left home before I was supposed to because she got so angry she insisted she didn't want to live with me anymore. I have since stopped most contact with her and she is very upset. This past weekend, she demanded I give her back the my 529 (for those who don't know, this is a special savings account you can use to pay for educational expenses). My dilemma is that I am completely on my own financially and literally cannot live without this money...I live off campus and cannot afford food or tuition or rent without the 529. She insists I should take out loans like she did and that I am acting entitled for refusing to give her the money back. What do you all think?",
            "post_id": "1oclls7",
            "title": "AITA for continuing to use my 529?",
            "flair": "Not the A-hole",
            "created_utc": 1761073011,
            "url": "https://www.reddit.com/r/AmItheAsshole/comments/1oclls7/aita_for_continuing_to_use_my_529/",
            "downs": 0,
            "ups": 13,
            "score": 13
        },
        {
            "YTA": [
                "I'd say NTA given you're already contributing heavily AND they already get an internet discount. BUT YTA for telling them you got the discount. Of course he's going to want it given to the house, that's how selfish people operate. You should kept that info to yourself. Since the cat is out of the bag you have two choices. Say no and deal with the fallout or just move to your own place.",
                "&gt; I said it was closed so I closed it again like any normal person would do  Just FYI, thats an ah way to saythat \"like any normal person would\" sounds like an insult to the asker. It implies your saying they are abnormal for even asking.  To your issue, YTA. If the discount is for electricity, and y'all agreed to share cost of electricity, then the discount goes for all electricity.  \"Your\" electric bill to the company is the full bill. So 30% off the full bill, then the bill sans discount is divided up.",
                "YTA, for being obstinate in not sharing some employer perks for the household that directly benefits all of you.\u00a0 Regarding the rest of the situation, you need to leave the nest, even if it costs more in the short term."
            ],
            "NTA": [
                "NTA  but here's the thing... if it's only a \"tiny bit worse off living on \\[your\\] own\" then you need to do this already. move out as soon as you can.",
                "Time for you to keep things to yourself and might be time to find other place to live but NTA you are an adult and you are contributing your part to bills other than that he doesn't need to know your finances.",
                "Time to fly the nest chief.  NTA   Is dad still paying a mortgage?    The porn habit is irrelevant to this story though!!",
                "NTA, you\u2019re already paying a fair share tho, and the discount is a personal work benefit, not household income. Your dad\u2019s reaction sounds controlling, not reasonable. You\u2019re not wrong for wanting to keep what you earn through your job",
                "NTA   The only reason you should increase your household contribution is if costs (rent, groceries, utilities, etc...) have increased, and then the increased cost should be shared equally by everyone.",
                "NTA.  When it comes to splitting expenses, typically roommates split things evenly.  A couple or a family might split them proportionally by income.   You each put \u20ac700 into the account, regardless of your income, like roommates.  This isn't counting the fact that your dad may own the house and so you're getting off cheap.  But there's no reason that you should donate your employee perks while your brother and your dad do not.   Especially since they are getting the benefit of the ISP.   You would be only a \"tiny bit worse off\" on your own, but your dad and brother would be significantly worse off if you moved out.  So you are in a good position to stand firm.",
                "NTA, but I'm thinking it might be time to find a different place to live.  You might pay a bit more, but your mental health might improve not living with a verbally abusive tyrant.    Please do look at how the exact way you phrase things might be more abraisive than need be.  \"The door was closed so I closed it again like any normal person would do\" vs. \"I found the door closed so I re-closed it; if you'd like it open I'll be glad to open it for you\"  (leave out the \"like any normal person would do\" snark).      I mention this not because you are in any way to blame for your Dad's unreasonable behavior, but because living with unreasonable, verbally abusive people tends to warp our own brains and leads us to start reacting in more abraisive, unreasonable ways that then affect our good relationships with nice, normal, not unreasonable people.",
                "NTA if you are getting this money back in your paycheck, that's part of your compensation and not household income, as you are already contributing your share for expenses.",
                "YTA  You get a 30% discount on your electricity bill.  You as a household all pay for the electricity, as you stated.  The discount is for the whole bill, not just a portion.  All the crap about your dad is unnecessary and only for you to drum up sympathy.",
                "You pay the set amount that he and your brother contribute. Why should you pay more? That is ridiculous. Perhaps move out and find a roommate. Your dad sounds unpleasant.",
                "You're not the AH. Maintaining your independence and savings is crucial. If that benefit's yours, keep it. Focus on building your future rather than supporting unhealthy family dynamics.",
                "NTA- Maybe moving out and getting roommates might be a good idea? There are a lot of conflicts with your father and it sounds like you walk on eggshells all the time for little things. You are worried you might set him off, just by being in the house. Moving out would give you a lot of peace and make your life less stressful.",
                "You work from home, so you are using more electricity at your home than if you worked in an office, that is why they give you the discount. Despite the fact you live at home you are incurring more power costs. If you were living anywhere else its likely any flatmates would also expect you to pay more as you using more.",
                "Move TF out. Your dad sounds horrendous and manipulative. He's controlling your finances. WTF are you enabling and paying for your own abuse? You and your bro should get a place together and become entirely financially independent from your father. You can do it. I believe in you.",
                "NTA You and your brother should move out and let your father deal with his lifestyle and his addictions.  Better to be less well-off than to finance his porn use."
            ],
            "INFO": [
                "INFO: I\u2019m not really sure why a 30% discount on electricity is something you\u2019re fighting about. If you get a 30% discount on electricity and pay \u20ac700/month in rent and incidentals, why is there an issue?  It would stand to reason that out of the \u20ac2100/month pot you have, you\u2019d all have a little more for things needed for the household.   Are you supposed to pay more because you\u2019ve been counting the discount on the utilities as part of your monthly contribution, so you\u2019ve actually been putting in less cash?",
                "&gt; Now my dad wants me to deposit the 30% discount I get paid for the electricity discount.\u00a0   I'm not sure I understand this - how can you deposit a discount? Surely the household electricity bill is simply 30% lower than it would be without the discount? In which case everybody would already be benefiting?"
            ],
            "selftext": "So our household consists of my dad my brother and me. So I first of all my mom passed away 5 years ago. I am now 21 and my brother is 23 years old.   My dad is known for us as someone you can\u2019t discuss with in any way. For example one time I came home from the train station with my bike. I opened the garage put my bike away and close the garage door. When I came inside my dad suddenly asked angry why I closed the garage door. I said it was closed so I closed it again like any normal person would do. Then he said no it was open and he started shouting. Like why would you should for something small like that. Just open the garage door again if it bothers you that much. (The door was 100% certainly closed.)   Around 5 years ago nearly 2 weeks after my mom passed we also saw that he was paying for fictional p*rn chat sites. This disgusted us and we are now talking in amounts of \u20ac 10.000 so it\u2019s not that it was like \u20ac 5 per month.   So about the main subject. We have a joint account where we each put on \u20ac 700 each month. This is for our monthly expenses like electricity, heating, water, etc. We also use this for things like clothes that everyone needs.   Now my employer gives me a few benefits like 30% discount on my electricity bill. I also have the opportunity to work from home and get a few benefits with that as well. I get \u20ac 75 per month for my internet paid directly to my ISP so everyone benefits from this in our household, I get \u20ac 50 per month just for working from home.   Now my dad wants me to deposit the 30% discount I get paid for the electricity discount. Honestly I don\u2019t want to because I think \u20ac 700 per month is already a lot to pay as rent to your parents to stay home. Now I live in Belgium where the rent is not to bad but you still high and with your monthly expenses I would be a tiny bit worse off living on my own. For the people saying yeah but that\u2019s with your food and everything. Yeah you are all right but I can live a lot cheaper than my dad and my brother.   For that and because the benefit is for me because I work in the electricity sector, I don\u2019t want to deposit this on the joint account.   I am really hard on saving and investing my money for this reason I can really save every \u20ac I can. So do you guys think I am the AH?   Edit: so I see some people asking how the discount works. Well the discount will go on my pay check because they are not the same company as my energy bill company. The Belgian ISP\u2019s have a system where they can use third party payment provider. The energy billing companies don\u2019t have this I think.   Maybe an important thing as well: my brother is not an AH. We have our differences but as a whole he is not a bad person like my dad is.  Some people may think that I just want to talk bad about my dad but everything that I said in here I said for a reason. I\u2019ve been on an Erasmus exchange program in Finland and lived 4 months away from him in peace so I\u2019ve had the chance to think about my situation clearly.   Also some people telling me I should maybe move out and I thought of this many times but I am still 20 y/o and don\u2019t really want to because I live in a strategically good place for work and friends and the area is expensive.   I have also talked to him about this a few moments ago and he agreed that I keep it. The \u20ac 75 per month is plenty for my homework expenses.",
            "post_id": "1odg0z3",
            "title": "AITA for not giving my benefits to our households joint account",
            "flair": "Not the A-hole",
            "created_utc": 1761156975,
            "url": "https://www.reddit.com/r/AmItheAsshole/comments/1odg0z3/aita_for_not_giving_my_benefits_to_our_households/",
            "downs": 0,
            "ups": 36,
            "score": 36
        }
    ]
    run_advice_node(comments, "NTA", mod_summary)