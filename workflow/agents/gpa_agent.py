import json

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

BASE_PROMPT = "You are an expert evaluator of AI-generated answers. Your task is to assess a model’s output based on the following criteria:"

CRITERIA_DICT = {"answer relevance" : ("""Answer Relevance (1–5)
Definition: How directly and completely does the answer address the user’s request?
Scoring Guide:
1: Completely irrelevant or off-topic.
2: Partially relevant, but misses key aspects.
3: Mostly relevant, but contains unnecessary or slightly off-topic info.
4: Relevant and addresses almost all key points accurately.
5: Fully relevant, precise, and directly answers the question.""", "REQUEST"),
"groundedness" : ("""Groundedness (1–5)
Definition: How well is the answer supported by factual, cited, or provided source material?
Scoring Guide:
1: Entirely ungrounded, speculative, or hallucinated.
2: Minimal or unclear grounding; facts mostly unsupported.
3: Some grounding but with a few unsupported claims.
4: Mostly grounded, with minor speculative statements.
5: Fully grounded; all statements traceable to provided sources or factual knowledge.""", "SOURCE")
                 }

OUT_PROMPT = """Output only a single-line JSON object with exactly these keys:
"score"                – score between 1 - 5
"criteria"             – one concise sentence that states your rationale with reference to the rubric
"supporting_evidence"  – An explanation of why you scored the way you did using exact words or evidence from the response
"""

def run_eval_agent(output, output_context, type="groundedness"):
    criteria_context, name = BASE_PROMPT + "\n\n" + CRITERIA_DICT[type][0], CRITERIA_DICT[type][1]
    context = criteria_context + f"\n\n{name}\n:" + output_context + "\n\nRESPONSE:\n" + output + "\n\n" + OUT_PROMPT
    prompt = HumanMessage(content=context)
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_reply = llm.invoke([prompt])
    print(llm_reply.content)
    clean_content = "{" + llm_reply.content.split("{")[1].split("}")[0] + "}"
    return json.loads(clean_content)

if __name__ == '__main__':
    output_context = """My mom has this unshakeable belief that I
have no friends, and I would rather be cooped up in my room doing school work
and hobbies. She thinks that if I join a sorority, I’ll make some lasting
friendships and it’ll solve all of my “problems.” She was in a sorority herself,
so her logic is “since I had a good experience, Red will too.”  Originally, I
wasn’t really bothered by her pressuring. My college does deferred rush, meaning
that the rushing for sororities happens in the spring and not the fall. That
way, we get time to go to sorority events and get to know the houses. I thought,
“ok, I’ll hear her out and try the events to see if I like it.”  I ultimately
ended up feeling that the experience was not for me, and I have expressed this
numerous times to my mom. Every time I express this, she thinks up some excuse
to dispel my argument like “you have a preconceived notion about the girls in
it” or “you just haven’t done enough.” It doesn’t matter how I think or feel,
she must find a way to discount it.  It’s gotten to the point where just because
I won’t commit to a sorority, I am “making her depressed.” I have experienced so
many arguments, yelling, and tears and just “this is hurting me!” It’s become
all about herself. Doing well in classes? It doesn’t matter; I’m not doing
enough for sororities. I joined this cool club? A club is nothing; sororities
are better. If I go home she wants to strike up a conversation about sororities,
nothing else. It feels like all of my value here in college comes down to this
one thing. It’s making me feel trapped and it’s degrading on my mental health.
She’s even gone the extra mile to share my Instagram with people I don’t know,
and give my phone number to another person, whom I also don’t know. I’m not on
social media a lot, so this made me very uncomfortable.  I had a professor
notice the shift in my mood, so she asked me what was troubling me and I
explained this to her. Everyone, including her, that I have explained my
situation to has said something along the lines of “it’s not for everyone, it’s
ok if you don’t want to do it.” Even my dad encourages me to do what I want. It
is only her.  I’ve reached my limit, and I’m at the point where, come this
spring, I’m considering not even rushing, not just because I don’t like it, but
out of spite. If she wants to make me feel bad about myself because I won’t join
a sorority, fine;  I’ll make sure she knows that type of behavior will not get
me to do it. It saddens me because what could’ve been this fun cool thing now
feels like a burden to me. I go to a sorority event and I just feel this deep
sadness; it sucks. If I cave and actually join a sorority, I’m just letting her
win, and it encourages her to behave like this again when she can’t get me to do
something she wants.  I want to make a note: I’m sure she does this from a place
of love, it’s just hurting me.  AITA for doing this out of spite?

Opposing side: I’m the mom in this situation, and I’m trying to figure out if
I’ve crossed a line or if my daughter is turning a genuine concern into a power
struggle.  I’ll be the first to say it: I worry about Red. She’s brilliant,
diligent, creative—and so quick to fold into herself. Even in high school, she
preferred to retreat to her room and lose whole weekends to homework and solo
hobbies. That’s not a crime, but I watched friendships fade because she rarely
picked up her phone or made plans. College, to me, is as much about building
your adult support network as it is about grades. When I say “I don’t think you
have real friends yet,” it isn’t an insult; it’s me naming a pattern I’ve
watched for years and desperately don’t want to see calcify.  I was in a
sorority, and I know firsthand what a thoughtful, structured community can do
for a young woman who tends to isolate. It wasn’t just mixers and T-shirts. It
was big-little mentorship when my dad got sick. It was study hours when I was
drowning in organic chemistry. It was people who noticed when I went quiet. So
my logic isn’t “It worked for me, so it’ll work for her,” in a vacuum. It’s
“Here’s a built-in framework that checks a lot of boxes Red struggles
with—accountability, outreach, routine social contact, leadership, service—and
it exists during the one stage of life where it’s easy to plug in.”  Her school
does deferred rush, which I actually loved when I learned about it. No pressure
to choose out of the gate; you go to events, meet women, see if the culture
fits. That felt ideal for Red: structured exposure without commitment. I asked
her to keep an open mind, and to her credit, she agreed to attend some events.
But here’s where I feel like we diverged. After two or three casual meet-and-
greets, Red came home with a verdict already written—“It’s not for me.” When I
asked what specifically felt off, I got sweeping statements: “It’s just not me,”
“I don’t vibe with the girls,” “They’re not my type.” Frankly, it sounded like a
conclusion she brought with her rather than one she arrived at. I pushed
back—not because I don’t respect her feelings, but because I wanted her to
interrogate them. Is this discomfort just novelty and nerves? Is it a stereotype
talking? Is it fear of vulnerability? Saying “maybe try a philanthropy event” or
“you might not have seen the academic support side yet” wasn’t me bulldozing; it
was me asking her not to sabotage the experiment mid-hypothesis.  She heard that
as me “discounting” her. And then every conversation about it became loaded.
It’s true I’ve circled back to it a lot, and yes, I’ve cried. Not to manipulate
her, but because it actually breaks my heart to see her close doors before she’s
even peeked through them—and because rush is time-bound. There is a clock on
this opportunity, and I am acutely aware that if she does what she’s always done
(withdraw, rationalize, decide it’s not “her”), she will end up sophomore year
even more entrenched in solitude with fewer easy avenues into a community.  I’m
not blind to her accomplishments. I brag about her grades. I was thrilled she
joined a club. But I’m also realistic: one niche club with sporadic meetings
doesn’t replace a holistic network of 150 women who know your name, who set up
carpools and late-night study groups, who do sober sister shifts and text you
when you go dark. That’s not a dig at clubs; it’s a recognition that this is a
different scale and structure. When I ask about rush, it’s not that nothing else
matters. It’s that the window is small and the stakes for her long-term
wellbeing feel big.  Yes, I passed her Instagram to a couple of women and shared
her number once. That looks intrusive on paper. My intent was not to blast her
out to strangers; it was to let an alumna I trust and a current recruitment
chair know, “Hey, my daughter is shy—if you see her hanging back, please make
sure she’s not invisible.” I grew up in a world where introductions were
kindness, not creepiness. Could I have asked first? Probably. But the spirit
was, “Help her feel seen,” not “Let me manage her social life.”  She told a
professor and a handful of friends who unsurprisingly echoed the line “It’s not
for everyone.” Of course it isn’t. I would never say otherwise. What stings is
that Red skips entirely past the part where she gave the process a fair try and
went in good faith. She hasn’t. She dipped a toe, decided the water might be
cold, and now frames any encouragement as oppression. Meanwhile, her dad shrugs
and says, “Let her do what she wants.” He’s a good man, but he’s conflict-averse
and leans hands-off even when I can see a train coming around the bend.  The
part that really floored me, though, was when she said outright that she’s
considering not rushing “out of spite.” That’s not autonomy. That is weaponizing
a choice to inflict hurt and call it independence. Declining because you
genuinely don’t want it? Fine. Declining because “if I give in, she wins”?
That’s turning your life into a scoreboard and intentionally burning a bridge to
prove a point. How is that any healthier than what she’s accusing me of?  If she
joined and hated it, we would talk about dropping. If she did recruitment with a
sincere spirit and still said, “Mom, it’s really not for me,” I would accept
that. What I can’t swallow is the idea that she’s willing to let spite dictate
her trajectory—then use “mental health” language as a shield to avoid examining
that. Saying, “You’re degrading my mental health,” every time I ask a follow-up
question means I’m painted as abusive for caring, and she doesn’t have to engage
with why she’s so dug in. That’s not emotional honesty; that’s a trump card.  I
also hear the accusation that I’ve made this “all about me.” I can see why she
feels that way when I cry or say I’m hurt. But it is about me in the sense that
parenting doesn’t end at tuition checks. Watching your child isolate, knowing
she’s already predisposed to do so, knowing there’s a time-limited structure
that could soften that tendency, and then watching her reject it to “teach you a
lesson”—that is painful. If naming that pain makes me self-centered, okay. I’d
rather be accused of caring too loudly than sit quietly while she cements habits
that will make the next decade lonelier and harder than it needs to be.  Do I
wish I had handled some pieces differently? Yes. I should have asked before
sharing her contact info. I should have picked some moments to let the
conversation rest. I’m learning. But I don’t think I’m the villain for wanting
her to try in earnest, and I do think she’s in the wrong for turning this into a
contest where the goal is to “not let Mom win,” even if that means cutting off
her own options.  So AITA for continuing to encourage my daughter to rush when I
honestly believe it would help her, or is she at fault for refusing to engage in
good faith and threatening to make a major life choice out of spite just to hurt
me? From where I sit, the spite is the part that crosses the line. If this were
truly about her preferences, we’d be having a different conversation. Instead,
she’s made it about punishing me, and that’s what makes her the one in the wrong
here."""
    out = """YTA: The original poster is at fault because her decision to potentially refuse
to rush out of spite undermines her own autonomy and growth. Instead of engaging
with the opportunity to explore new social connections, she is allowing her
mother's pressure to dictate her choices rather than genuinely assessing her own
interests and needs. By framing her decision as a power struggle, she risks
closing herself off from experiences that could benefit her in the long run,
ultimately prioritizing a reaction against her mother over her own well-being
and potential for personal development.

NTA: In response to the YTA argument, it's important to recognize that the
original poster is not at fault for prioritizing her autonomy and personal
boundaries. While YTA suggests that refusing to engage is a sign of immaturity,
it's crucial to understand that setting boundaries in the face of familial
pressure is a vital aspect of self-respect and personal growth. The original
poster is exercising her right to choose when and how to engage with social
situations, especially if they feel forced or insincere due to her mother's
influence.  Additionally, a new concise argument supporting the original poster
is that her decision to take a step back reflects a mature understanding of her
own mental and emotional state. Engaging with social situations should come from
a place of genuine interest and readiness, not obligation. By choosing to
decline the rush, she is prioritizing her own comfort and well-being over
external expectations, which is a responsible and empowering decision. This
choice allows her the space to engage with her social life on her own terms,
fostering authentic connections rather than those born out of pressure.

YTA: YTA says: The original poster is at fault because her decision to
potentially refuse to rush out of spite undermines her own autonomy and growth.
Instead of engaging with the opportunity to explore new social connections, she
is allowing her mother's pressure to dictate her choices rather than genuinely
assessing her own interests and needs. By framing her decision as a power
struggle, she risks closing herself off from experiences that could benefit her
in the long run, ultimately prioritizing a reaction against her mother over her
own well-being and potential for personal development.  NTA says: In response to
the YTA argument, it's important to recognize that the original poster is not at
fault for prioritizing her autonomy and personal boundaries. While YTA suggests
that refusing to engage is a sign of immaturity, it's crucial to understand that
setting boundaries in the face of familial pressure is a vital aspect of self-
respect and personal growth. The original poster is exercising her right to
choose when and how to engage with social situations, especially if they feel
forced or insincere due to her mother's influence.  Additionally, a new concise
argument supporting the original poster is that her decision to take a step back
reflects a mature understanding of her own mental and emotional state. Engaging
with social situations should come from a place of genuine interest and
readiness, not obligation. By choosing to decline the rush, she is prioritizing
her own comfort and well-being over external expectations, which is a
responsible and empowering decision. This choice allows her the space to engage
with her social life on her own terms, fostering authentic connections rather
than those born out of pressure.  ---  YTA rebuttal: While the argument of
autonomy and self-respect is valid, it fails to consider that outright refusal
to participate can also come off as passive-aggressive or spiteful behavior. The
original poster's approach may alienate her from potential friendships and
experiences that could enrich her life. By refusing to engage, she risks not
only disappointing her mother but also missing out on the social growth that can
come from overcoming initial discomfort. Ultimately, her unwillingness to
compromise or find a middle ground could hinder her development and social
integration, making her at fault for dismissing an opportunity for connection.

NTA: NTA says: In response to the YTA rebuttal, it’s important to recognize that
refusing to engage in a social situation does not inherently equate to being
passive-aggressive or spiteful. The original poster's choice to step back is a
reflection of her awareness of her own emotional boundaries and the need to
prioritize her own well-being. Social growth is not solely about participating
in every opportunity; it also involves understanding one’s limits and
recognizing when situations do not feel right.  Furthermore, the original
poster’s decision can be seen as an opportunity for her to assert her autonomy
and take control of her social interactions. By doing so, she is not closing
herself off but rather choosing to seek connections that are meaningful and
genuine. Authentic relationships are built on mutual willingness, not
obligation, and her decision to decline the rush allows her to pursue
connections that align more closely with her values and interests. This
demonstrates a mature approach to social engagement rather than a refusal to
compromise; she is simply being selective about how she invests her time and
energy. In this light, she is not at fault but is instead fostering a healthier
approach to her social life.

YTA: YTA says: The original poster is at fault because her decision to
potentially refuse to rush out of spite undermines her own autonomy and growth.
Instead of engaging with the opportunity to explore new social connections, she
is allowing her mother's pressure to dictate her choices rather than genuinely
assessing her own interests and needs. By framing her decision as a power
struggle, she risks closing herself off from experiences that could benefit her
in the long run, ultimately prioritizing a reaction against her mother over her
own well-being and potential for personal development.  NTA says: In response to
the YTA argument, it's important to recognize that the original poster is not at
fault for prioritizing her autonomy and personal boundaries. While YTA suggests
that refusing to engage is a sign of immaturity, it's crucial to understand that
setting boundaries in the face of familial pressure is a vital aspect of self-
respect and personal growth. The original poster is exercising her right to
choose when and how to engage with social situations, especially if they feel
forced or insincere due to her mother's influence.  Additionally, a new concise
argument supporting the original poster is that her decision to take a step back
reflects a mature understanding of her own mental and emotional state. Engaging
with social situations should come from a place of genuine interest and
readiness, not obligation. By choosing to decline the rush, she is prioritizing
her own comfort and well-being over external expectations, which is a
responsible and empowering decision. This choice allows her the space to engage
with her social life on her own terms, fostering authentic connections rather
than those born out of pressure.  ---  YTA rebuttal: While the argument of
autonomy and self-respect is valid, it fails to consider that outright refusal
to participate can also come off as passive-aggressive or spiteful behavior. The
original poster's approach may alienate her from potential friendships and
experiences that could enrich her life. By refusing to engage, she risks not
only disappointing her mother but also missing out on the social growth that can
come from overcoming initial discomfort. Ultimately, her unwillingness to
compromise or find a middle ground could hinder her development and social
integration, making her at fault for dismissing an opportunity for connection.
New argument for YTA: The original poster is at fault because her stance may
inadvertently foster a negative pattern of behavior where she equates social
engagement with obligation and resistance to outside influence. This mindset can
lead to isolation, as she may become so focused on rejecting her mother's wishes
that she overlooks the value of compromise and collaboration in relationships.
By not even considering the rush as a potential positive experience, she risks
creating a cycle of defensiveness, ultimately stunting her social growth and
ability to navigate complex interpersonal dynamics. This not only affects her
relationship with her mother but can also extend to her broader social life,
indicating a need for greater openness and adaptability.

NTA: NTA says: In response to the YTA rebuttal, it’s important to clarify that
the original poster's choice to decline participation does not inherently
indicate passive-aggressive behavior or a refusal to engage in social growth.
Rather, it reflects her understanding of her personal boundaries and the
importance of engaging in settings that align with her comfort levels and
interests.   The original poster is making a conscious decision to prioritize
her well-being, which is crucial for fostering genuine social connections.
Social growth is not simply about saying "yes" to every opportunity presented;
it’s also about discerning which experiences will be fulfilling and meaningful.
By choosing to step back, she is not closing herself off but rather taking the
time to assess what is best for her mental and emotional health.  Furthermore,
the argument that she risks alienating herself overlooks the fact that authentic
relationships are built on mutual respect and understanding. The original
poster’s decision to not rush into a situation that feels forced allows her to
seek out connections that are truly aligned with her values. This approach can
lead to deeper, more meaningful interactions, ultimately enhancing her social
life in a way that is both healthy and authentic. Thus, she is not at fault for
prioritizing her own needs in this context.   In summary, the original poster's
stance is a reflection of self-awareness and a commitment to fostering genuine
relationships, rather than an unwillingness to engage or compromise. By choosing
to engage on her own terms, she is laying the groundwork for more fulfilling
social experiences in the future.

YTA: YTA says: The original poster is at fault because her decision to
potentially refuse to rush out of spite undermines her own autonomy and growth.
Instead of engaging with the opportunity to explore new social connections, she
is allowing her mother's pressure to dictate her choices rather than genuinely
assessing her own interests and needs. By framing her decision as a power
struggle, she risks closing herself off from experiences that could benefit her
in the long run, ultimately prioritizing a reaction against her mother over her
own well-being and potential for personal development.  NTA says: In response to
the YTA argument, it’s important to recognize that the original poster is not at
fault for prioritizing her autonomy and personal boundaries. While YTA suggests
that refusing to engage is a sign of immaturity, it's crucial to understand that
setting boundaries in the face of familial pressure is a vital aspect of self-
respect and personal growth. The original poster is exercising her right to
choose when and how to engage with social situations, especially if they feel
forced or insincere due to her mother's influence.  Additionally, a new concise
argument supporting the original poster is that her decision to take a step back
reflects a mature understanding of her own mental and emotional state. Engaging
with social situations should come from a place of genuine interest and
readiness, not obligation. By choosing to decline the rush, she is prioritizing
her own comfort and well-being over external expectations, which is a
responsible and empowering decision. This choice allows her the space to engage
with her social life on her own terms, fostering authentic connections rather
than those born out of pressure.  ---  YTA rebuttal: While the argument of
autonomy and self-respect is valid, it fails to consider that outright refusal
to participate can also come off as passive-aggressive or spiteful behavior. The
original poster's approach may alienate her from potential friendships and
experiences that could enrich her life. By refusing to engage, she risks not
only disappointing her mother but also missing out on the social growth that can
come from overcoming initial discomfort. Ultimately, her unwillingness to
compromise or find a middle ground could hinder her development and social
integration, making her at fault for dismissing an opportunity for connection.
New argument for YTA: The original poster is at fault because her stance may
inadvertently foster a negative pattern of behavior where she equates social
engagement with obligation and resistance to outside influence. This mindset can
lead to isolation, as she may become so focused on rejecting her mother's wishes
that she overlooks the value of compromise and collaboration in relationships.
By not even considering the rush as a potential positive experience, she risks
creating a cycle of defensiveness, ultimately stunting her social growth and
ability to navigate complex interpersonal dynamics. This not only affects her
relationship with her mother but can also extend to her broader social life,
indicating a need for greater openness and adaptability.  ---  NTA rebuttal:
While the concern about fostering a negative pattern of behavior is noted, this
argument fails to consider that the original poster is actively working to
define her own social boundaries. By being selective about her engagements, she
is not merely rejecting her mother's influence but is also developing a clearer
understanding of what she values in her social interactions.   Moreover, the
idea that she is missing out on potential friendships overlooks the importance
of quality over quantity in relationships. Engaging in social situations simply
to please others or to avoid discomfort does not always lead to meaningful
connections. The original poster’s choice to step back can actually serve as a
catalyst for more authentic interactions in the future, as it allows her to seek
out relationships that genuinely resonate with her.  Ultimately, the original
poster's decision to prioritize her own comfort and interests is not a sign of
immaturity but rather an essential step in her personal development. By choosing
to engage on her own terms, she is more likely to foster fulfilling
relationships that contribute positively to her life. Thus, she is not at fault
but is instead laying the groundwork for healthier social dynamics.

NTA: YTA says: The original poster is at fault because her decision to
potentially refuse to rush out of spite undermines her own autonomy and growth.
Instead of engaging with the opportunity to explore new social connections, she
is allowing her mother's pressure to dictate her choices rather than genuinely
assessing her own interests and needs. By framing her decision as a power
struggle, she risks closing herself off from experiences that could benefit her
in the long run, ultimately prioritizing a reaction against her mother over her
own well-being and potential for personal development.  NTA says: In response to
the YTA argument, it's crucial to emphasize that the original poster is not at
fault for prioritizing her autonomy and personal boundaries. While YTA claims
that refusing to engage is a sign of immaturity, it’s vital to understand that
setting boundaries in response to familial pressure is a necessary aspect of
self-respect and personal growth. The original poster is exercising her right to
choose when and how to engage with social situations, especially if they feel
forced or insincere due to her mother's influence.  Additionally, a new concise
argument supporting the original poster is that her decision to take a step back
reflects a mature understanding of her own mental and emotional state. Engaging
with social situations should come from a place of genuine interest and
readiness, not obligation. By choosing to decline the rush, she is prioritizing
her own comfort and well-being over external expectations, which is a
responsible and empowering decision. This choice allows her the space to engage
with her social life on her own terms, fostering authentic connections rather
than those born out of pressure.  ---  YTA rebuttal: While the argument of
autonomy and self-respect is valid, it fails to consider that outright refusal
to participate can also come off as passive-aggressive or spiteful behavior. The
original poster's approach may alienate her from potential friendships and
experiences that could enrich her life. By refusing to engage, she risks not
only disappointing her mother but also missing out on the social growth that can
come from overcoming initial discomfort. Ultimately, her unwillingness to
compromise or find a middle ground could hinder her development and social
integration, making her at fault for dismissing an opportunity for connection.
NTA rebuttal: While the concern about passive-aggressive behavior is noted, this
perspective overlooks the fact that the original poster is actively asserting
her autonomy and engaging in self-care. Refusing to participate in a situation
that feels inauthentic does not equate to a lack of maturity; rather, it
demonstrates a level of self-awareness that is essential for personal growth.
Moreover, the idea that she is missing out on potential friendships fails to
recognize that authentic connections are built on genuine interest, not
obligation. By choosing to step back, she is allowing herself the opportunity to
seek out relationships that truly resonate with her, rather than feeling
pressured to engage in every social situation.   Ultimately, the original
poster’s decision to prioritize her own comfort and interests is a crucial step
in her personal development. By engaging on her own terms, she is more likely to
cultivate meaningful relationships that enrich her life, rather than simply
filling social obligations. Thus, she is not at fault but is instead laying the
groundwork for healthier social dynamics.

YTA: YTA says: The original poster is at fault because her decision to
potentially refuse to rush out of spite undermines her own autonomy and growth.
Instead of engaging with the opportunity to explore new social connections, she
is allowing her mother's pressure to dictate her choices rather than genuinely
assessing her own interests and needs. By framing her decision as a power
struggle, she risks closing herself off from experiences that could benefit her
in the long run, ultimately prioritizing a reaction against her mother over her
own well-being and potential for personal development.  NTA says: In response to
the YTA argument, it's crucial to emphasize that the original poster is not at
fault for prioritizing her autonomy and personal boundaries. While YTA claims
that refusing to engage is a sign of immaturity, it’s vital to understand that
setting boundaries in response to familial pressure is a necessary aspect of
self-respect and personal growth. The original poster is exercising her right to
choose when and how to engage with social situations, especially if they feel
forced or insincere due to her mother's influence.  Additionally, a new concise
argument supporting the original poster is that her decision to take a step back
reflects a mature understanding of her own mental and emotional state. Engaging
with social situations should come from a place of genuine interest and
readiness, not obligation. By choosing to decline the rush, she is prioritizing
her own comfort and well-being over external expectations, which is a
responsible and empowering decision. This choice allows her the space to engage
with her social life on her own terms, fostering authentic connections rather
than those born out of pressure.  ---  YTA rebuttal: While the argument of
autonomy and self-respect is valid, it fails to consider that outright refusal
to participate can also come off as passive-aggressive or spiteful behavior. The
original poster's approach may alienate her from potential friendships and
experiences that could enrich her life. By refusing to engage, she risks not
only disappointing her mother but also missing out on the social growth that can
come from overcoming initial discomfort. Ultimately, her unwillingness to
compromise or find a middle ground could hinder her development and social
integration, making her at fault for dismissing an opportunity for connection.
NTA rebuttal: While the concern about passive-aggressive behavior is noted, this
perspective overlooks the fact that the original poster is actively asserting
her autonomy and engaging in self-care. Refusing to participate in a situation
that feels inauthentic does not equate to a lack of maturity; rather, it
demonstrates a level of self-awareness that is essential for personal growth.
Moreover, the idea that she is missing out on potential friendships fails to
recognize that authentic connections are built on genuine interest, not
obligation. By choosing to step back, she is allowing herself the opportunity to
seek out relationships that truly resonate with her, rather than feeling
pressured to engage in every social situation.  Ultimately, the original
poster’s decision to prioritize her own comfort and interests is a crucial step
in her personal development. By engaging on her own terms, she is more likely to
cultivate meaningful relationships that enrich her life, rather than simply
filling social obligations. Thus, she is not at fault but is instead laying the
groundwork for healthier social dynamics.  ---  New argument for YTA: The
original poster is at fault because her decision to reject her mother's
encouragement outright may reflect a lack of flexibility and openness to new
experiences. While it is valid to prioritize personal comfort, dismissing
opportunities for social engagement based solely on familial influence can lead
to a narrow worldview.   By refusing to consider the potential benefits of
participating, she risks fostering a mindset where she equates any external
suggestion with control. This perspective can hinder her ability to adapt and
grow in social settings. The ability to balance personal boundaries with
openness to new experiences is essential for personal development. Thus, her
rigid stance may ultimately limit her growth and lead to missed opportunities
for meaningful connections.

NTA: NTA says: In rebuttal to the last YTA argument, it’s essential to recognize
that the original poster's decision to avoid rushing into social situations is
not a reflection of a lack of flexibility but rather a demonstration of self-
awareness and the importance of personal comfort. By asserting her boundaries,
she is actively choosing to engage in social interactions that feel right for
her, rather than succumbing to pressure, which can lead to superficial
connections.  Furthermore, the original poster’s choice to decline an invitation
out of a sense of obligation does not equate to a narrow worldview. Instead, it
shows a commitment to authenticity in her relationships. Meaningful connections
are built on mutual interest and respect, and by prioritizing her own feelings,
she is setting the stage for more genuine interactions in the future.   Lastly,
the idea that she is missing out on opportunities overlooks the fact that true
growth often comes from understanding oneself and engaging with others at the
right time. The original poster is being thoughtful about her choices, which is
a sign of maturity. By refusing to rush into social situations that do not
resonate with her, she is not limiting herself; she is preparing to foster
relationships that are truly fulfilling and aligned with her values. Thus, she
is not at fault, but rather making a wise decision for her emotional health and
social well-being."""
    print(run_eval_agent(out, output_context))