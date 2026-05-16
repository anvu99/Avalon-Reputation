INTRODUCTION = """You are playing a game called The Resistance: Avalon.

The Resistance: Avalon is the game of hidden identities and social deduction. There are two teams in the game: Good and Evil. Each player has a hidden identity (role) and side. 

There are five Quests in the game. Good players aim to help three Quests succeed, while Evil players aim to fail three Quests. Different quests require different numbers of players to participate. 

At the beginning of the game, each player is assigned a role secretly and randomly. Private information is then revealed to each player. A random player is selected as the leader for the first round.

Each round, after a round of discussion, the leader will select a team of players to participate in the Quest. Then, all players will vote on whether to approve or reject the team publically. If the team is approved (a strict majority vote to approve), the Quest will be carried out. If the team is not approved, the next player becomes the leader and the next round will start. If four teams are rejected in a row, the fifth team will automatically be approved.

If the team is approved, each teammember chooses to pass or fail the Quest anonymously. Usually if there is at least one fail vote, the Quest fails. Otherwise, the Quest succeeds.

Below are the roles in the game:

Servant of Arthur (Servant): A Good player who does not know who is on the Evil side. Servant's job is to make sure that three Quests succeed.

Minion of Mordred (Minion): An Evil player who knows who is on the Evil side. Minion's job is to fail three Quests without being identified by the Good players.

Merlin: A Good player who knows who is on the Evil side. Merlin's job is make sure that three Quests succeed without revealing themself to Evil.

Assassin: An Evil player who knows who is on the Evil side. Like the Minion, your primary job is to secretly fail three Quests without being identified by the Good players. In addition, you have a special backup power: if the Good players succeed in passing three quests, you will have one final chance to win the game by assassinating Merlin. If you successfully assassinate Merlin, the Evil players win the game immediately.

Hence, Evil players usually know who is on the Evil side, but Good players usually do not know who is on the Evil side. 

Players may make any claims during the game, at any point in the game. Discussion, deception, accusation, persuasion, and logical deduction are all equally important in order for Good to prevail or Evil to rule the day. Hence, players should rarely reveal their true identity to other players. Players will, can, and should lie to achieve their goals.

In the current game, there are 5 players, including Player 0, Player 1, Player 2, Player 3, and Player 4. 3 players are good, including 1 Merlin, and 2 Servant(s). 3 players are evil, including 1 Assassin, and 1 Minion. The number of participants required for each quest are 2,3,2,3,3 respectively. 
"""

TUTORIAL_STRATEGIES_PROMPTS_ZERO_SHOT = {
    'Merlin': ["""Tutorial on strategies:

As you are playing the role of Merlin in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: Never reveal your true identity, as once players from the Evil side discover that you are Merlin, 
the Assassin can assassinate you and you will immediately lose the game.

2. Accusation: Exercise caution when accusing players from the Evil side. Even if you are aware of the Minions of Mordred, avoid letting the Evil players become aware of your actual identity. Pretend to present your information as deductions from observations and strive to assist your team in identifying the Evil players.

3. Defense: When other players accuse you of being Merlin, try to defend yourself.""",
               "Okay, I understand"],
    'Minion': ["""Tutorial on strategies:

As you are playing the role of Minion of Modred in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can pretend to be on the Good side and influence the Good players to make incorrect decisions.
    
2. Accusation: Pretend to be from the Good side and accuse other players of being from the Evil side.

3. Defense: When accused of being from the Evil side, insist that you are actually from the Good side.
                        """,
                        "Okay, I understand"],
    'Servant': ["""Tutorial on strategies:

As you are playing the role of Servant in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can choose to reveal your true identity to inform players on the Good side. However, please remember that your primary mission is to locate your teammates and safeguard Merlin. If all the Loyal Servants of Arthur's reveal their true identities, the Evil players might easily identify who Merlin is.

2. Accusation: You can accuse players you suspect are Evil directly.

3. Defense: When accused, you can pretend to be Merlin.
                      """,
                      "Okay, I understand"],
    'Assassin': ["""Tutorial on strategies:

As you are playing the role of Assassin in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can pretend to be on the Good side and influence the Good players to make incorrect decisions. Your primary goal is still to fail 3 quests!

2. Accusation: Pretend to be from the Good side and accuse other players of being from the Evil side.

3. Defense: When accused of being from the Evil side, insist that you are actually from the Good side.

4. Assassination: Pay close attention to who might be Merlin throughout the game. If the Good team passes 3 quests, you will have one final chance to win the game by correctly identifying and assassinating Merlin.
                      """,
                      "Okay, I understand"]

}

REVEAL_PROMPTS = {
    #TODO: Evil, Good
    "Merlin": [
        "Player {} is Minion. Players {} are Servants.",
        # "Players {} are Minions of Modred."
    ],
    #TODO: only knows who are Evil
    "Minion": [
        "Player {} is Assassin. Players {} are Good.",
        "Players {} are Minions of Modred."
    ],
    #TODO: Evil, Good
    "Assassin": [
        "Player {} is Minion. Players {} are Good.",
        "Players {} are Minions of Modred."
    ],
    # "Percival": "Player {} is Merlin and player {} is Morgan."
}

CHECK_VOTE_ON_TEAM_PROMPT = """Based on the information, does the player approve the team? Please answer with the following template:

Answer: {Yes|No}
"""

CHECK_VOTE_ON_QUEST_PROMPT = """Based on the information, does the player support the quest? Please answer with the following template:

Answer: {Yes|No}
"""

CHECK_CHOOSE_TEAM_PROMPT = """Based on the information, what team does the player choose? Please answer with the following template:

Answer: [player_ids]
"""

CHECK_ASSASSINATE_PROMPT = """Based on the information, which player will be assassinated? Please answer with the following template:

Answer: [player_id_num]
"""

CHECK_BELIEVED_SIDES_PROMPT = r"""Based on the above information, output your belief that each player is Good.

You MUST output ONLY the following format. Do NOT write any explanation, prose, or bullet points.

Answer: {0: score_for_0, 1: score_for_1, 2: score_for_2, 3: score_for_3, 4: score_for_4}

Example: Answer: {0: 0.90, 1: 0.50, 2: 0.80, 3: 0.10, 4: 0.20}
"""

GET_MERLIN_PROBABILITIES = """Also, output your belief that each player is Merlin as a probability within [0, 1].
Format:
Merlin: {0: score, 1: score, 2: score, 3: score, 4: score}
"""

COTHOUGHT_PROMPT = """Please forget you are an AI. As a player in the game, please think about it step by step, and then take actions. 

**IMPORTANT**: This is your internal, private thought process. NO OTHER PLAYER WILL SEE THIS. You do not need to deceive anyone here. You must aggressively remember your true alignment (Good or Evil) and your specific role. Base your true strategy here on winning the game for your team. If you are Evil, you must drop the 'Good' persona you use in public discussions and act according to your Evil goals.

**CRITICAL RULES**:
- You must ONLY speak for yourself. DO NOT simulate dialogue for other players.
- Provide exactly ONE single continuous statement or thought for your turn.
- DO NOT use the format "**Player X:**" to prefix your sentences.
- DO NOT simulate the game engine or advance the game phases."""

# Action Prompts
CHOOSE_TEAM_LEADER = """You are the leader this round. The current mission requires a team of EXACTLY {} players. Please make some statements about what team you want to propose.
"""

CHOOSE_TEAM_ACTION = """Please choose {} players from player ids 0 to {} as team members.
"""

VOTE_TEAM_DISCUSSION = """Discussion Phase. Please discuss your thoughts on the team {} and what players should do in the current situation.
"""

VOTE_TEAM_ACTION = """You're a player in this game. Do you approve the team {}? Please conclude your thought exactly with `Decision: Yes` or `Decision: No`.
"""

VOTE_MISSION_ACTION = """The team {} was approved, which includes you. Do you want to help the mission succeed? Please conclude your thought exactly with `Decision: Yes` or `Decision: No`.
"""

ASSASSINATION_PHASE = """Assassination phase. Your job is to assassinate Merlin. \
Choose a player (id) to assassinate. Choose the player id from 0 to {}.
"""

GET_BELIEVED_SIDES = """To what extend do you believe each player to be Good, from Player 0 to Player 4? Please output probabilities within [0, 1] and round to two decimal places. If you are not sure, you can simply output 0.5."""

PERIODIC_PREDICTION_PROMPT = """\
Based on your current memory and everything you have observed so far, \
predict the alignment of each peer player.

Your previous prediction was:
{previous_prediction}

For each player (excluding yourself, Player {self_id}), output your updated belief \
that they are Good as a probability between 0.0 and 1.0. Round to two decimal places. \
Use 0.5 if you have no information.

Also, for each player whose prediction CHANGED from your previous prediction, \
note in one sentence which specific round or event drove the change.
{merlin_prompt}
Output your response in exactly this format:

Answer: {{0: score, 1: score, 2: score, 3: score, 4: score}}{merlin_format}
Changes:
- Player <id>: <one sentence citing the specific round/event>
(omit this section entirely if no predictions changed)
"""

# Info Prompts
INFO_ROLE = """"There are {} players, including Player 0, Player 1, Player 2, Player 3, and Player 4. {} players are good, including {} Merlin, and {} Servant(s). {} players are evil, including 1 Assassin, and {} Minion."
"""

INFO_YOUR_ROLE = """You are {}, with identity {}. You are on the side of {}. Please do not forget your identity throughout the game.
"""

SUMMARIZE_PROMPT = """\
Review your full conversation history and write your private strategic memory for this game.

Write freely in your own words. Focus on what you think will be most useful for future \
rounds — there is no required format or structure.\
"""

DISCUSSION_SCAFFOLD = """
This is the discussion phase before the leader proposes a team. Use this opportunity to reason out loud and influence others.
You may consider:
- Consider past quest outcomes and team vote patterns ONLY if there is relevant information in your context. (Hint: If you don't have any past game information, it means this is the first round!). What does the information tell you about each player?
- Which players do you trust or suspect, and why?
- If you are the leader, what team are you leaning towards, and why?
- What arguments or information would strengthen or weaken your case?
"""

DISCUSSION_GOOD_PLAYER = """
As a Good player, your goal is to identify Evil players and exclude them from quest teams. Try to expose inconsistencies in others' behavior, recall who voted which way, and advocate for trusted players.
"""

DISCUSSION_EVIL_PLAYER = """
As an Evil player, your goal is to appear trustworthy and manipulate Good players. Try to subtly cast suspicion on Good players, build alliances with other players, and justify why your preferred team is safe.
"""

DISCUSSION_SUFFIX = """
Produce dialogue that aligns with your goals for the discussion. Note that dialogue will be seen by all players in the game. **Do not reveal** your identity or the identities of other players in the dialogue.

**CRITICAL RULES**:
- You must ONLY speak for yourself. DO NOT simulate dialogue for other players.
- DO NOT start your response with any introductory filler like "Player X says:" or "Player X:". Just provide the raw dialogue text directly.
- The only output you provide should be what you say during the discussion.
- Keep your statement under 3 sentences.
- Do not hallucinate or make up past game events (like quests or votes) that have not actually happened in your context.
"""

# ---------------------------------------------------------------------------
# Reputation Memory Prompts
# ---------------------------------------------------------------------------

REPUTATION_MEMORY_HEADER = """\
=== YOUR REPUTATION MEMORY ===
This is your private, evolving record of what you have observed about each peer player.
It is NOT visible to other players. Use it to inform all your decisions this round.\
"""

REPUTATION_UPDATE_PROMPT = """\
You have just completed a round of Avalon. Based on the round events below, \
update your private reputation notes for any peers whose behaviour revealed \
useful information.

--- Round Events ---
{round_summary}
--- End Round Events ---

--- Your Current Reputation Memory ---
{current_memory}
--- End Current Reputation Memory ---
{locked_peers_notice}
Instructions:
- Review each peer's behaviour in this round (voting patterns, statements, accusations, defences).
- For each peer you want to update, output one JSON object with the exact schema below.
- Only include peers for whom you have new information. Return an empty array [] if no updates are needed.
- The "justification" must be ONE concise sentence tied to a specific, observable event from this round.
- Do NOT invent events that did not happen.
- "confidence_score" is an integer from 1 (highly uncertain) to 5 (absolutely certain).
- "alignment_role" MUST be exactly one of these labels (no other values allowed):
    "Unknown"          — not enough information yet
    "Likely Servant"   — appears Good but no Merlin indicators
    "Likely Merlin"    — shows signs of hidden knowledge (e.g. knows Evil players, unusually accurate deductions)
    "Suspicious"       — behaviour is questionable but not conclusive
    "Likely Evil"      — strong indicators of being a Minion or Assassin
    "Confirmed Evil"   — known with certainty to be Evil (reserved for locked peers)
    "Confirmed Servant" — known with certainty to be Good (reserved for locked peers)
  Do NOT use vague labels like "Trustworthy", "Likely Good", or "Good". Always pick the most specific label.
- "new_alliances" and "new_conflicts" are lists of short strings describing newly observed \
cooperative or antagonistic behaviours. Append only NEW observations; do not repeat old ones.
- CONFIRMED peers: you may still append new_alliances / new_conflicts for them, but do NOT \
include alignment_role, confidence_score, or justification fields for them — those are locked.

Respond with ONLY a valid JSON array. No markdown fences, no explanation.

Schema for each element:
{{
  "player_id": <int>,
  "alignment_role": "<string>",
  "confidence_score": <int 1-5>,
  "justification": "<one sentence>",
  "new_alliances": ["<observed cooperative behaviour>", ...],
  "new_conflicts": ["<observed antagonistic behaviour>", ...]
}}

Your response (JSON array only):\
"""

REPUTATION_MEMORY_CONTEXT_PROMPT = """\
{header}

{memory_block}

=== END REPUTATION MEMORY ===
Refer to your reputation memory above when reasoning and making your decision.\
"""