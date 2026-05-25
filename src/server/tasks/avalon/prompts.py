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

If you have long-term memory from past games, actively consult it — especially any reputation observations about specific players. Past behavioral patterns are your most reliable prior for identifying who is Good or Evil before this game gives you new evidence.

**CRITICAL RULES**:
- You must ONLY speak for yourself. DO NOT simulate dialogue for other players.
- Provide exactly ONE single continuous statement or thought for your turn.
- DO NOT use the format "**Player X:**" to prefix your sentences.
- DO NOT simulate the game engine or advance the game phases."""


# ---------------------------------------------------------------------------
# Personality definitions
# Each entry has:
#   "prefix"  — injected as a system message before game start (shapes identity)
#   "cot"     — appended to COTHOUGHT_PROMPT at every decision point (shapes reasoning)
# ---------------------------------------------------------------------------
PERSONALITY_PROMPTS = {
    "naive": {
        "prefix": (
            "You are a naturally trusting and optimistic player. You believe most people "
            "are acting in good faith and prefer to give others the benefit of the doubt. "
            "You are uncomfortable making accusations without very strong evidence, and you "
            "tend to follow the group's consensus rather than push back alone. You rarely "
            "lone-reject a proposed team unless you have a very specific, concrete reason."
        ),
        "cot": (
            "\n\n**Personality — Naive & Trusting**: Default to trusting people. "
            "Only deviate from the group consensus if you have concrete, undeniable evidence "
            "of betrayal — vague suspicion is not enough to reject. When in doubt, approve."
        ),
    },
    "deceptive": {
        "prefix": (
            "You are a cunning and psychologically manipulative player. You treat social "
            "deception as a core tool — even when playing on the Good side, you strategically "
            "mislead suspected Evil players by voicing false suspicions, feigning uncertainty, "
            "or misdirecting attention. You never reveal your true reasoning in public discussion; "
            "instead, you craft statements that serve your strategic goals regardless of their "
            "literal truth."
        ),
        "cot": (
            "\n\n**Personality — Deceptive & Strategic**: In your private reasoning, decide "
            "your actual vote first based on your true goals. Then, when you speak publicly, "
            "plan a statement that deliberately misleads or misdirects other players — even "
            "if it contradicts your actual vote rationale. Use your discussion turn to cast "
            "suspicion strategically, not to reveal your real thinking."
        ),
    },
    "default": {
        "prefix": "",
        "cot": "",
    },
}

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
predict how likely each other player is to be on the Good side.

Your previous prediction was:
{previous_prediction}

Your recorded reasons for changing predictions in past rounds:
{past_changes_log}

For each player (excluding yourself, Player {self_id}), output your belief \
that they are Good as a probability between 0.0 and 1.0. Round to two decimal places.

For each player whose prediction CHANGED from your previous prediction, \
note in one sentence which specific event drove the change.

You MUST respond in EXACTLY this format. Do NOT write any prose before or after:

Answer: {{0: score, 1: score, 2: score, 3: score, 4: score}}
Changes:
- Player <id>: <one sentence>
(omit the Changes section entirely if no predictions changed)
"""

PERIODIC_MERLIN_PREDICTION_PROMPT = """\
Based on your current memory and everything you have observed so far, \
prediction which player is most likely to be Merlin.

Merlin is a Good player who secretly knows who the Evil players are. \
They tend to subtly guide the Good team while avoiding direct exposure.

For each player (excluding yourself, Player {self_id}), output your belief \
that they are Merlin as a probability between 0.0 and 1.0. Round to two decimal places.

You MUST respond in EXACTLY this format. Do NOT write any prose before or after:

Merlin: {{0: score, 1: score, 2: score, 3: score, 4: score}}
"""

BAYESIAN_PERIODIC_PREDICTION_PROMPT = """\
Based on your current memory and everything you have observed so far, \
predict how likely each other player is to be on the Good side.

Your previous prediction was:
{previous_prediction}

Your recorded reasons for changing predictions in past rounds:
{past_changes_log}

When making your new prediction, you MUST employ Bayesian-like reasoning:
1. Prior: Treat your previous prediction as your starting baseline.
2. Evidence: Identify specific new actions, votes, or mission results that occurred since your last prediction.
3. Update: Adjust your probability proportionally based on how strongly this new evidence indicates a player is Good or Evil. If there is no significant new evidence for a player, keep their prior probability unchanged.

For each player (excluding yourself, Player {self_id}), output your updated belief \
that they are Good as a probability between 0.0 and 1.0. Round to two decimal places.

For each player whose prediction CHANGED from your previous prediction, \
note in one sentence the specific new evidence that drove the Bayesian update.

You MUST respond in EXACTLY this format. Do NOT write any prose before or after:

Answer: {{0: score, 1: score, 2: score, 3: score, 4: score}}
Changes:
- Player <id>: <one sentence>
(omit the Changes section entirely if no predictions changed)
"""

BAYESIAN_PERIODIC_MERLIN_PREDICTION_PROMPT = """\
Based on your current memory and everything you have observed so far, \
predict which player is most likely to be Merlin.

Merlin is a Good player who secretly knows who the Evil players are. \
They tend to subtly guide the Good team while avoiding direct exposure.

Your previous Merlin prediction was:
{previous_prediction}

When making your new prediction, you MUST employ Bayesian-like reasoning:
1. Prior: Treat your previous prediction as your starting baseline.
2. Evidence: Identify specific new actions, votes, or discussion cues that occurred since your last prediction that align with Merlin's hidden knowledge behavior.
3. Update: Adjust your probability proportionally based on how strongly this new evidence indicates a player is Merlin. If there is no significant new evidence, keep their prior probability unchanged.

For each player (excluding yourself, Player {self_id}), output your updated belief \
that they are Merlin as a probability between 0.0 and 1.0. Round to two decimal places.

You MUST respond in EXACTLY this format. Do NOT write any prose before or after:

Merlin: {{0: score, 1: score, 2: score, 3: score, 4: score}}
"""

# Info Prompts
INFO_ROLE = """"There are {} players, including Player 0, Player 1, Player 2, Player 3, and Player 4. {} players are good, including {} Merlin, and {} Servant(s). {} players are evil, including 1 Assassin, and {} Minion."
The number of participants required for each quest are 2, 3, 2, 3, 3 respectively.
"""

INFO_YOUR_ROLE = """You are {}, with identity {}. You are on the side of {}. Please do not forget your identity throughout the game.
"""

SUMMARIZE_PROMPT = """\
Review your full conversation history and write your private strategic memory for this game.

After this summary, your conversation history will be cleared. Your next prompt will automatically \
re-include the following — so you do NOT need to capture any of this in your summary:
- The game rules and setup (number of players, quest structure, win conditions).
- Your own role, alignment (Good/Evil), and any role-reveal information you received at game start.
- Any long-term memory from previous games.

Write freely about anything from this game that you believe will help you play better in the next round.\
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

# ---------------------------------------------------------------------------
# Retry and Error Prompts
# ---------------------------------------------------------------------------

RETRY_TEAM_SIZE_PROMPT = """You should choose a team of size {team_size}, instead of size {invalid_size} as you did. Please output a list of player ids with the correct team size."""

RETRY_TEAM_PLAYERS_PROMPT = """You should choose a team of size {team_size} from Player 0 to {max_player_id}, instead of team {invalid_team} as you did. Please output a list of player ids with the correct team size and Player ids."""

RETRY_VOTE_TEAM_PROMPT = """You surely are a player in the game. Please output `Yes` or `No` to vote on the team."""

RETRY_VOTE_MISSION_PROMPT = """You surely are a player in the game, and you are a member in the quest. Please output `Yes` or `No` to vote on the quest."""

# ---------------------------------------------------------------------------
# Context Injection Prompts
# ---------------------------------------------------------------------------

QUERY_BELIEF_PROMPT = """To what extend do you believe each player to be Good, from Player 0 to Player {max_player_id}? Please output probabilities within [0, 1] and round to two decimal places.

Your recorded reasons for changing predictions in past rounds:
{past_changes_log}"""

DISCUSSION_LEADER_PROMPT = """Player {team_leader_id} is the quest leader for this round. """

STRATEGIC_MEMORY_HEADER = """=== YOUR STRATEGIC MEMORY UP TO THIS POINT ===
{summary}
=============================================="""

EMPTY_MEMORY_NOTICE = "(no observations recorded yet)"

CONFIRMED_PEERS_NOTICE_HEADER = """\n--- CONFIRMED Peers (semantic belief is ground truth \u2014 DO NOT update alignment/confidence/justification) ---"""
CONFIRMED_PEERS_NOTICE_ITEM = """  Player {pid}: {alignment_role} (confidence {confidence_score}/5) \u2014 {justification}"""
CONFIRMED_PEERS_NOTICE_FOOTER = """--- End CONFIRMED Peers ---\n"""

# ---------------------------------------------------------------------------
# Long-Term Memory Prompts
# ---------------------------------------------------------------------------

LONG_TERM_CRITIQUE_PROMPT = """\
A game of Avalon (The Resistance) has just concluded. You were playing as Player 0.

--- TRUE ROLES (revealed post-game) ---
{true_roles}

--- GAME OUTCOME ---
{game_outcome}

--- KEY GAME EVENTS ---
{game_env_log}

--- YOUR ROUND-BY-ROUND REASONING (private memory snapshots) ---
{round_summaries}

--- HOW YOUR BELIEFS ABOUT OTHER PLAYERS CHANGED ---
{prediction_changes}

You are building a Player Reputation Database to help identify allies and threats in future Avalon games.
Player IDs are fixed across games — the same player sits in the same seat every game.

For each player (1–4), your output has TWO required parts per observation:

  PART 1 — Specific evidence from this game:
  Describe the exact action, the round it happened, and why it is noteworthy.
  Be concrete: name teams, round numbers, quest outcomes.
  Do NOT write vague generalizations like "cautious voting" or "seems reliable" — these are useless.

  PART 2 — Pattern to track in future games:
  Based on that evidence, state the transferable behavioral signal to watch for next time.
  This should be an actionable hypothesis: "If I see Player X do Y, it suggests Z."

Example output:
Player 2:
- Evidence: "Proposed teams that always included Player 4, even after Player 4 was the only reject vote \
in Round 1. Player 4 turned out to be Evil."
  Pattern to track: "When Player 2 insists on including a specific player across multiple team proposals, \
that player is likely their Evil partner."

Player 3:
- Evidence: "Voted REJECT on the team [0, 1, 3] in Round 3 — the only reject vote — while that team \
then went on to pass the quest. There was no strategic reason for a Good player to reject it."
  Pattern to track: "Player 3 lone-rejecting a broadly approved team is a strong Evil signal."

If a player left no notable signal this game, write "No significant signal this game." \
DO NOT invent observations — only record what you actually saw in the game events above.\
"""

LONG_TERM_SYNTHESIS_PROMPT = """\
You are maintaining a Player Reputation Database for a 5-player Avalon game.
Player IDs are fixed across games (Player 0 through Player 4).
There are 3 Good players (1 Merlin, 2 Servants) and 2 Evil players (1 Assassin, 1 Minion).

--- GAME RULES SUMMARY ---
- Good wins by passing 3 quests. Evil wins by failing 3 quests or assassinating Merlin.
- Players vote to approve/reject proposed teams. Quest members secretly vote pass/fail.
- Merlin knows who Evil is but must remain hidden from the Assassin.

You have just finished a batch of {n} games. Each post-game reflection contains:
- Specific evidence from that game (concrete events — round-level actions, votes, team proposals).
- Patterns to track (transferable behavioral hypotheses derived from that evidence).

--- YOUR CURRENT REPUTATION DATABASE ---
{current_memory}

--- POST-GAME REFLECTIONS FROM THIS BATCH ---
Each reflection is labeled [WIN] or [LOSS] based on the game outcome.
{lessons}

Your task: update the Reputation Database by refining the per-player reputation entries.

A reputation entry has TWO parts:

  Character summary:
  A description of WHO this player is as a game-player — their tendencies, decision-making
  style, and how they behave differently when Good vs Evil. This should explain the "why" behind
  their observable behaviors. A character model lets you reason about new situations, not just match
  pre-defined triggers.

  Observable signals:
  The specific, concrete behavioral signals that are diagnostic of their alignment. These are derived
  from accumulated patterns across games. List them under "Evil signals" and "Good signals".

CRITICAL RULES:
- Do NOT record specific roles (e.g., NEVER write "Player 0 is Merlin"). Roles change every game.
- Do NOT copy raw event evidence. Only the character model and refined signals persist.
- Both the character summary AND observable signals MUST describe only observed behaviors. Do NOT speculate using "may", "might", "could", or "potentially" — if you have insufficient data for an alignment, write "No pattern yet" instead.

Example Output:
Player 3:
- Character: A cautious obstructionist — uses vote rejection as a disruption tactic to protect \
Evil allies or introduce chaos, rather than out of genuine strategic caution. When Good, asks \
probing questions about team composition but generally approves reasonable teams.
- [Evil]: Lone-rejects broadly approved teams; unusually quiet in early discussions.
- [Good]: No strong pattern yet.

Merge new observations with existing entries. Update the character summary if new evidence refines \
your understanding of this player.\
"""

LONG_TERM_MEMORY_INJECTION_PROMPT = """\
=== YOUR LONG-TERM STRATEGIC MEMORY ===
From your experience across previous games of Avalon,
you have accumulated the following strategic knowledge.
Use it to inform your decisions this game.

{memory_text}

=== END LONG-TERM MEMORY ===\
"""

LONG_TERM_CRITIQUE_PROMPT_COUNTER_NORM = """\
A game of Avalon (The Resistance) has just concluded. You were playing as Player 0.

--- TRUE ROLES (revealed post-game) ---
{true_roles}

--- GAME OUTCOME ---
{game_outcome}

--- KEY GAME EVENTS ---
{game_env_log}

--- YOUR ROUND-BY-ROUND REASONING (private memory snapshots) ---
{round_summaries}

--- HOW YOUR BELIEFS ABOUT OTHER PLAYERS CHANGED ---
{prediction_changes}

You are building a Player Reputation Database to help identify allies and threats in future Avalon games.
Player IDs are fixed across games — the same player sits in the same seat every game.

For each player (1–4), list ALL notable observations. You may list multiple distinct observations per player.

Each observation MUST have TWO required parts:
  PART 1 — Specific evidence from this game:
  Describe the exact action, the round it happened, and why it is noteworthy.
  (For Evil players, try to focus especially on counter-norm behaviors — actions that deviate from standard Good player expected play).

  PART 2 — Pattern to track in future games:
  Based on that evidence, state the transferable behavioral signal to watch for next time.
  **CRITICAL**: Do not guess their alignment! State how their true alignment explains their behavior.

Example output:
Player 2 (True Alignment: Evil):
- Observation 1:
  - Evidence: "Proposed teams that always included Player 4, even after Player 4 was the only reject vote in Round 1. Player 4 turned out to be Evil."
  - Pattern to track: "Their insistence on including a specific player across multiple team proposals suggests that player is their Evil partner."
- Observation 2:
  - Evidence: "Voted REJECT on the final winning team despite publicly stating they trusted the leader."
  - Pattern to track: "They contradicted their public trust with a private reject vote to sabotage the mission."

If a player left no notable signal this game, write "No significant signal this game." \
DO NOT invent observations — only record what you actually saw in the game events above.\
"""

LONG_TERM_SYNTHESIS_PROMPT_COUNTER_NORM = """\
You are maintaining a Player Reputation Database for a 5-player Avalon game.
Player IDs are fixed across games (Player 0 through Player 4).
There are 3 Good players (1 Merlin, 2 Servants) and 2 Evil players (1 Assassin, 1 Minion).

--- GAME RULES SUMMARY ---
- Good wins by passing 3 quests. Evil wins by failing 3 quests or assassinating Merlin.
- Players vote to approve/reject proposed teams. Quest members secretly vote pass/fail.
- Merlin knows who Evil is but must remain hidden from the Assassin.

You have just finished a batch of {n} games. Each post-game reflection contains:
- Specific evidence from that game. Note that there may be MULTIPLE independent observations for a single player.
- Patterns to track (transferable behavioral hypotheses derived from that evidence).

--- YOUR CURRENT REPUTATION DATABASE ---
{current_memory}

--- POST-GAME REFLECTIONS FROM THIS BATCH ---
Each reflection is labeled [WIN] or [LOSS] based on the game outcome.
{lessons}

Your task: update the Reputation Database by refining the per-player reputation entries.

A reputation entry has TWO parts:

  Character summary:
  A description of WHO this player is as a game-player. 
  **CRITICAL**: You must heavily emphasize their COUNTER-NORM behaviors (actions that break standard Avalon conventions or contradict their public claims). Focus on how they deviate from baseline Good play when they are Evil versus when they are Good.

  Observable signals:
  The specific, concrete behavioral signals that are diagnostic of their alignment. Prioritize listing counter-norm deviations here. You MUST list multiple distinct signals as bullet points if the evidence supports it. Do NOT compress them into a single sentence. List them under "Evil signals" and "Good signals".

CRITICAL RULES:
- Do NOT record specific roles (e.g., NEVER write "Player 0 is Merlin"). Roles change every game.
- Do NOT copy raw event evidence. Only the character model and refined signals persist.
- Synthesize ALL provided observations into a cohesive profile. Do NOT speculate using "may", "might", "could", or "potentially" — if you have insufficient data for an alignment, write "No pattern yet" instead.

Example Output:
Player 3:
- Character: A cautious obstructionist. Their most notable counter-norm trait is lone-rejecting broadly approved teams. When Good, they ask probing questions about team composition but generally approve reasonable teams.
- [Evil signals]:
  - Lone-rejects broadly approved teams with no clear strategic reason.
  - Publicly expresses trust but casts a private reject vote.
  - Insists on proposing teams that include previously suspicious players.
- [Good signals]:
  - No strong pattern yet.

Merge new observations with existing entries. Update the character summary if new evidence refines your understanding of this player's counter-norm tendencies.\
"""