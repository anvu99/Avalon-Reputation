import json
import re
from typing import List, Dict, Tuple, Optional
from .agent import Agent
from ..engine import AvalonBasicConfig
from ..wrapper import AvalonSessionWrapper, Session
from ..prompts import (
    INTRODUCTION, INFO_ROLE, INFO_YOUR_ROLE, REVEAL_PROMPTS,
    CHOOSE_TEAM_LEADER, CHOOSE_TEAM_ACTION, VOTE_TEAM_ACTION, VOTE_MISSION_ACTION,
    ASSASSINATION_PHASE, COTHOUGHT_PROMPT, DISCUSSION_SCAFFOLD,
    DISCUSSION_GOOD_PLAYER, DISCUSSION_EVIL_PLAYER, DISCUSSION_SUFFIX,
    REPUTATION_MEMORY_HEADER, REPUTATION_UPDATE_PROMPT, REPUTATION_MEMORY_CONTEXT_PROMPT,
    TUTORIAL_STRATEGIES_PROMPTS_ZERO_SHOT, SUMMARIZE_PROMPT, PERIODIC_PREDICTION_PROMPT,
)
from copy import deepcopy
from ..utils import verbalize_team_result, verbalize_mission_result
from src.utils import ColorMessage
from ..utils import get_game_logger
from ..reputation_memory import ReputationMemory
import logging

class LLMAgentWithDiscussion(Agent):
    r"""LLM agent with the ability to discuss with other agents."""

    def __init__(self, name: str, num_players: int, id: int, role: int, role_name: str, config:AvalonBasicConfig, session: AvalonSessionWrapper=None, side=None, seed=None, **kwargs):
        self.name = name
        self.id = id
        self.num_players = num_players
        self.role = role
        self.role_name = role_name
        self.side = side # 1 for good, 0 for evil
        self.session = session
        self.discussion = kwargs.pop('discussion', None)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.seed = seed

        self.config = config

        # Reputation memory — None unless activated for this agent by task.py
        self.reputation_memory: Optional[ReputationMemory] = None

        # Track previous round's prediction
        self.previous_prediction: dict = {}

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def see_sides(self, sides):
        self.player_sides = sides
    
    async def initialize_game_info(self, player_list, **kwargs) -> None:
        """Initiliaze the game info for the agent, which includes game introduction, role, and reveal information for different roles."""
        # Introduction Prompt
        verbal_side = ["Evil", "Good"]
        intro_prompt = INTRODUCTION
        intro_prompt += '\n'
        content_prompt = intro_prompt + INFO_ROLE.format(self.num_players, self.num_good, int(self.merlin), self.num_good - int(self.merlin) - int(self.percival), self.num_evil, self.num_evil - int(self.morgana) - int(self.mordred) - int(self.oberon) - 1)
        identity_prompt = INFO_YOUR_ROLE.format(self.name, self.role_name, verbal_side[self.side]) # and do not pretend to be other roles throughout the game."
        self.identity_prompt = identity_prompt

        # Reveal Prompt
        reveal_info = ''
        minion_list = []
        servant_list = []
        assassin = ''
        merlin = ''
        for idx, player_info in enumerate(player_list):
            if player_info[1] == "Minion":
                minion_list.append(str(idx))
            elif player_info[1] == "Servant":
                servant_list.append(str(idx))
            elif player_info[1] == "Assassin":
                assassin = str(idx)
            elif player_info[1] == "Merlin":
                merlin = str(idx)
        if self.role_name == "Merlin":
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Merlin'][0].format(', '.join(minion_list), ', '.join(servant_list))
            elif len(minion_list) > 1:
                reveal_info = REVEAL_PROMPTS['Merlin'][1].format(', '.join(minion_list))
        if self.role_name == "Minion":
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Minion'][0].format(assassin, ', '.join(servant_list + [merlin]))
            elif len(minion_list) > 1:
                reveal_info = REVEAL_PROMPTS['Minion'][1].format(', '.join(minion_list))
        if self.role_name == "Assassin":
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Assassin'][0].format(', '.join(minion_list), ', '.join(servant_list + [merlin]))

        # Seperately pass the reveal info to the agent, so as to meet the requirement in filer_messages
        # TODO: is `system` allowed? 
        self.session.inject({
            "role": "user",
            "content": content_prompt,
            "mode": "system",
        })
        self.session.inject({
            # "role": "system",
            "role": "user",
            "content": identity_prompt + '\n' + reveal_info,
            "mode": "system",
        })
        self.system_info = content_prompt + '\n' + identity_prompt + '\n' + reveal_info

        # Inject zero-shot tutorial strategies
        tutorial = TUTORIAL_STRATEGIES_PROMPTS_ZERO_SHOT.get(self.role_name)
        if tutorial:
            self.session.inject({
                "role": "user",
                "content": tutorial[0],
                "mode": "system",
            })
            self.system_info += '\n\n' + tutorial[0]

    async def summarize(self, round_num: int = 0, mission_id: int = 0, **kwargs) -> None:
        summary = await self.session.action({
            "role": "user",
            "content": SUMMARIZE_PROMPT,
            "mode": "summarize"
        })
        get_game_logger().info(f"##### Memory Snapshot (Mission {mission_id}, Round {round_num}) #####\n{summary}")
        self.session.overwrite_history([])
        self.session.inject({
            'role': "user",
            'content': self.system_info
        })
        self.session.inject({
            'role': "user",
            'content': f"=== YOUR STRATEGIC MEMORY UP TO THIS POINT ===\n{summary}\n=============================================="
        })
        return self.session.get_history()

    async def periodic_predict(self, round_num: int = 0, mission_id: int = 0, **kwargs) -> None:
        merlin_prompt = ""
        merlin_format = ""
        if self.role_name != "Merlin":
            merlin_prompt = "Also, output your belief that each player is Merlin as a probability within [0, 1].\n"
            merlin_format = "\nMerlin: {0: score, 1: score, 2: score, 3: score, 4: score}"
            
        prompt = PERIODIC_PREDICTION_PROMPT.format(
            previous_prediction=self.previous_prediction if self.previous_prediction else "{}",
            self_id=self.id,
            merlin_prompt=merlin_prompt,
            merlin_format=merlin_format
        )

        past_history = list(self.session.session.history)
        self.session.session.history = []

        self.session.session.inject({
            "role": "user",
            "content": prompt,
        })

        try:
            response = await self.session.session.action()
            raw = response.content if response.content else ""
        except Exception as e:
            get_game_logger().warning(f"[Periodic Prediction] LLM call failed: {e}")
            raw = ""
        finally:
            self.session.session.history = list(past_history)

        prediction_dict = {}
        changes_text = ""

        if "Answer:" in raw:
            answer_part = raw.split("Answer:")[1]
            if "Changes:" in answer_part:
                answer_part = answer_part.split("Changes:")[0]
            answer_part = answer_part.strip()
            try:
                dict_start = answer_part.find('{')
                dict_end = answer_part.rfind('}') + 1
                if dict_start != -1 and dict_end != -1:
                    dict_str = answer_part[dict_start:dict_end]
                    prediction_dict = eval(dict_str)
            except Exception as e:
                get_game_logger().warning(f"[Periodic Prediction] Failed to parse dict: {e}")

        if "Changes:" in raw:
            changes_text = raw.split("Changes:")[1].strip()

        if prediction_dict:
            self.previous_prediction = prediction_dict

        get_game_logger().info(f"##### Periodic Prediction (Mission {mission_id}, Round {round_num}) #####")
        get_game_logger().info(f"[Prediction] {self.previous_prediction}")
        if changes_text:
            get_game_logger().info(f"[Changes]\n{changes_text}")
    async def observe_mission(self, team, mission_id, num_fails, votes, outcome, **kwargs) -> None:
        pass

    # ------------------------------------------------------------------
    # Reputation Memory helpers
    # ------------------------------------------------------------------

    async def update_reputation_memory(self, round_summary: str, round_num: int = 0) -> None:
        """Ask the LLM to update its private reputation memory based on round events.

        Uses a scratch context (history cleared then restored) identical to the
        parse_result pattern so the update call does not pollute conversation history.
        """
        if self.reputation_memory is None:
            return

        current_memory = self.reputation_memory.to_prompt_block() or "(no observations recorded yet)"

        # Build the locked-peers notice block for the prompt
        locked_ids = self.reputation_memory.locked_peer_ids()
        if locked_ids:
            locked_lines = [
                "\n--- CONFIRMED Peers (semantic belief is ground truth — DO NOT update alignment/confidence/justification) ---"
            ]
            for pid in locked_ids:
                pr = self.reputation_memory.peers[pid]
                locked_lines.append(
                    f"  Player {pid}: {pr.semantic.alignment_role} "
                    f"(confidence {pr.semantic.confidence_score}/5) — {pr.semantic.justification}"
                )
            locked_lines.append("--- End CONFIRMED Peers ---\n")
            locked_peers_notice = "\n".join(locked_lines)
        else:
            locked_peers_notice = "\n"

        update_prompt = REPUTATION_UPDATE_PROMPT.format(
            round_summary=round_summary,
            current_memory=current_memory,
            locked_peers_notice=locked_peers_notice,
        )

        # Scratch context — save and clear history
        past_history = list(self.session.session.history)
        self.session.session.history = []

        self.session.session.inject({
            "role": "user",
            "content": update_prompt,
        })

        try:
            response = await self.session.session.action()
            raw = response.content if response.content else ""
        except Exception as e:
            get_game_logger().warning(f"[Memory Update] LLM call failed: {e}")
            raw = ""
        finally:
            # Restore original history regardless of outcome
            self.session.session.history = list(past_history)

        # Parse the JSON array returned by the LLM
        updates = []
        try:
            # Strip markdown fences if the LLM added them
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            updates = json.loads(cleaned)
            if not isinstance(updates, list):
                updates = []
        except (json.JSONDecodeError, ValueError):
            get_game_logger().warning(
                f"[Memory Update] Could not parse LLM response as JSON. Raw: {raw[:200]}"
            )

        # Apply updates
        self.reputation_memory.apply_llm_updates(updates)

        # --- Log the full updated memory state ---
        get_game_logger().info(f"##### Memory Update (after Round {round_num}) #####")
        for peer_id, record in self.reputation_memory.peers.items():
            lock_tag = " [LOCKED]" if record.is_locked else ""
            get_game_logger().info(
                f"[Memory Update] Player {peer_id}{lock_tag}: "
                f"alignment={record.semantic.alignment_role} | "
                f"confidence={record.semantic.confidence_score}/5 | "
                f"justification={record.semantic.justification}"
            )
            get_game_logger().info(
                f"[Memory Update]   alliances: {record.interactions.alliances}"
            )
            get_game_logger().info(
                f"[Memory Update]   conflicts: {record.interactions.conflicts}"
            )

    def inject_reputation_context(self, label: str = "") -> None:
        """Inject the reputation memory block into the session before an action.

        No-op if reputation_memory is None or has no non-trivial data yet.

        Parameters
        ----------
        label:
            Short string identifying which action is about to fire, e.g.
            "team_discussion", "propose_team", "vote_on_team".
        """
        if self.reputation_memory is None:
            return
        if not self.reputation_memory.has_non_trivial_data():
            return

        memory_block = self.reputation_memory.to_prompt_block()
        context_msg = REPUTATION_MEMORY_CONTEXT_PROMPT.format(
            header=REPUTATION_MEMORY_HEADER,
            memory_block=memory_block,
        )
        self.session.inject({
            "role": "user",
            "content": context_msg,
        })
        get_game_logger().info(
            f"[Memory Injected] Reputation context injected before {label}"
        )

    def lock_known_peers(self) -> None:
        """Lock peers whose alignment is already revealed at game start.

        Reads self.player_sides (set by see_sides() during initialization) and
        calls reputation_memory.lock_peer() for every peer whose side is known
        to be Evil (side == 0). This prevents the LLM from wasting tokens
        re-deriving ground-truth information each round.

        Only call this AFTER reputation_memory has been assigned.
        """
        if self.reputation_memory is None:
            return
        if not hasattr(self, 'player_sides') or self.player_sides is None:
            return

        for peer_id, side in enumerate(self.player_sides):
            if peer_id == self.id:
                continue
            if side == 0:  # Evil — already known with certainty
                self.reputation_memory.lock_peer(
                    peer_id,
                    alignment_role="Confirmed Evil",
                    justification="Role revealed at game start via faction knowledge.",
                )
                get_game_logger().info(
                    f"[ReputationMemory] Locked Player {peer_id} as Confirmed Evil "
                    f"(known at game start)"
                )
            elif side == 1 and self.role == 0:  # 0 is the role ID for Merlin
                self.reputation_memory.lock_peer(
                    peer_id,
                    alignment_role="Confirmed Servant",
                    justification="Role known to be Good by elimination.",
                )
                get_game_logger().info(
                    f"[ReputationMemory] Locked Player {peer_id} as Confirmed Servant "
                    f"(known by Merlin at game start)"
                )

    async def observe_team_result(self, mission_id, team: frozenset, votes: List[int], outcome: bool, **kwargs) -> None:
        # self.session.inject()
        await self.session.action({
            "role": "user",
            "content": verbalize_team_result(team, votes, outcome),
        })
    
    async def get_believed_sides(self, num_players: int, **kwargs) -> List[float]:
        input = {
            "role": "user",
            "content": f"To what extend do you believe each player to be Good, from Player 0 to Player {self.num_players - 1}? Please output probabilities within [0, 1] and round to two decimal places. If you are not sure, you can simply output 0.5.",
            "mode": "get_believed_sides",
            "role_name": self.role_name
        }
        # self.session.inject(input)
        believed_player_sides = await self.session.action(input)

        believed_player_sides = await self.session.parse_result(
            input   =   input,
            result  =   believed_player_sides
        )
        if isinstance(believed_player_sides, str):
            try:
                believed_player_sides = json.loads(believed_player_sides)
            except Exception:
                try:
                    believed_player_sides = eval(believed_player_sides)
                except Exception:
                    believed_player_sides = ([0.5] * self.num_players, [0.5] * self.num_players)
        get_game_logger().info(f"Sides: {believed_player_sides}")
        return believed_player_sides

    # async def discussion_end(self):
    #     content_prompt = f"Discussion has ended. Here are the contents:\nStatement from Leader {leader}: \n\"{leader_statement}\"\nAnd words from other players:\n{' '.join(discussion_history)}"
    #     self.session.inject({
    #         "role": "user",
    #         "content": content_prompt,
    #     })

    async def team_discussion(self, team_size, team_leader_id, mission_id, **kwargs):
        """Team discussion phase."""
        self.inject_reputation_context(label="team_discussion")

        fails_required = self.config.num_fails_for_quest[mission_id]
        
        side_prompt = DISCUSSION_GOOD_PLAYER if self.side == 1 else DISCUSSION_EVIL_PLAYER
        discussion_guidance = DISCUSSION_SCAFFOLD + side_prompt + DISCUSSION_SUFFIX
        
        content_prompt = CHOOSE_TEAM_LEADER.format(team_size) + discussion_guidance
        if self.id == team_leader_id:
            self.session.inject({
                "role": "user",
                "content": content_prompt,
            })
        else:
            self.session.inject({
                "role": "user",
                "content": f"Player {team_leader_id} is the quest leader for this round. " + discussion_guidance
            })

        dialogue = await self.session.action(receiver="all")
        return dialogue


    async def quest_discussion(self, team_size, team, team_leader_id, discussion_history, mission_id, **kwargs):
        fails_required = self.config.num_fails_for_quest[mission_id]

    
    async def propose_team(self, team_size, mission_id, **kwargs):
        self.inject_reputation_context(label="propose_team")
        content_prompt = CHOOSE_TEAM_ACTION.format(team_size, self.num_players-1)

        thought = COTHOUGHT_PROMPT
        input = {
            "role": "user",
            "content": content_prompt + '\n' + thought,
            "team_size": team_size,
            "seed": self.seed,
            "role_name": self.role_name,
            "mode": "choose_quest_team_action",
        }
        # self.session.inject(input)
        proposed_team = await self.session.action(input)

        get_game_logger().info(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####")
        get_game_logger().info(f"Thought: {proposed_team}")

        if isinstance(self.session.session, Session):
            proposed_team = await self.session.parse_result(input, proposed_team)
            try:
                proposed_team = json.loads(proposed_team)
            except Exception:
                try:
                    proposed_team = eval(proposed_team)
                except Exception:
                    proposed_team = list(range(team_size))
        proposed_team = frozenset(proposed_team)
        get_game_logger().info(f"Proposed Team: {proposed_team}")

        if isinstance(proposed_team, frozenset):
            return proposed_team
        else:
            raise ValueError(
                "Type of proposed_team must be frozenset, instead of {}.".format(type(proposed_team))
            )
        
    
    async def vote_on_team(self, team, mission_id, **kwargs):
        """Vote to approve or reject a team."""
        self.inject_reputation_context(label="vote_on_team")
        content_prompt = VOTE_TEAM_ACTION.format(list(team))
        
        thought = COTHOUGHT_PROMPT
        input = {
            "role": "user",
            "content": content_prompt + "\n" + thought,
            "side": int(self.side),
            "mode": "vote_on_team",
            "seed": self.seed,
            "role_name": self.role_name,
        }
        # self.session.inject(input)
        vote_result = await self.session.action(input)

        get_game_logger().info(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####")
        get_game_logger().info(f"Thought: {vote_result}")

        if isinstance(self.session.session, Session):
            vote_result = await self.session.parse_result(input, vote_result)
        vote_result = int(vote_result)

        if isinstance(vote_result, int):
            return vote_result
        else:
            raise ValueError(
                "Vote result should be either 0 or 1, instead of {}.".format(type(vote_result))
            )
    
    async def vote_on_mission(self, team, mission_id, **kwargs):
        self.inject_reputation_context(label="vote_on_mission")
        content_prompt = VOTE_MISSION_ACTION.format(list(team))

        thought = COTHOUGHT_PROMPT
        input = {
            "role": "user",
            "content": content_prompt + "\n" + thought,
            "side": int(self.side),
            "mode": "vote_on_mission",
            "seed": self.seed,
            "role_name": self.role_name,
        }
        # self.session.inject(input)
        vote_result = await self.session.action(input)

        get_game_logger().info(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####")
        get_game_logger().info(f"Thought: {vote_result}")

        if isinstance(self.session.session, Session):
            vote_result = await self.session.parse_result(input, vote_result)

        vote_result = int(vote_result)
        if isinstance(vote_result, int):
            return vote_result
        else:
            raise ValueError(
                "Vote result should be either 0 or 1, instead of {}.".format(type(vote_result))
            )
        

    async def assassinate(self, **kwargs):
        if self.role != 7:
            raise ValueError("Only the Assassin can assassinate.")
        self.inject_reputation_context(label="assassinate")

        thought = COTHOUGHT_PROMPT
        input = {
            "role": "user",
            "content": ASSASSINATION_PHASE.format(self.num_players-1) + "\n" + thought,
            "mode": "assassination",
            "seed": self.seed,
            "role_name": self.role_name,
        }
        # self.session.inject(input)
        assassinate_result = await self.session.action(input)
        # assassinate_result = int(assassinate_result)

        get_game_logger().info(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####")
        get_game_logger().info(f"Thought: {assassinate_result}")

        if isinstance(self.session.session, Session):
            assassinate_result = await self.session.parse_result(input, assassinate_result)
            assassinate_result = int(assassinate_result)

        if isinstance(assassinate_result, int):
            return assassinate_result
        else:
            raise ValueError(
                "Assassination result should be an integer, instead of {}.".format(type(assassinate_result))
            )