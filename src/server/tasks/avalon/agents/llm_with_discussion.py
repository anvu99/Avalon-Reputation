import json
import re
from typing import List, Dict, Tuple, Optional
from .agent import Agent
from ..engine import AvalonBasicConfig
from ..wrapper import AvalonSessionWrapper, Session
from ..prompts import (
    INTRODUCTION, INFO_ROLE, INFO_YOUR_ROLE, REVEAL_PROMPTS,
    CHOOSE_TEAM_LEADER, CHOOSE_TEAM_ACTION, VOTE_TEAM_ACTION, VOTE_MISSION_ACTION,
    ASSASSINATION_PHASE, COTHOUGHT_PROMPT, PERSONALITY_PROMPTS, DISCUSSION_SCAFFOLD,
    DISCUSSION_GOOD_PLAYER, DISCUSSION_EVIL_PLAYER, DISCUSSION_SUFFIX,
    REPUTATION_MEMORY_HEADER, REPUTATION_UPDATE_PROMPT, REPUTATION_MEMORY_CONTEXT_PROMPT,
    TUTORIAL_STRATEGIES_PROMPTS_ZERO_SHOT, SUMMARIZE_PROMPT, PERIODIC_PREDICTION_PROMPT,
    PERIODIC_MERLIN_PREDICTION_PROMPT, STRATEGIC_MEMORY_HEADER, EMPTY_MEMORY_NOTICE, CONFIRMED_PEERS_NOTICE_HEADER, CONFIRMED_PEERS_NOTICE_ITEM, CONFIRMED_PEERS_NOTICE_FOOTER, QUERY_BELIEF_PROMPT, DISCUSSION_LEADER_PROMPT,
    BAYESIAN_PERIODIC_PREDICTION_PROMPT, BAYESIAN_PERIODIC_MERLIN_PREDICTION_PROMPT,
    LONG_TERM_MEMORY_INJECTION_PROMPT
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
        # Personality: one of "naive", "deceptive", "default"
        self.personality = kwargs.pop('personality', 'default')
        if self.personality not in PERSONALITY_PROMPTS:
            self.personality = 'default'
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.seed = seed

        self.config = config
        self.use_bayesian_prediction = kwargs.get('use_bayesian_prediction', False)

        # Reputation memory — None unless activated for this agent by task.py
        self.reputation_memory: Optional[ReputationMemory] = None

        # Track previous round's prediction
        self.previous_prediction: dict = {}
        self.previous_merlin_prediction: dict = {}
        self.prediction_changes_log: List[str] = []

        self.summaries_log: List[str] = []
        self.ltm_text: str = ""

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_cothought(self) -> str:
        """Return COTHOUGHT_PROMPT with the personality CoT addendum appended."""
        cot_addon = PERSONALITY_PROMPTS.get(self.personality, PERSONALITY_PROMPTS['default'])['cot']
        return COTHOUGHT_PROMPT + cot_addon

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

        # Inject personality prefix if set
        personality_prefix = PERSONALITY_PROMPTS.get(self.personality, PERSONALITY_PROMPTS['default'])['prefix']
        if personality_prefix:
            self.session.inject({
                "role": "user",
                "content": personality_prefix,
                "mode": "system",
            })
            self.system_info += '\n\n' + personality_prefix
            get_game_logger().info(f"[Personality] Player {self.id} assigned personality='{self.personality}'")

    async def summarize(self, round_num: int = 0, mission_id: int = 0, log_snapshot: bool = True, **kwargs) -> None:
        summary = await self.session.action({
            "role": "user",
            "content": SUMMARIZE_PROMPT,
            "mode": "summarize"
        })
        self.summaries_log.append(f"[Mission {mission_id}, Round {round_num}]\n{summary}")
        if log_snapshot:
            get_game_logger().info(f"##### Memory Snapshot (Mission {mission_id}, Round {round_num}) #####\n{summary}")
        self.session.overwrite_history([])
        self.session.inject({
            'role': "user",
            'content': self.system_info
        })
        if self.ltm_text:
            self.session.inject({
                'role': "user",
                'content': LONG_TERM_MEMORY_INJECTION_PROMPT.format(memory_text=self.ltm_text)
            })
        self.session.inject({
            'role': "user",
            'content': STRATEGIC_MEMORY_HEADER.format(summary=summary)
        })
        return self.session.get_history()

    async def periodic_predict(self, round_num: int = 0, mission_id: int = 0, **kwargs) -> None:
        """Run two separate scratch-context LLM calls: one for Good/Evil prediction,
        one (non-Merlin only) for Merlin identity prediction."""

        past_history = list(self.session.session.history)

        # ------------------------------------------------------------------
        # Helper: run one scratch-context call and return raw text
        # ------------------------------------------------------------------
        async def _scratch_call(prompt_text: str) -> str:
            self.session.session.inject({"role": "user", "content": prompt_text})
            try:
                response = await self.session.session.action()
                return response.content if response.content else ""
            except Exception as e:
                get_game_logger().warning(f"[Periodic Prediction] LLM call failed: {e}")
                return ""
            finally:
                self.session.session.history = list(past_history)

        # ------------------------------------------------------------------
        # Helper: extract a {{int: float}} dict from any raw LLM response
        # ------------------------------------------------------------------
        def _parse_dict(raw: str) -> dict:
            match = re.search(r'\{[^{}]+\}', raw)
            if not match:
                return {}
            try:
                parsed = eval(match.group())
                if isinstance(parsed, dict) and all(isinstance(k, int) for k in parsed):
                    return parsed
            except Exception as e:
                get_game_logger().warning(f"[Periodic Prediction] Failed to parse dict: {e}")
            return {}

        # ------------------------------------------------------------------
        # Call 1: Good/Evil alignment
        # ------------------------------------------------------------------
        past_changes_str = "\n".join(self.prediction_changes_log) if self.prediction_changes_log else "(None recorded yet)"
        
        if self.use_bayesian_prediction:
            good_prompt_template = BAYESIAN_PERIODIC_PREDICTION_PROMPT
        else:
            good_prompt_template = PERIODIC_PREDICTION_PROMPT

        good_prompt = good_prompt_template.format(
            previous_prediction=self.previous_prediction if self.previous_prediction else "{}",
            self_id=self.id,
            past_changes_log=past_changes_str,
        )
        raw_good = await _scratch_call(good_prompt)

        prediction_dict = _parse_dict(raw_good)
        if prediction_dict:
            self.previous_prediction = prediction_dict

        changes_text = ""
        if "Changes:" in raw_good:
            changes_text = raw_good.split("Changes:")[1].strip()
            if changes_text and not changes_text.startswith("(omit the Changes"):
                self.prediction_changes_log.append(f"[Mission {mission_id}, Round {round_num}]\n{changes_text}")

        # ------------------------------------------------------------------
        # Call 2: Merlin identity (non-Merlin roles only)
        # ------------------------------------------------------------------
        merlin_dict = {}
        if self.role_name != "Merlin":
            if self.use_bayesian_prediction:
                merlin_prompt = BAYESIAN_PERIODIC_MERLIN_PREDICTION_PROMPT.format(
                    self_id=self.id,
                    previous_prediction=self.previous_merlin_prediction if self.previous_merlin_prediction else "{}"
                )
            else:
                merlin_prompt = PERIODIC_MERLIN_PREDICTION_PROMPT.format(self_id=self.id)
            raw_merlin = await _scratch_call(merlin_prompt)
            merlin_dict = _parse_dict(raw_merlin)
            if merlin_dict:
                self.previous_merlin_prediction = merlin_dict

        # ------------------------------------------------------------------
        # Log
        # ------------------------------------------------------------------
        get_game_logger().info(f"##### Periodic Prediction by Player {self.id} (Mission {mission_id}, Round {round_num}) #####")
        get_game_logger().info(f"[Good/Evil] {self.previous_prediction}")
        if self.previous_merlin_prediction:
            get_game_logger().info(f"[Merlin]    {self.previous_merlin_prediction}")
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

        current_memory = self.reputation_memory.to_prompt_block() or EMPTY_MEMORY_NOTICE

        # Build the locked-peers notice block for the prompt
        locked_ids = self.reputation_memory.locked_peer_ids()
        if locked_ids:
            locked_lines = [CONFIRMED_PEERS_NOTICE_HEADER]
            for pid in locked_ids:
                pr = self.reputation_memory.peers[pid]
                locked_lines.append(
                    CONFIRMED_PEERS_NOTICE_ITEM.format(
                        pid=pid,
                        alignment_role=pr.semantic.alignment_role,
                        confidence_score=pr.semantic.confidence_score,
                        justification=pr.semantic.justification
                    )
                )
            locked_lines.append(CONFIRMED_PEERS_NOTICE_FOOTER)
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
        past_changes_str = "\n".join(self.prediction_changes_log) if self.prediction_changes_log else "(None recorded yet)"
        input = {
            "role": "user",
            "content": QUERY_BELIEF_PROMPT.format(max_player_id=self.num_players - 1, past_changes_log=past_changes_str),
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
                "content": DISCUSSION_LEADER_PROMPT.format(team_leader_id=team_leader_id) + discussion_guidance
            })

        dialogue = await self.session.action(receiver="all")
        return dialogue


    async def quest_discussion(self, team_size, team, team_leader_id, discussion_history, mission_id, **kwargs):
        fails_required = self.config.num_fails_for_quest[mission_id]

    
    async def propose_team(self, team_size, mission_id, **kwargs):
        self.inject_reputation_context(label="propose_team")
        content_prompt = CHOOSE_TEAM_ACTION.format(team_size, self.num_players-1)

        thought = self.get_cothought()
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
            
            if not isinstance(proposed_team, list) or len(proposed_team) != team_size:
                import random
                if not isinstance(proposed_team, list):
                    proposed_team = list(proposed_team) if isinstance(proposed_team, (set, frozenset, tuple)) else []
                proposed_team = list(set([x for x in proposed_team if isinstance(x, int) and 0 <= x < self.num_players]))
                while len(proposed_team) > team_size:
                    proposed_team.pop()
                while len(proposed_team) < team_size:
                    candidate = random.randint(0, self.num_players - 1)
                    if candidate not in proposed_team:
                        proposed_team.append(candidate)

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
        
        thought = self.get_cothought()
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

        thought = self.get_cothought()
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

        thought = self.get_cothought()
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