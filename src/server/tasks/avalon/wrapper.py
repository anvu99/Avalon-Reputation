from copy import deepcopy
from typing import Dict, Union
from src.server.task import Session
from .utils import get_team_result, get_vote_result, get_assassination_result, get_believed_player_sides, get_game_logger
from .prompts import CHECK_CHOOSE_TEAM_PROMPT, CHECK_VOTE_ON_QUEST_PROMPT, CHECK_VOTE_ON_TEAM_PROMPT, CHECK_ASSASSINATE_PROMPT, CHECK_BELIEVED_SIDES_PROMPT, GET_MERLIN_PROBABILITIES, RETRY_TEAM_SIZE_PROMPT, RETRY_TEAM_PLAYERS_PROMPT, RETRY_VOTE_TEAM_PROMPT, RETRY_VOTE_MISSION_PROMPT, CHECK_BELIEVED_SIDES_DISCRETE_PROMPT, GET_MERLIN_PROBABILITIES_DISCRETE
from src.typings import SampleStatus
from src.typings import AgentContextLimitException
from .avalon_exception import AvalonAgentActionException
from src.utils import ColorMessage

from multi_agent.typings import FakeSession, Proxy
from multi_agent.session_wrapper import SessionWrapper

class FakeSession:
    def __init__(self):
        self.history: list=[]    # Fake history

    async def action(self, input: Dict):
        # try:
        #     return input["naive_result"]
        # except:
        #     return "No naive results provided."
        pass

    def inject(self, input: Dict):
        pass

class AvalonSessionWrapper(SessionWrapper):
    def __init__(self, session: Union[Session, FakeSession], proxy: Proxy, task=None):
        # super().__init__(session, proxy)
        self.session = session
        self.proxy = proxy
        self.task = task
        self.decorate_method('action')
        self.decorate_method('inject')
        self.decorate_method('parse_result')

    def decorate_method(self, method_name):
        # Get the method
        method = getattr(self, method_name)

        # Decorate and replace the method
        setattr(self, method_name, self.proxy.method_wrapper(method))

    def get_history(self):
        return self.session.history

    def overwrite_history(self, history: list):
        self.proxy.history[self.proxy.current_agent] = list(history)
        self.session.history = list(history)

    def inject(self, input: Dict, **kwargs):
        if isinstance(self.session, Session):
            # print("SESSION")
            self.session.inject({
                'role': input['role'],
                'content': input['content']
            })
        elif isinstance(self.session, FakeSession):
            pass

    async def action(self, input: Dict=None, **kwargs):
        if isinstance(self.session, Session):
            if input is not None:
                self.session.inject({
                    'role': input['role'],
                    'content': input['content']
                })
            self.proxy.balance_history()
            
            logger = get_game_logger()
            if logger.name.endswith("game_0") and getattr(self.proxy, 'current_agent', -1) == 0:
                import os, json
                if logger.handlers:
                    log_path = logger.handlers[0].baseFilename
                    dir_name = os.path.dirname(log_path)
                    prompt_log_path = os.path.join(dir_name, "agent_0_prompts_game_0.log")
                    
                    try:
                        history_str = json.dumps(self.session.history, indent=2)
                    except Exception:
                        history_str = str(self.session.history)

                    # 1. Write prompt to prompt-specific log
                    with open(prompt_log_path, "a") as f:
                        f.write("========== NEW PROMPT SENT TO AGENT 0 ==========\n")
                        f.write(history_str)
                        f.write("\n\n")

            response = await self.session.action()

            if response.status == SampleStatus.AGENT_CONTEXT_LIMIT:
                raise AgentContextLimitException()
            if response.content is None:
                raise RuntimeError("Response content is None.")

            thinking = getattr(response.content, 'thinking', '')
            finish_reason = getattr(response.content, 'finish_reason', '')
            if thinking:
                curr_agent = getattr(self.proxy, 'current_agent', -1)
                logger.info(f"Player {curr_agent} (Thinking Process) [finish_reason: {finish_reason}]:\n{thinking}")

            if logger.name.endswith("game_0") and getattr(self.proxy, 'current_agent', -1) == 0:
                import os
                if logger.handlers:
                    log_path = logger.handlers[0].baseFilename
                    dir_name = os.path.dirname(log_path)
                    prompt_log_path = os.path.join(dir_name, "agent_0_prompts_game_0.log")
                    
                    # 1. Write response to prompt-specific log
                    with open(prompt_log_path, "a") as f:
                        f.write("========== GENERATED RESPONSE FROM AGENT 0 ==========\n")
                        f.write(str(response.content))
                        f.write("\n\n")

            return response.content
        elif isinstance(self.session, FakeSession):
            return input.pop('naive_result', None)
        
    async def parse_result(self, input: Dict, result: str):
        # print(result)
        mode = input['mode']
        past_history = list(self.session.history) # Store the history before the action
        # print("Past history: ", past_history)
        self.session.history = [] # Clear the history
        
        use_single = getattr(self.task, 'use_single_stage_parse', False)

        if mode == "choose_quest_team_action":
            team_size = input['team_size']
            if use_single:
                import re
                # First try to isolate the answer block to avoid matching intermediate numbers in reasoning
                match = re.search(r'Answer:\s*\[([^\]]+)\]', result, re.IGNORECASE)
                if match:
                    answer = get_team_result(match.group(0))
                else:
                    answer = get_team_result(result)
            else:
                self.session.inject({
                    "role": "user",
                    "content": result + '\n\n' + CHECK_CHOOSE_TEAM_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                answer = get_team_result(answer)
            
            # Clean and deduplicate player IDs extracted to prevent repeated list failures
            unique_ids = []
            for x in answer:
                if isinstance(x, int) and 0 <= x < self.proxy.num_agents:
                    if x not in unique_ids:
                        unique_ids.append(x)
            answer = unique_ids

            if len(answer) != team_size:
                # Run another action to get the correct team size
                self.session.history = list(past_history)
                self.session.inject({
                    "role": "user",
                    "content": RETRY_TEAM_SIZE_PROMPT.format(team_size=team_size, invalid_size=len(answer))
                })
                answer = await self.session.action()
                answer = answer.content
                past_history = list(self.session.history) # Update the history
                self.session.history = [] # Clear the history

                self.session.inject({
                    "role": "user",
                    "content": answer + '\n\n' + CHECK_CHOOSE_TEAM_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                try:
                    answer = get_team_result(answer)
                    unique_ids = []
                    for x in answer:
                        if isinstance(x, int) and 0 <= x < self.proxy.num_agents:
                            if x not in unique_ids:
                                unique_ids.append(x)
                    while len(unique_ids) > team_size:
                        unique_ids.pop()
                    import random
                    while len(unique_ids) < team_size:
                        candidates = [c for c in range(self.proxy.num_agents) if c not in unique_ids]
                        if not candidates:
                            break
                        unique_ids.append(random.choice(candidates))
                    answer = unique_ids
                    assert len(answer) == team_size
                    assert isinstance(answer, list)
                except:
                    get_game_logger().warning(f"Warning: Defaulting team to first {team_size} players due to invalid size retry: {answer}")
                    answer = list(range(team_size))

        elif mode == "vote_on_team":
            answer = get_vote_result(result)

            result_dict = {
                "No": 0,
                "Yes": 1
            }

            if answer not in ["No", "Yes"]:
                # Run another action to get the correct vote result
                self.session.history = list(past_history)
                self.session.inject({
                    "role": "user",
                    "content": RETRY_VOTE_TEAM_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                past_history = list(self.session.history) # Update the history
                self.session.history = [] # Clear the history

                self.session.inject({
                    "role": "user",
                    "content": answer + '\n\n' + CHECK_VOTE_ON_TEAM_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                answer = get_vote_result(answer)
            try:
                answer = result_dict[answer]
            except:
                get_game_logger().warning(f"Warning: Defaulting team vote to No (reject) due to invalid output: {answer}")
                answer = 0

        elif mode == "vote_on_mission":
            answer = get_vote_result(result)

            result_dict = {
                "No": 0,
                "Yes": 1
            }

            if answer not in ["No", "Yes"]:
                # Run another action to get the correct vote result
                self.session.history = list(past_history)
                self.session.inject({
                    "role": "user",
                    "content": RETRY_VOTE_MISSION_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                past_history = list(self.session.history) # Update the history
                self.session.history = [] # Clear the history

                self.session.inject({
                    "role": "user",
                    "content": answer + '\n\n' + CHECK_VOTE_ON_QUEST_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                answer = get_vote_result(answer)
            try:
                answer = result_dict[answer]
            except:
                get_game_logger().warning(f"Warning: Defaulting mission vote to Yes (pass) due to invalid output: {answer}")
                answer = 1

        elif mode == "assassination":
            if use_single:
                import re
                try:
                    match = re.search(r'Answer:\s*\[?(\d+)\]?', result, re.IGNORECASE)
                    if match:
                        answer = int(match.group(1))
                    else:
                        answer = int(get_assassination_result(result, ""))
                except Exception:
                    answer = 0
            else:
                self.session.inject({
                    "role": "user",
                    "content": result + '\n\n' + CHECK_ASSASSINATE_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                try:
                    answer = int(get_assassination_result(result, answer))
                except:
                    answer = 0
            if not isinstance(answer, int) or not (0 <= answer < self.proxy.num_agents):
                get_game_logger().warning(f"Warning: Defaulting assassination to Player 0 due to invalid output: {answer}")
                answer = 0
                
        elif mode == "get_believed_sides":
            use_discrete = getattr(self.task, 'use_discrete_rating', False)
            default_val = 3.0 if use_discrete else 0.5

            if use_single:
                try:
                    scores, merlin_scores = get_believed_player_sides(result)
                    answer_good = []
                    answer_merlin = []
                    for i in range(self.proxy.num_agents):
                        answer_good.append(float(scores.get(i, default_val)) if isinstance(scores, dict) else default_val)
                        answer_merlin.append(float(merlin_scores.get(i, default_val)) if isinstance(merlin_scores, dict) else default_val)
                    answer = (answer_good, answer_merlin)
                except Exception:
                    get_game_logger().warning(f"Warning: Defaulting believed sides to {default_val} due to invalid output: {result}")
                    answer = ([default_val] * self.proxy.num_agents, [default_val] * self.proxy.num_agents)
            else:
                side_prompt = CHECK_BELIEVED_SIDES_DISCRETE_PROMPT if use_discrete else CHECK_BELIEVED_SIDES_PROMPT
                merlin_prompt = GET_MERLIN_PROBABILITIES_DISCRETE if use_discrete else GET_MERLIN_PROBABILITIES
                prompt = result + '\n\n' + side_prompt
                role_name = input.get("role_name", "")
                if role_name != "Merlin":
                    prompt += '\n' + merlin_prompt
                    
                self.session.inject({
                    "role": "user",
                    "content": prompt
                })
                answer = await self.session.action()
                answer = answer.content
                try:
                    scores, merlin_scores = get_believed_player_sides(answer)
                    answer_good = []
                    answer_merlin = []
                    for i in range(self.proxy.num_agents):
                        answer_good.append(float(scores.get(i, default_val)) if isinstance(scores, dict) else default_val)
                        answer_merlin.append(float(merlin_scores.get(i, default_val)) if isinstance(merlin_scores, dict) else default_val)
                    answer = (answer_good, answer_merlin)
                except:
                    get_game_logger().warning(f"Warning: Defaulting believed sides to {default_val} due to invalid output: {answer}")
                    answer = ([default_val] * self.proxy.num_agents, [default_val] * self.proxy.num_agents)

        elif mode == "get_believed_merlin":
            use_discrete = getattr(self.task, 'use_discrete_rating', False)
            default_val = 3.0 if use_discrete else 0.5

            if use_single:
                try:
                    _, merlin_scores = get_believed_player_sides(result)
                    answer_merlin = []
                    for i in range(self.proxy.num_agents):
                        answer_merlin.append(float(merlin_scores.get(i, default_val)) if isinstance(merlin_scores, dict) else default_val)
                    answer = answer_merlin
                except Exception:
                    get_game_logger().warning(f"Warning: Defaulting believed Merlin to {default_val} due to invalid output: {result}")
                    answer = [default_val] * self.proxy.num_agents
            else:
                merlin_prompt = GET_MERLIN_PROBABILITIES_DISCRETE if use_discrete else GET_MERLIN_PROBABILITIES
                prompt = result + '\n\n' + merlin_prompt
                self.session.inject({
                    "role": "user",
                    "content": prompt
                })
                answer = await self.session.action()
                answer = answer.content
                try:
                    _, merlin_scores = get_believed_player_sides(answer)
                    answer_merlin = []
                    for i in range(self.proxy.num_agents):
                        answer_merlin.append(float(merlin_scores.get(i, default_val)) if isinstance(merlin_scores, dict) else default_val)
                    answer = answer_merlin
                except:
                    get_game_logger().warning(f"Warning: Defaulting believed Merlin to {default_val} due to invalid output: {answer}")
                    answer = [default_val] * self.proxy.num_agents

        # Restore the history
        self.session.history = list(past_history)

        verbal_team_act = {
            0: "Reject the team" if mode == "vote_on_team" else "Fail the mission",
            1: "Approve the team" if mode == "vote_on_team" else "Pass the mission",
        }
        if mode in ["vote_on_team", "vote_on_mission"]:
            get_game_logger().info(f"Action: {verbal_team_act[answer]}")
        elif mode == "choose_quest_team_action":
            get_game_logger().info(f"Action: Propose team {answer}")
        elif mode == "assassination":
            get_game_logger().info(f"Action: Assassinate Player {answer}")
        elif mode == "get_believed_sides":
            get_game_logger().info(f"Action: Believed sides: {answer}")
        return answer