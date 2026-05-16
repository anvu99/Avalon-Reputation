from copy import deepcopy
from typing import Dict, Union
from src.server.task import Session
from .utils import get_team_result, get_vote_result, get_assassination_result, get_believed_player_sides, get_game_logger
from .prompts import CHECK_CHOOSE_TEAM_PROMPT, CHECK_VOTE_ON_QUEST_PROMPT, CHECK_VOTE_ON_TEAM_PROMPT, CHECK_ASSASSINATE_PROMPT, CHECK_BELIEVED_SIDES_PROMPT, GET_MERLIN_PROBABILITIES
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
    def __init__(self, session: Union[Session, FakeSession], proxy: Proxy):
        # super().__init__(session, proxy)
        self.session = session
        self.proxy = proxy
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
            response = await self.session.action()

            if response.status == SampleStatus.AGENT_CONTEXT_LIMIT:
                raise AgentContextLimitException()
            if response.content is None:
                raise RuntimeError("Response content is None.")
            return response.content
        elif isinstance(self.session, FakeSession):
            return input.pop('naive_result', None)
        
    async def parse_result(self, input: Dict, result: str):
        # print(result)
        mode = input['mode']
        past_history = list(self.session.history) # Store the history before the action
        # print("Past history: ", past_history)
        self.session.history = [] # Clear the history
        if mode == "choose_quest_team_action":
            team_size = input['team_size']
            self.session.inject({
                "role": "user",
                "content": result + '\n\n' + CHECK_CHOOSE_TEAM_PROMPT
            })
            answer = await self.session.action()
            answer = answer.content
            answer = get_team_result(answer)
            if len(answer) != team_size:
                # Run another action to get the correct team size
                self.session.history = list(past_history)
                self.session.inject({
                    "role": "user",
                    "content": f"You should choose a team of size {team_size}, instead of size {len(answer)} as you did. Please output a list of player ids with the correct team size."
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
                    assert len(answer) == team_size
                    assert isinstance(answer, list)
                except:
                    get_game_logger().warning(f"Warning: Defaulting team to first {team_size} players due to invalid size retry: {answer}")
                    answer = list(range(team_size))
            elif max(answer) >= self.proxy.num_agents or min(answer) < 0:
                # Run another action to get the correct team size
                self.session.history = list(past_history)
                self.session.inject({
                    "role": "user",
                    "content": f"You should choose a team of size {team_size} from Player 0 to {self.proxy.num_agents-1}, instead of team {answer} as you did. Please output a list of player ids with the correct team size and Player ids."
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
                    assert len(answer) == team_size
                    assert isinstance(answer, list)
                    assert max(answer) < self.proxy.num_agents and min(answer) >= 0
                except:
                    get_game_logger().warning(f"Warning: Defaulting team to first {team_size} players due to invalid output: {answer}")
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
                    "content": f"You surely are a player in the game. Please output `Yes` or `No` to vote on the team."
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
                    "content": "You surely are a player in the game, and you are a member in the quest. Please output `Yes` or `No` to vote on the quest."
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
            self.session.inject({
                "role": "user",
                "content": result + '\n\n' + CHECK_ASSASSINATE_PROMPT
            })
            answer = await self.session.action()
            answer = answer.content
            try:
                answer = int(get_assassination_result(result, answer))
            except:
                get_game_logger().warning(f"Warning: Defaulting assassination to Player 0 due to invalid output: {answer}")
                answer = 0
                
        elif mode == "get_believed_sides":
            prompt = result + '\n\n' + CHECK_BELIEVED_SIDES_PROMPT
            role_name = input.get("role_name", "")
            if role_name != "Merlin":
                prompt += '\n' + GET_MERLIN_PROBABILITIES
                
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
                    answer_good.append(scores.get(i, 0.5) if isinstance(scores, dict) else 0.5)
                    answer_merlin.append(merlin_scores.get(i, 0.5) if isinstance(merlin_scores, dict) else 0.5)
                answer = (answer_good, answer_merlin)
            except:
                get_game_logger().warning(f"Warning: Defaulting believed sides to 0.5 due to invalid output: {answer}")
                answer = ([0.5] * self.proxy.num_agents, [0.5] * self.proxy.num_agents)

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