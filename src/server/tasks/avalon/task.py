import sys
import json
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional

from src.server.task import Task, Session
from src.typings import TaskSampleExecutionResult, TaskOutput, SampleIndex, AgentOutputStatus, SampleStatus
from src.utils import ColorMessage

from .engine import *
from .task_scoring import *

from .prompts import *
from .agents.baseline_agents import *

from .wrapper import FakeSession, AvalonSessionWrapper
from .utils import verbalize_team_result, verbalize_mission_result, get_game_logger

from .agents.llm_with_discussion import LLMAgentWithDiscussion
from .reputation_memory import ReputationMemory
from .long_term_memory import LongTermMemory

from src.typings import AgentContextLimitException
from .avalon_exception import AvalonAgentActionException

from multi_agent.proxy import MultiAgentProxy
import logging
from .dialogue import AvalonDiagloue

AGENT_FINDER = {
    'naive': find_naive_agent,
    'llm': LLMAgentWithDiscussion,
}


def build_round_summary(
    leader: int,
    team: frozenset,
    votes: list,
    outcome: bool,
    discussion_history: list,
    quest_result: bool = None,
    num_fails: int = None,
) -> str:
    """Build a human-readable summary of a completed round for ReputationMemory updates.

    Parameters
    ----------
    leader:
        Player id of the quest leader who proposed the team.
    team:
        The proposed team (frozenset of player ids).
    votes:
        List of team votes, one per player (1 = approve, 0 = reject).
    outcome:
        True if the team was approved, False if rejected.
    discussion_history:
        List of "Player X:\n<text>" strings from this round's discussion.
    quest_result:
        True if the quest succeeded, False if it failed, None if team was rejected.
    num_fails:
        Number of fail votes cast on the quest (None if team was rejected).
    """
    lines = []
    lines.append(f"Quest Leader: Player {leader}")
    lines.append(f"Proposed Team: {sorted(list(team))}")

    vote_labels = []
    for pid, v in enumerate(votes):
        vote_labels.append(f"Player {pid}: {'Approve' if v == 1 else 'Reject'}")
    lines.append("Team Vote Results: " + ", ".join(vote_labels))
    lines.append(f"Team Vote Outcome: {'Approved' if outcome else 'Rejected'}")

    if quest_result is not None:
        lines.append(f"Quest Outcome: {'Succeeded' if quest_result else 'Failed'}")
        if num_fails is not None:
            lines.append(f"Number of Fail Votes: {num_fails}")

    if discussion_history:
        lines.append("Discussion this round:")
        for entry in discussion_history:
            lines.append(f"  {entry}")
    else:
        lines.append("Discussion this round: (none)")

    return "\n".join(lines)


class AvalonBench(Task):
    def __init__(self, num_players, agent_list, discussion, data_file,
                 use_reputation_memory: bool = False,
                 long_term_memories: Optional[Dict[int, 'LongTermMemory']] = None,
                 long_term_memory: Optional['LongTermMemory'] = None,  # backward compat
                 personality_list: Optional[List[str]] = None,
                 **configs):
        super().__init__("avalon", **configs)

        self.num_players = num_players
        self.agent_list = agent_list

        self.discussion = discussion
        self.data_file = data_file
        self.num_discussion_rounds = configs.pop('num_discussion_rounds', 1)
        self.use_reputation_memory = use_reputation_memory

        # Per-player personality list ("naive", "deceptive", "default").
        # Defaults to "default" for every player if not provided.
        if personality_list is not None:
            self.personality_list = personality_list
        else:
            self.personality_list = ["default"] * num_players

        # Support both new dict API and legacy single-LTM API
        if long_term_memories is not None:
            self.long_term_memories = long_term_memories
        elif long_term_memory is not None:
            self.long_term_memories = {0: long_term_memory}
        else:
            self.long_term_memories = {}

        # Keep legacy attribute pointing to Player 0's LTM for backward compat
        self.long_term_memory = self.long_term_memories.get(0, None)

        tracked_ids = sorted(self.long_term_memories.keys())
        log_mem = configs.pop('log_memory_snapshots_for', None)
        self.log_memory_snapshots_for = log_mem if log_mem is not None else (tracked_ids if tracked_ids else [0])
        
        pred_for = configs.pop('predict_for', None)
        self.predict_for = pred_for if pred_for is not None else (tracked_ids if tracked_ids else [0])
        self.num_repeats = configs.pop('num_repeats', 1)
        self.use_bayesian_prediction = configs.pop('use_bayesian_prediction', False)

        self.data: List[Tuple[dict, set]] = []
        self.inputs = []
        with open(self.data_file, "r") as f:
            data_object = json.load(f)
            
        start_idx = configs.pop('start_idx', None)
        end_idx = configs.pop('end_idx', None)
        if start_idx is not None or end_idx is not None:
            data_object = data_object[start_idx:end_idx]
            
        for data_item in data_object:
            for _ in range(self.num_repeats):
                self.data.append((data_item, -1))
                self.inputs.append(data_item)

        # Shuffle the duplicated data so agents encounter different setups in a mixed order
        import random
        combined = list(zip(self.data, self.inputs))
        random.Random(42).shuffle(combined)
        self.data, self.inputs = map(list, zip(*combined))

        self.seed = configs.pop('seed', 0)

    @staticmethod
    def compute_batch_metrics(results: List[TaskOutput], batch_num: int = -1, ltm_size_chars: int = 0) -> Dict[str, Any]:
        """Compute per-batch performance metrics for a list of TaskOutput results.

        Parameters
        ----------
        results:
            List of TaskOutput objects for this batch (exceptions already filtered out).
        batch_num:
            The zero-indexed batch number (-1 if not in LTM mode).
        ltm_size_chars:
            Character length of the LTM text that was *available entering* this batch.
            0 means the agent played with no memory yet.
        """
        win_counter = 0
        deduc_acc_sum = 0.0
        merlin_correct = 0
        merlin_total = 0
        win_as_good = 0
        win_as_evil = 0
        games_as_good = 0
        games_as_evil = 0
        valid_games = 0

        for result in results:
            if result.status != SampleStatus.COMPLETED or result.result is None:
                continue
            r = result.result
            llm_idx = r.get('llm_idx', 0)
            valid_games += 1

            won = r.get(f'Player_{llm_idx}_wins', False)
            if won:
                win_counter += 1

            deduc_acc_sum += r.get(f'Player_{llm_idx}_deduc_acc', 0.0)

            role = r.get(f'role_of_Player_{llm_idx}', '')
            side = 1 if role in ('Merlin', 'Servant') else 0
            if side == 1:
                games_as_good += 1
                if won:
                    win_as_good += 1
            else:
                games_as_evil += 1
                if won:
                    win_as_evil += 1

            # Merlin detection: only meaningful when Player 0 is NOT Merlin
            if role != 'Merlin':
                true_merlin = r.get('true_merlin_id')
                merlin_pred = r.get(f'Player_{llm_idx}_merlin_pred', {})
                if true_merlin is not None and isinstance(merlin_pred, (list, dict)) and merlin_pred:
                    try:
                        # Convert list or dict to dict with int keys, ensuring float values
                        if isinstance(merlin_pred, list):
                            merlin_pred_int = {i: float(v) for i, v in enumerate(merlin_pred)}
                        else:
                            merlin_pred_int = {int(k): float(v) for k, v in merlin_pred.items()}
                        
                        if merlin_pred_int:
                            predicted_merlin = max(merlin_pred_int, key=merlin_pred_int.get)
                            if predicted_merlin == true_merlin:
                                merlin_correct += 1
                            merlin_total += 1
                    except (ValueError, TypeError, AttributeError):
                        pass

        denom = max(valid_games, 1)
        merlin_denom = max(merlin_total, 1)

        metrics = {
            "batch_num": batch_num,
            "n_valid_games": valid_games,
            "ltm_size_chars": ltm_size_chars,
            "win_rate": round(win_counter / denom, 4),
            "avg_deduction_acc": round(deduc_acc_sum / denom, 4),
            "merlin_detection_acc": round(merlin_correct / merlin_denom, 4) if merlin_total > 0 else None,
            "win_rate_as_good": round(win_as_good / max(games_as_good, 1), 4) if games_as_good > 0 else None,
            "win_rate_as_evil": round(win_as_evil / max(games_as_evil, 1), 4) if games_as_evil > 0 else None,
        }
        return metrics

    def calculate_overall(self, results: List[TaskOutput], per_batch_metrics: List[Dict] = None) -> Dict[str, Any]:
        overall = self.compute_batch_metrics(results, batch_num=-1)

        summary = {
            "n_valid_games": overall["n_valid_games"],
            "win_rate": overall["win_rate"],
            "avg_deduction_acc": overall["avg_deduction_acc"],
            "merlin_detection_acc": overall["merlin_detection_acc"],
            "win_rate_as_good": overall["win_rate_as_good"],
            "win_rate_as_evil": overall["win_rate_as_evil"],
        }
        if per_batch_metrics:
            summary["per_batch_learning_curve"] = per_batch_metrics

        return summary

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.data)))

    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        assert isinstance(index, int), "Index must be an integer"
        assert self.inputs[index]['num_players'] == self.num_players, "Number of players must be the same"
        proxy = MultiAgentProxy(session, self.num_players)
        sessions = [AvalonSessionWrapper(session, proxy) for _ in range(self.num_players)]
        proxy.initialize_sessions(sessions)
        env = AvalonGameEnvironment.from_presets(self.inputs[index])
        scoring = AvalonScoring(env.config)

        true_player_sides = []
        believed_player_sides = []
        believed_merlin_sides = []
        game_env_log = []

        llm_idx = 0

        num_players = self.num_players

        player_list = []

        if num_players != len(sessions):
            raise ValueError(
                f"Number of players {num_players} doesn't match number of sessions {len(sessions)}"
            )
        
        dialogue_history = AvalonDiagloue()
        discussion_history = []  # local per-game, avoids cross-game corruption

        get_game_logger().info("Check initialization")
        # Initialize players. Please remember to let Merlin and Evil players see the sides of all players.
        for i, (role_i, role_name, side) in enumerate(env.get_roles()):
            player_list.append(AGENT_FINDER[self.agent_list[i]](
                                        id          =   i,
                                        name        =   f"Player {i}",
                                        config      =   env.config,
                                        side        =   side,
                                        role        =   role_i,
                                        num_players =   num_players,
                                        session     =   sessions[i],
                                        role_name   =   role_name,
                                        merlin      =   env.config.merlin,
                                        percival    =   env.config.percival,
                                        morgana     =   env.config.morgana,
                                        mordred     =   env.config.mordred,
                                        oberon      =   env.config.oberon,
                                        num_good    =   env.config.num_good,
                                        num_evil    =   env.config.num_evil,
                                        discussion  =   self.discussion,
                                        seed        =   self.seed, # TODO: seed
                                        use_bayesian_prediction = self.use_bayesian_prediction,
                                        personality = self.personality_list[i] if i < len(self.personality_list) else 'default',
                                        ))
            # If the player is Merlin or Evil, let them see the sides of all players.
            player_sides = [side for _, _, side in env.get_roles()]
            if player_list[i].role == 0 or player_list[i].side == 0:
                player_list[i].see_sides(player_sides)
                await player_list[i].initialize_game_info(player_list=env.get_roles(), env=env)
            else:
                await player_list[i].initialize_game_info(player_list=env.get_roles(), env=env)
            
            proxy.get_next_agent()

        # Activate ReputationMemory for Player 0 (llm_idx) if enabled
        if self.use_reputation_memory and hasattr(player_list[llm_idx], 'reputation_memory'):
            player_list[llm_idx].reputation_memory = ReputationMemory(
                player_id=llm_idx,
                num_players=num_players,
            )
            get_game_logger().info(
                f"[ReputationMemory] Initialized for Player {llm_idx} "
                f"(tracking peers: {sorted(player_list[llm_idx].reputation_memory.peers.keys())})"
            )
            # Lock any peers whose alignment is already known at game start
            player_list[llm_idx].lock_known_peers()

        # Inject Long-Term Memory for each tracked agent
        for pid, ltm in self.long_term_memories.items():
            if not ltm.is_empty():
                player_list[pid].ltm_text = ltm.memory_text
                ltm_block = ltm.to_prompt_block()
                sessions[pid].inject({
                    "role": "user",
                    "content": ltm_block,
                    "mode": "system",
                })
                get_game_logger().info(
                    f"##### [Long-Term Memory Injected for Player {pid}] ({len(ltm.memory_text)} chars) #####\n"
                    f"{ltm.memory_text}"
                )
            else:
                get_game_logger().info(f"##### [Long-Term Memory] Player {pid}: No memory yet — playing without LTM this game. #####")

        # Track per-round state needed for build_round_summary across phases
        current_leader: int = -1
        current_team: frozenset = frozenset()
        current_votes: list = []
        
        # try:
        while not env.done:
            phase = env.get_phase()[0]
            phase_name = {0: "Selection", 1: "Team Voting", 2: "Quest Voting", 3: "Assassination"}.get(phase, "Unknown")
            print(f"[Game {index}] Mission {env.turn}, Round {env.round} - {phase_name} Phase")
            get_game_logger().info(f"##### Mission {env.turn}, Round {env.round} #####")
            
            # if phase is team selection phase, ask for team
            if phase == 0:
                leader = env.get_quest_leader()
                current_leader = leader   # capture for round summary
                game_env_log.append(f"Selection Phase, the leader is Player {leader}")
                get_game_logger().info("##### System #####")
                get_game_logger().info(f"Selection Phase, the leader is Player {leader}")
                """
                Leader speaks & Discussion
                """
                speaking_order = []
                private_informations = []
                roles = []
                # intended_team_list = []
                if self.discussion:
                    get_game_logger().info("##### Discussion Starts #####")
                    # dialogue_history: list[tuple[int, str]] = []
                    # Leader speaks
                    summaries = []
                    proxy.set_current_agent(leader)
                    for idx, player in enumerate(player_list):
                        if hasattr(player, 'summarize'):
                            # Skip summarization on the very first round (Mission 0, Round 0) 
                            # because there is no game history yet, which causes the LLM to hallucinate past rounds.
                            if env.turn == 0 and env.round == 0:
                                continue
                            
                            proxy.set_current_agent(idx)
                            try:
                                summary_item = await player.summarize(
                                env=env, 
                                round_num=env.round, 
                                mission_id=env.turn,
                                log_snapshot=(idx in self.log_memory_snapshots_for)
                            )
                            except Exception as e:
                                import traceback
                                with open('/nas/longleaf/home/anvu/Avalon/Avalon-Reputation/logs/crash_log.txt', 'a') as crash_f:
                                    crash_f.write(f'CRASH IN SUMMARIZE P{idx}: {e}\n{traceback.format_exc()}\n')
                                raise e
                            if hasattr(player, 'periodic_predict') and idx in self.predict_for:
                                await player.periodic_predict(round_num=env.round, mission_id=env.turn)
                            if summary_item and len(summary_item) > 0:
                                last_item = summary_item[-1]
                                content = last_item.get('content', '') if isinstance(last_item, dict) else getattr(last_item, 'content', '')
                                summaries.append(str(content))
                    # print("Test: ", player_list[leader].team_discussion)
                    # team, statement = await player_list[leader].test()
                    # print(leader)
                    # print(player_list[leader].team_discussion)
                    for _ in range(self.num_discussion_rounds):
                        if hasattr(player_list[leader], 'team_discussion'):
                            proxy.set_current_agent(leader)
                            dialogue = await player_list[leader].team_discussion(
                                    team_size           =   env.get_team_size(),
                                    team_leader_id      =   leader,
                                    mission_id          =   env.turn,
                                    env                 =   env,
                                    dialogue_history    =   dialogue_history,
                                )
                            if dialogue is not None:
                                if isinstance(dialogue, dict):
                                    dialogue = dialogue.get('content', '')
                                else:
                                    dialogue = getattr(dialogue, 'content', str(dialogue))
                                get_game_logger().info(f"Player {leader}(Leader): {dialogue}")
                                roles.append(player_list[leader].role)
                                dialogue_history.append(leader, dialogue)
                                discussion_history.append(f"Player {leader}:\n{dialogue}")
                        speaking_order.append(leader)
                        private_informations.append(getattr(player_list[leader], 'system_info', ''))
                        # intended_team = await player_list[leader].propose_team(
                        #     team_size           =   env.get_team_size(),
                        #     mission_id          =   env.turn,
                        #     env                 =   env,
                        # )
                        # intended_team_list.append(list(intended_team))

                        # Discussion (sequential, once, in order for now) and Summarize
                        for idx in range(leader+1, leader + num_players):
                            player_id = idx % num_players
                            player = player_list[player_id]
                            if hasattr(player, 'team_discussion'):
                                proxy.set_current_agent(player_id)
                                dialogue = await player.team_discussion(
                                    team_size           =   env.get_team_size(),
                                    team_leader_id      =   leader,
                                    mission_id          =   env.turn,
                                    dialogue_history    =   dialogue_history,
                                    env                 =   env,
                                )
                                if dialogue is not None:
                                    if isinstance(dialogue, dict):
                                        dialogue = dialogue.get('content', '')
                                    else:
                                        dialogue = getattr(dialogue, 'content', str(dialogue))
                                    get_game_logger().info(f"Player {player_id}: {dialogue}")
                                    roles.append(player.role)
                                    dialogue_history.append(player_id, dialogue)
                                    discussion_history.append(f"Player {player_id}:\n{dialogue}")
                            
                            speaking_order.append(player_id)
                            private_informations.append(getattr(player, 'system_info', ''))

                    # query the intended teams after discussion
                    # for idx, player in enumerate(player_list):
                    #     proxy.set_current_agent(idx)
                    #     if idx == leader:
                    #         continue
                    #     intended_team = await player.propose_team(
                    #         team_size           =   env.get_team_size(),
                    #         mission_id          =   env.turn,
                    #         env                 =   env,
                    #     )
                    #     intended_team_list.append(list(intended_team))
                    # for idx, player in enumerate(player):
                    #     proxy.set_current_agent(idx)
                    #     player.discussion_end(
                    #         leader              =   leader,
                    #         leader_statement    =   statement,
                    #     )
                    # for idx, player in enumerate(player_list):
                    
                    # print(AvalonState.init_from_env(env).get_state_tuple())
                    # print(dialogue_history.dialogue_tuple_to_list())
                    # self.data_loader.add_data_point(
                    #     discussion_history_summary=summaries,
                    #     state_info=AvalonState.init_from_env(env).get_state_tuple(),
                    #     intended_actions=intended_team_list,
                    #     private_informations=private_informations,
                    #     roles=roles,
                    #     dialogue=dialogue_history.dialogue_tuple_to_list(),
                    #     speaking_order=speaking_order,
                    # )
                    # self.data_loader.save_data(self.FILE_PATH)
                # Choose a team
                # print(player_list[leader].propose_team)
                    get_game_logger().info("##### Discussion Ends #####")
                proxy.set_current_agent(leader)
                team = await player_list[leader].propose_team(
                    team_size           =   env.get_team_size(),
                    mission_id          =   env.turn,
                    env                 =   env,
                )
                current_team = team   # capture for round summary
                env.choose_quest_team(
                    team   =  frozenset(team),
                    leader =  leader
                )
                game_env_log.append(f"Leader Player {leader} chooses team {list(team)}")
                get_game_logger().info("##### System #####")
                get_game_logger().info(f"Leader Player {leader} chooses team {list(team)}")

            # if phase is team voting phase, ask for votes
            elif phase == 1:
                game_env_log.append("Team Voting Phase")
                get_game_logger().info("##### System #####")
                get_game_logger().info("Team voting Phase")
                votes = []
                proxy.set_current_agent(0)
                for i in range(num_players):
                    proxy.set_current_agent(i)
                    vote = await player_list[i].vote_on_team(
                        team                =   env.get_current_quest_team(),
                        mission_id          =   env.turn,
                        env                 =   env,
                        )
                    votes.append(vote)
                current_votes = votes   # capture for round summary
                try:
                    outcome = env.gather_team_votes(votes)
                except Exception as e:
                    get_game_logger().warning(f"Warning: gather_team_votes failed: {e}, defaulting to rejection")
                    # Default to all-reject outcome: (False, votes, False, 0)
                    outcome = (False, votes, False, 0)
                game_env_log.append(f"Team votes at this round: {str(votes)}")

                # Observe results of Team Selection
                for idx, player in enumerate(player_list):
                    proxy.set_current_agent(idx)
                    await player.observe_team_result(
                        mission_id  =   env.turn,
                        team        =   env.get_current_quest_team(),
                        votes       =   votes,
                        outcome     =   outcome[2],
                    )

                game_env_log.append("Team result: " + verbalize_team_result(team=env.get_current_quest_team(), votes=votes, outcome=outcome[2]))
                get_game_logger().info("##### System #####")
                get_game_logger().info("Team result: " + verbalize_team_result(team=env.get_current_quest_team(), votes=votes, outcome=outcome[2]))

                # Update reputation memory after a REJECTED team vote
                if not outcome[2] and self.use_reputation_memory:
                    proxy.set_current_agent(llm_idx)
                    round_summary = build_round_summary(
                        leader=current_leader,
                        team=current_team,
                        votes=current_votes,
                        outcome=False,
                        discussion_history=discussion_history,
                        quest_result=None,
                        num_fails=None,
                    )
                    await player_list[llm_idx].update_reputation_memory(
                        round_summary=round_summary,
                        round_num=env.round,
                    )
                    discussion_history.clear()


            # if phase is quest voting phase, ask for votes
            elif phase == 2:
                game_env_log.append("Quest Voting Phase")
                get_game_logger().info("##### System #####")
                get_game_logger().info("Quest Voting Phase")
                '''
                TODO: Can have a discussion before voting on quest
                '''
                votes = []
                for i in env.get_current_quest_team():
                    proxy.set_current_agent(i)
                    vote = await player_list[i].vote_on_mission(
                        team        =   env.get_current_quest_team(),
                        mission_id  =   env.turn,
                        env         =   env,
                    )
                    votes.append(vote)
                outcome = env.gather_quest_votes(votes)
                game_env_log.append(f"Quest votes at this round: {str(votes)}")

                # Observe mission/quest result
                proxy.set_current_agent(0)
                for idx, player in enumerate(player_list):
                    proxy.set_current_agent(idx)
                    await player.observe_mission(
                        team        =   env.get_current_quest_team(),
                        mission_id  =   env.turn-1,
                        num_fails   =   outcome[3],
                        votes       =   votes,
                        outcome     =   outcome[2],
                        env         =   env,
                    )

                game_env_log.append("Quest result: " + verbalize_mission_result(team=env.get_current_quest_team(), outcome=outcome[2]))
                get_game_logger().info("##### System #####")
                get_game_logger().info("Quest result: " + verbalize_mission_result(team=env.get_current_quest_team(), outcome=outcome[2]))

                # Update reputation memory after a completed quest
                if self.use_reputation_memory:
                    proxy.set_current_agent(llm_idx)
                    round_summary = build_round_summary(
                        leader=current_leader,
                        team=current_team,
                        votes=current_votes,
                        outcome=True,
                        discussion_history=discussion_history,
                        quest_result=outcome[2],
                        num_fails=outcome[3],
                    )
                    await player_list[llm_idx].update_reputation_memory(
                        round_summary=round_summary,
                        round_num=env.turn - 1,
                    )
                    discussion_history.clear()

            
            # if phase is assassination phase, ask for assassination
            elif phase == 3:
                game_env_log.append("Assassination phase")
                get_game_logger().info("##### System #####")
                get_game_logger().info("Assassination phase")
                '''
                    TODO: Discussion before Assassination Phase
                '''
                # assassin = env.get_assassin()
                assassin = None
                for idx, player in enumerate(player_list):
                    if player.role == 7:
                        assassin = idx
                if assassin is None:
                    get_game_logger().warning("Warning: No Assassin found in player_list, defaulting to Player 0")
                    assassin = 0
                proxy.set_current_agent(assassin)
                target = int(await player_list[assassin].assassinate(
                    env=env,
                    ))
                _, _, assassinated = env.choose_assassination_target(assassin, target)
                game_env_log.append(f"Assassin Player {assassin} chooses to assassinate Player {target}")
                get_game_logger().info("##### System #####")
                get_game_logger().info(f"Assassin Player {assassin} chooses to assassinate Player {target}")
        # reflect sides of each player at the end of the game
        for idx, player in enumerate(player_list):
            proxy.set_current_agent(idx)
            if idx == llm_idx:
                llm_believed_player_sides, llm_believed_merlin_sides = await player.get_believed_sides(
                    num_players =   self.num_players,
                    env         =   env,
                    )

                true_player_sides.append(list(map(int, env.is_good)))
                believed_player_sides.append(llm_believed_player_sides)
                believed_merlin_sides.append(llm_believed_merlin_sides)

        if env.good_victory:
            answer = 1
        else:
            if sum(env.quest_results) >= 3:
                answer = 0
            else:
                answer = -1
        finish_reason = SampleStatus.COMPLETED

        # except AgentContextLimitException as e1:
        #     return TaskSampleExecutionResult(status=SampleStatus.AGENT_CONTEXT_LIMIT)
        # except AvalonAgentActionException as e2:
        #     return TaskSampleExecutionResult(status=SampleStatus.AGENT_INVALID_ACTION, result={"result": False, "error": e2})
        # except Exception as e:
        #     finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
        #     return TaskSampleExecutionResult(status=finish_reason, result={"result": False, "error": e})
        
        verbal_game_result = {
            -1: "Evil wins by mission!",
            0: "Evil wins by assassination!",
            1: "Good wins!"
        }
        
        get_game_logger().info("##### Game Over #####")
        get_game_logger().info(f"Result: {verbal_game_result[answer]}")
        
        # Identify the true Merlin player ID for post-hoc detection accuracy
        true_merlin_id = next(
            (i for i, (_, role_name, _) in enumerate(env.get_roles()) if role_name == "Merlin"),
            None
        )

        result_dict = {
            "game_result": verbal_game_result[answer],
            "llm_idx": llm_idx,
            f"role_of_Player_{llm_idx}": player_list[llm_idx].role_name,
            f"Player_{llm_idx}_wins": (answer > 0) == bool(player_list[llm_idx].side),
            f"Player_{llm_idx}_deduc_acc": scoring.deduction_acc(true_player_sides, believed_player_sides) if true_player_sides else 0.0,
            f"Player_{llm_idx}_merlin_pred": believed_merlin_sides[0] if believed_merlin_sides else {},
            "true_merlin_id": true_merlin_id,
            "game_env_log": game_env_log,
        }

        # Write periodic prediction snapshots for all players in predict_for
        for pid in self.predict_for:
            if pid < len(player_list):
                player = player_list[pid]
                if hasattr(player, 'previous_prediction') and player.previous_prediction:
                    result_dict[f"Player_{pid}_periodic_good_pred"] = player.previous_prediction
                if hasattr(player, 'previous_merlin_prediction') and player.previous_merlin_prediction:
                    result_dict[f"Player_{pid}_periodic_merlin_pred"] = player.previous_merlin_prediction
        
        for i in range(self.num_players):
            result_dict[f"history for player {i}"] = proxy.history[i]

        if self.long_term_memories:
            for pid, ltm in self.long_term_memories.items():
                player_won = (answer > 0) == bool(player_list[pid].side)
                lesson = await self._run_game_critique(sessions[pid], player_list[pid], env, result_dict, observer_id=pid)
                ltm.add_lesson(lesson, won=player_won)

        return TaskSampleExecutionResult(status=finish_reason, result=result_dict)

    async def _run_game_critique(self, session, agent0, env, result_dict, observer_id: int = 0):
        true_roles_lines = [
            f"Player {i}: {role_name} ({'Good' if is_good else 'Evil'})"
            for i, (_, role_name, is_good) in enumerate(env.get_roles())
        ]
        true_roles = "\n".join(true_roles_lines)

        game_outcome = (
            f"{result_dict['game_result']}\n"
            f"Player {observer_id} (role: {result_dict.get(f'role_of_Player_{observer_id}', '?')}) "
            f"{'won' if result_dict.get(f'Player_{observer_id}_wins') else 'lost'}."
        )

        game_env_log = "\n".join(result_dict["game_env_log"]) if result_dict["game_env_log"] else "(No events recorded)"
        
        round_summaries = "\n\n".join(agent0.summaries_log) if agent0.summaries_log else "(No summaries recorded \u2014 game may have ended before round 1)"
        
        prediction_changes = "\n\n".join(agent0.prediction_changes_log) if agent0.prediction_changes_log else "(No prediction changes recorded)"

        critique_prompt = LONG_TERM_CRITIQUE_PROMPT.format(
            true_roles=true_roles,
            game_outcome=game_outcome,
            game_env_log=game_env_log,
            round_summaries=round_summaries,
            prediction_changes=prediction_changes
        )

        past_history = list(session.get_history())
        session.inject({"role": "user", "content": critique_prompt})
        try:
            response = await session.action()
            lesson = response.content if hasattr(response, 'content') else (response if isinstance(response, str) else str(response))
            get_game_logger().info(f"##### [LTM Critique] #####\n{lesson}")
            return lesson
        except Exception as e:
            get_game_logger().warning(f"[LTM Critique] LLM call failed: {e}")
            return ""
        finally:
            session.overwrite_history(past_history)