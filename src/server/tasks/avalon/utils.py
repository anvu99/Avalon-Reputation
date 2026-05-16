import re
import logging
import contextvars
import os
from contextlib import asynccontextmanager

game_logger_var = contextvars.ContextVar('game_logger')

def get_game_logger():
    try:
        return game_logger_var.get()
    except LookupError:
        return logging.getLogger("avalon_fallback")

@asynccontextmanager
async def game_logger_context(index, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"game_{index}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, f"game_{index}.log"), mode='w')
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    token = game_logger_var.set(logger)
    try:
        yield logger
    finally:
        game_logger_var.reset(token)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
def get_vote_result(answer: str):
    answer_clean = answer.strip()
    
    match = re.search(r'Decision:\s*(Yes|No)', answer_clean, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
        
    if answer_clean.lower().startswith("yes"): return "Yes"
    if answer_clean.lower().startswith("no"): return "No"
    
    # Remove the template if the LLM hallucinated it to prevent falsely extracting 'No'
    answer_clean = answer_clean.replace("{Yes|No}", "").replace("{yes|no}", "")
    
    match_vote = "Yes|No"
    vote_result = re.findall(match_vote, answer_clean, re.IGNORECASE)

    if len(vote_result) == 0: 
        return ''
    
    return vote_result[-1].capitalize()

def get_team_result(answer: str):
    match_num = r"\d+"
    player_list = []
    
    player_list = re.findall(match_num, answer)

    player_list = [int(id) for id in player_list]

    return player_list

def get_assassination_result(message: str, answer: str): 
    match_num = r"\d+"
    player_id = []
        
    player_id = re.findall(match_num, str(message)+str(answer)) 

    player_id = int(player_id[-1])

    return player_id

def get_believed_player_sides(answer):
    try:
        match_good = re.search(r'Answer:\s*({[^}]+})', answer)
        scores = eval(match_good.group(1)) if match_good else eval(answer.split("Answer: ")[-1].split("\n")[0])
    except:
        scores = {}
        
    merlin_scores = {}
    try:
        match_merlin = re.search(r'Merlin:\s*({[^}]+})', answer)
        if match_merlin:
            merlin_scores = eval(match_merlin.group(1))
    except:
        pass
        
    return scores, merlin_scores

def verbalize_team_result(team: frozenset, votes, outcome: bool):
    verbal_vote = {
        0: "reject",
        1: "approve"
    }
    verbalized_result = ""
    if outcome == True:
        verbalized_result = f"The team {str(list(team))} is approved."
    elif outcome == False:
        verbalized_result = f"The team {str(list(team))} is rejected."
    else:
        raise ValueError("Invalid outcome %s" % outcome)
    
    for idx, vote in enumerate(votes):
        verbalized_result += " Player %d voted %s." % (idx, verbal_vote[vote])
    
    return verbalized_result

def verbalize_mission_result(team: frozenset, outcome: bool):
    verbalized_result = ""
    if outcome == True:
        verbalized_result = "The mission succeeded."
    elif outcome == False:
        verbalized_result = "The mission failed."
    else:
        raise ValueError("Invalid outcome %s" % outcome)
    
    verbalized_result += " The team is %s, which contains" % str(list(team))
    for member in team:
        verbalized_result += " Player %s" % str(member)

    return verbalized_result

def slice_out_new_dialogue(dialogue_history: list[tuple[int, str]], player: int) -> list[tuple[int, str]]:
    '''
    Returns a list of dialogue that is new to the player (i.e. from the last time the player spoke, or the beginning of the dialogue if the player has not spoken yet)
    '''
    new_dialogue = dialogue_history
    for i, (speaker, utterance) in enumerate(dialogue_history):
        if speaker == player:
            new_dialogue = dialogue_history[i+1:]
            break
    return new_dialogue