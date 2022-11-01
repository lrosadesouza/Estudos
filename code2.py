import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from pprint import pprint
import seaborn as sns
import copy

# EPSILON = 0.1
GAMMA = 1
#Actions are defined as tuples of increments/decrements to velocity
actions = [(0,0), (1,0), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1)]

track = np.genfromtxt('A2/Racetrack1', delimiter = ',')
#qTable[state] = Action : Value
qTable = dict()
returns = dict()

INITIAL_ROW = 35
INITIAL_COL_RANGE = [6, 12]


class State: 
    pos = None
    vel = None
    #default values for start rows
    def __init__(self, pos = (0,0), vel = (0,0)): 
        if (not isinstance(pos,tuple)) or (not isinstance(vel,tuple)):
            raise Exception("Position and velocities must be tuples")
        if pos == (0,0):
            self.pos = (INITIAL_ROW, default_rng().choice([x for x in range(INITIAL_COL_RANGE[0], INITIAL_COL_RANGE[1] + 1)]))
        else: self.pos = pos
        self.vel = vel
    #__eq__ and __hash__ defined in order to use states as keys for mapping    
    def __eq__(self, __o: object) -> bool: 
        return (self.pos == __o.pos) and self.vel == __o.vel
    
    def __hash__(self) -> int:
        return hash(((self.pos),(self.vel)))


def rewardFunction(nextPos):
    if (track[nextPos[0]][nextPos[1]] == 0):
        return -100
    if((track[nextPos[0]][nextPos[1]] == 1) or (track[nextPos[0]][nextPos[1]] == 3)):
        return -1
    if((track[nextPos[0]][nextPos[1]] == 2)):
        return 0
    
    
def chooseActionEpsilonGreedy(st, EPSILON):
    if not isinstance(st, State):
        raise Exception("state must be instance of State")
    choice = default_rng().choice([0,1], p=[(1-EPSILON), EPSILON]) #Random choice based on probabilities 1-Epsilon and Epsilon
    if choice == 0: #greedy choice
        maxValue = -100000
        action = tuple()
        for key in qTable[st]:
            #choose maximum among valid actions, ties are broken arbitrarily
            if ((st.vel[0]+ key[0], st.vel[1] + key[1]) != (0 , 0)) and (st.vel[0]+ key[0]) < 5 and (st.vel[1] + key[1]) < 5 and (st.vel[0]+ key[0]) >= 0 and (st.vel[1] + key[1]) >= 0:
                if (qTable[st][key] > maxValue):
                    maxValue = qTable[st][key]
                    action = key
                elif qTable[st][key] == maxValue:
                    action = action if default_rng().choice([0, 1], p = [0.5,0.5]) == 0 else key 
        return action
    else:
        while(True):
            key = default_rng().choice(actions)
            if ((st.vel[0]+ key[0], st.vel[1] + key[1]) != (0 , 0)) and (st.vel[0]+ key[0]) < 5 and (st.vel[1] + key[1]) < 5 and (st.vel[0]+ key[0]) >= 0 and (st.vel[1] + key[1]) >= 0:
                action = key
                break
        return action
    
        
def getNextPos(state, action):
    return (state.pos[0] - (state.vel[0] + action[0]), state.pos[1] + (state.vel[1] + action[1]))


def generateEpisode(EPSILON):
    episode = list()
    temp = 0
    state = State() #initial state
    while(True):
        if(state not in qTable): #initializes estimated Q values of actions in state to 0 for states that had never been visited
            qTable[state] = {(0,0) : 0, (1,0): 0, (0,-1) :0, (1,0): 0, (1,1) :0 , (1,-1): 0, (-1,0): 0, (-1,1): 0, (-1,-1): 0}
        action = chooseActionEpsilonGreedy(state, EPSILON)
        episode.insert(0,(state,action)) #States in episode inserted stack-like
        nextPosition = getNextPos(state, action)
        #This version will keep the episode running when it hits the wall, like in the exercise 5.12 prompt
        if(rewardFunction(nextPosition) == 0):
            break
        if(rewardFunction(nextPosition) == -100):
            state = State()
#             nextPosition = (INITIAL_ROW, default_rng().choice([x for x in range(INITIAL_COL_RANGE[0], INITIAL_COL_RANGE[1] + 1)]))
#             state = State(nextPosition, (0,0))
        if(rewardFunction(nextPosition) == -1):
            state = State(nextPosition, (state.vel[0] + action[0], state.vel[1] + action[1]))
    print("Epsiode : ", episode)
    print("Epsiode : ", episode[0])
    return episode
            


def MC_Control(EPSILON):
    ep = None
    rewards = list()
    ep = copy.deepcopy(generateEpisode(EPSILON))
    
    episode2 = copy.deepcopy(ep)
    rewards = [rewardFunction(getNextPos(x[0], x[1])) for x in ep] #generating list of rewards from episode
    
    if(rewards[0] == 0):
        print("It FINISHED!!!!!!")
        print("Time to finish: {}".format(len(ep)))
        print(rewards)
    else:
        print("It did not Finish :/")
        print(rewards)
    
    G = 0
    while(ep):
        curStateActionPair = tuple()
        curStateActionPair = tuple(ep.pop(0))
        curState = curStateActionPair[0]
        curAction = tuple(curStateActionPair[1])
        if((curState, curAction) in ep):
            rewards.pop(0)
            continue
        G = GAMMA * G + rewards.pop(0)
        if((curState, curAction) not in returns):
            returns[(curState, curAction)] = list()
            
        returns[(curState, curAction)].append(G)
        #updating qtable with new avg        
        qTable[curState][curAction] = (sum(returns[(curState, curAction)])/len(returns[curState, curAction]))      
    return episode2
        
#Training phase: 
for i in range(10000):
    MC_Control(0.1)

print("******************************* turning off exploration -> Epsilon=0 *******************")

#Demonstration:
sns.set_theme()

#Generating an episode and plotting it using a heatmap:
ep = list()
ep = MC_Control(0)
track2 = np.copy(track)
for step in ep:
    track2[step[0].pos[0]][step[0].pos[1]] = 9
sns.heatmap(track2)

