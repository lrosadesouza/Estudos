#CS 138 - RL Coding Assignment 1 - K armed Bandit Problem
# Author: Lucas Rosa de Souza
# 09/26/2022
import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt

class action:
    STEP = 0.1
    realValue = 0
    estValue = 0
    nTimesTaken = 0
    
    def __init__(self, initialValue = 0):
        self.estValue = initialValue
        
    def __str__(self):
        return "{}".format(self.estValue)
    
    #Incremental sample-averaging value estimation
    def sampleAverageEstimation(self, reward):
        self.nTimesTaken+=1
        self.estValue = self.estValue + (1/self.nTimesTaken)*(reward - self.estValue)
    
    #Constant step-size value estimation   
    def constantStepEstimation(self, reward):
        self.estValue = self.estValue + self.STEP*(reward - self.estValue)
    
    @property    
    def getReward(self):
        return default_rng().normal(self.realValue, 1.0)
    #Random Walk
    def updateRealValue(self):
        self.realValue += default_rng().normal(0, 0.01)

        
class BanditProblem:
    EPSILON = 0.1
    actions = None
    
    def __init__(self):
        self.actions = [action() for i in range(10)]        
    
    #returns list of estimated values for all actions
    @property
    def estimatedActionValues(self): 
        return [x.estValue for x in self.actions]
    
    #returns list of real values for all actions
    @property
    def realActionValues(self):
        return [x.realValue for x in self.actions]        
    
    #returns index of greedy choice
    @property
    def greedyChoice(self):
        return self.estimatedActionValues.index(max(self.estimatedActionValues))
    
    #returns index of optimal action
    @property
    def optimalAction(self):
        return self.realActionValues.index(max(self.realActionValues))   

    #This method returns index of action chosen by E-greedy action-choice
    def chooseActionEpsilonGreedy(self): 
        #If probability < (1 - e) = 0.9, return index of greedy action
        if(default_rng().choice([1, 2], p=[(1 - self.EPSILON), self.EPSILON]) == 1):
            return self.greedyChoice
        else: #choose at random (with equal probability) among the available actions (Exploration)
            return self.estimatedActionValues.index(default_rng().choice(self.estimatedActionValues))
        
RUNS = 1000
STEPS = 10000

#First, sample-average methods          
sampAvgBandits = [BanditProblem() for x in range(RUNS)]
avgRewards = [0]*STEPS
optimalPercentage = [0]*STEPS

for bandit in sampAvgBandits:
    for i in range(STEPS):
        #on first run, sample all actions for initial value estimates
        if(i == 0):
            for a in bandit.actions: 
                a.updateRealValue()
                a.sampleAverageEstimation(a.getReward)
        #Use epsilon-greedy to choose action
        actionTaken = bandit.chooseActionEpsilonGreedy()
        #get reward
        reward = bandit.actions[actionTaken].getReward
        #update estimated value based on reward
        bandit.actions[actionTaken].sampleAverageEstimation(reward)

        avgRewards[i] += reward
        optimalPercentage[i] = (optimalPercentage[i]+1) if actionTaken == bandit.optimalAction else optimalPercentage[i]
        
        #perform random walk Initializing/updating q* values
        for a in bandit.actions:
            a.updateRealValue()
        
avgRewards = [x for x in map(lambda a : a/RUNS, avgRewards)]
optimalPercentage = [x*100 for x in map(lambda a : a/RUNS, optimalPercentage)]

plt.plot(avgRewards)
plt.ylim(-3, 3)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.show()


plt.plot(optimalPercentage)
plt.xlabel("Steps")
plt.ylabel("Percentage of Optimal Action Taken")
plt.ylim(0, 100)
plt.show()

#Second, constant step incrementation method (weighted average by recency)      

constStepBandits = [BanditProblem() for x in range(RUNS)]
avgRewards2 = [0]*STEPS
optimalPercentage2 = [0]*STEPS

for bandit in constStepBandits:
    for i in range(STEPS): 
        #on first run, sample all actions for initial value estimates
        if(i == 0):
            for a in bandit.actions: 
                a.updateRealValue()  
                a.constantStepEstimation(a.getReward)
        #Use epsilon-greedy to choose action
        actionTaken = bandit.chooseActionEpsilonGreedy()
        #get reward
        reward = bandit.actions[actionTaken].getReward
        #update estimated value based on reward
        bandit.actions[actionTaken].constantStepEstimation(reward)

        avgRewards2[i] += reward
        optimalPercentage2[i] = (optimalPercentage2[i]+1) if actionTaken == bandit.optimalAction else optimalPercentage2[i]
        
        #perform random walk Initializing/updating q* values
        for a in bandit.actions: 
            a.updateRealValue()
        
avgRewards2 = [x for x in map(lambda a : a/RUNS, avgRewards2)]
optimalPercentage2 = [x*100 for x in map(lambda a : a/RUNS, optimalPercentage2)]

plt.plot(avgRewards2)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.ylim(-3, 3)
plt.show()

plt.plot(optimalPercentage2)
plt.xlabel("Steps")
plt.ylabel("Percentage of Optimal Action Taken")
plt.ylim(0, 100)
plt.show()
    