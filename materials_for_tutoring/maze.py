# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')
from scipy import *
import sys, time
import pylab as py

from pybrain.rl.environments.mazes import Maze, MDPMazeTask
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment
from pybrain.rl.environments import Task

class MyMaze(Maze):
    # NOT WORKING
    def __init__(self, topology, goal, antiGoal, **args):
        self.angryMinotaur = antiGoal
        super(MyMaze, self).__init__(topology, goal, **args)

class MyTask(MDPMazeTask):
    # To change default reward parameters
    def getReward(self):
        """ compute and return the current reward (i.e. corresponding to the last action performed) """
        if self.env.goal == self.env.perseus:
            self.env.reset()
            reward = 1.
        else:
            reward = -0.02 
        return reward

structure = array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 1, 0, 0, 1, 0, 1],
                   [1, 0, 0, 1, 0, 0, 1, 0, 1],
                   [1, 0, 0, 1, 0, 1, 1, 0, 1],
                   [1, 0, 0, 0, 0, 0, 1, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

environment = MyMaze(structure, (7, 7), (1,7))
print MyMaze.goal
print MyMaze.angryMinotaur
exit()


environment = Maze(structure, (7, 7))

controller = ActionValueTable(81, 4)
controller.initialize(1.)

learner = Q()
agent = LearningAgent(controller, learner)

task = MyTask(environment)
experiment = Experiment(task, agent)

py.gray()
py.ion()

for i in range(120):
    experiment.doInteractions(100)
    agent.learn()
    agent.reset()
 
    py.pcolor(controller.params.reshape(81,4).max(1).reshape(9,9))
    py.draw()
