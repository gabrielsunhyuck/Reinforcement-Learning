import random as rd
import numpy as np
from grid import *
from agent import *

def main():
    env   = Grid()
    agent = QAgent()

    for n_epi in range(1000):
        done    = False
        history = []

        s       = env.reset()

        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s, a, r, s_prime))
            s = s_prime

        agent.update_table(history)
        agent.anneal_epsilon()

    agent.show_table()