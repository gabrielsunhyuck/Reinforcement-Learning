from agent import *
from grid  import *

def main():
    for n_epi in range(1000):
        done = False
        s    = GridWorld.reset()
        
        while not done:
            a = QAgent.select_action(s)
            s_prime, r, done = GridWorld.step(a)
            QAgent.update_table((s, a, r, s_prime))
            s = s_prime
        QAgent.anneal_epsilon()

    QAgent.show_table()