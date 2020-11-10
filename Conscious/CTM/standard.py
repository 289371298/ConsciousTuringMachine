from encoder.random_neurons_test import *
import numpy as np
from copy import deepcopy
import time
class Leaf:
    def __init__(self, neuron):
        self.neuron = neuron
        self.temporal_strength = 1 # for DROP operator
        self.last_chunk = []
        # neuron firing and correspondingly adjusting weights is a variant of feedback process.
        # TODO: broadcast mechanism(combine with interrupt constant)
        # TODO: links and labels

class UpTree:
    def __init__(self, sight):
        # downtree is a broadcast which need not be explicitly built.
        self.tot_leaves = 1000
        self.leaf = [Leaf(generate_new_neurons(i)) for i in range(self.tot_leaves)]
        self.links = {} # when you do not have label, you do not need to have links.
        self.last_LTM = []
        self.attention = [1 for i in range(4)]
        self.interrupt_constant = 100
        self.st = 0

    def reset_step(self):
        self.st = 0

    def add_leaf(self, neuron):
        self.leaf.append(Leaf(neuron))

    def step(self, mp): # return action
        t = time.time()
        self.last_LTM, value = {}, {}
        for i in range(len(self.leaf)):
            self.leaf[i].last_chunk = []
            if self.leaf[i].temporal_strength < 1:
                self.leaf[i].temporal_strength = min(1, self.leaf[i].temporal_strength * args.recover_factor)
        mod = []
        for j in range(4): # process each direction
            chunks, gists = [], ["0","1","2","3"]
            for i in range(len(self.leaf)):
                _, new_chunk = self.leaf[i].neuron.try_activate(mp, mask=masks[j], gists=gists[j], modifier=self.leaf[i].temporal_strength * self.attention[j])
                for nc in new_chunk:
                    self.leaf[i].last_chunk.append(nc)
                    chunks.append(nc)
            while len(chunks) > 1:
                new_chunks = []
                for i in range(0, len(chunks), 2):
                    if i < len(chunks) - 1: new_chunk = Chunk.compete(chunks[i], chunks[i+1])
                    else: new_chunk = chunks[i]
                    new_chunks.append(new_chunk)
                chunks = new_chunks
            self.last_LTM[gists[j]] = chunks[0]
            # execute the final result. the result is an output of action;
            # go to the direction with the highest mood.
            mod.append(chunks[0].address)
        t = time.time() - t
        print(self.st, "winning neurons:", mod, "elapsed time:", t)
        for j in range(4):
            self.leaf[mod[j]].temporal_strength *= args.CTM_temporal_decay
        mx, rec = 0, 0
        for j in range(4):
            value[gists[j]] = self.last_LTM[gists[j]].emotion
            if mx < value[gists[j]]:
                mx, rec = value[gists[j]], j

        # the links weaken FIXME
        for i in range(len(self.links)):
            pass

        # build up/strengthen links
        for i in range(args.link_frequency_per_step):
            a, b = np.random.randint(len(self.leaf)), np.random.randint(len(self.leaf))
            if a == b: continue
            # try to build up links
            #FIXME: link not implemented
        self.st += 1
        # if negative value: go to the opposite direction
        if mx < 0: return (rec + 2) % 4, rec
        # else: goto that direction
        else: return rec, rec
        # fixme: ABOUT THE BORDER!

    def broadcast(self, most_notified):
        # when self.last_LTM is broadcast, it will focus on a particular direction.
        # an attention mechanism that gives higher weight to the current focusing direction.
        # TODO: output a meaningful gist.
        for i in range(4): self.attention[i] = 1
        self.attention[int(most_notified)] = 1.2
    """
    although the original CTM uptree is a pipeline, we implement a non-pipeline CTM uptree instead.
    The pipelined CTM, with a latency of h steps, is too slow to effectively respond to anything.
    """

class CTM:
    def __init__(self):
        self.sight = 11
        self.tree = UpTree(self.sight)


    def reset(self):
        self.tree.last_LTM = []
        self.tree.attention = [1 for i in range(4)]
        # initialize neurons

    def process_input(self, mp):
        action, most_notified = self.tree.step(mp)
        self.tree.broadcast(most_notified)
        #self.tree.label_propagation()
        return action

Machine = CTM()

class CTMPool:
    def __init__(self):
        self.N = 12
        self.pool = [CTM() for i in range(self.N)]
        self.env = GridWorld()

    def evaluate(self, x):
        # self.env.reset()
        for i in range(args.eval_steps):
            mp = self.env.get_local_observation()
            action = self.pool[x].process_input(mp[1])
            self.env.move(action)
            self.env.debug_show_graph()
        return self.env.hp

    def regeneration(self):
        self.env = GridWorld()
        self.env.reset()
        tmp_env = deepcopy(self.env)
        r = np.zeros((len(self.pool)))
        for i in range(len(self.pool)):
            self.env = deepcopy(tmp_env)
            r[i] = self.evaluate(i)
            print("r[",i,"] =", r[i])

        r -= r.max()
        s = 0
        for i in range(len(self.pool)):
            s += math.exp(r[i])
        ret = deepcopy(r)
        r = np.exp(r) / s
        for i in range(1, len(self.pool)):
            r[i] += r[i-1]
        p = np.random.random()
        rec = 0
        for i in range(len(self.pool)):
           if r[i] <= p: rec = i
        print("final decision:", rec)
        for i in range(len(self.pool)):
            if i == rec: continue
            self.pool[i] = deepcopy(self.pool[rec])
        return ret.sum() / self.N

    def mutation(self):
        for j in range(len(self.pool)//2):
            x = np.random.randint(4)
            if True: # replace neurons
                for i in range(args.new_mutation):
                    x = np.random.randint(len(self.pool))
                    for _ in range(args.new_neurons_per_turn):
                        j = np.random.randint(self.pool[i].tree.tot_leaves)
                        self.pool[x].tree.leaf[j] = Leaf(generate_new_neurons(j))
            else: # drop links
                pass # FIXME: not implementing links

pool = CTMPool()

left_mask, right_mask = np.zeros((Machine.sight, Machine.sight, 3)), np.zeros((Machine.sight, Machine.sight, 3))
up_mask, down_mask = np.zeros((Machine.sight, Machine.sight, 3)), np.zeros((Machine.sight, Machine.sight, 3))
for i in range(Machine.sight):
    for j in range(Machine.sight):
        if i >= j and i <= Machine.sight - j - 1: left_mask[i, j, :] = 1
        if i >= j and i >= Machine.sight - j - 1: down_mask[i, j, :] = 1
        if j >= i and i <= Machine.sight - j - 1: up_mask[i, j, :] = 1
        if j >= i and i >= Machine.sight - j - 1: right_mask[i, j, :] = 1
masks = [left_mask, right_mask, up_mask, down_mask]
gists = ['left', 'right', 'up', 'down']
dx = [(0, -1), (0, 1), (-1, 0), (1, 0)]

if __name__ == "__main__":
    f = open("rew.txt", "w")
    while True:
        avg_reward = pool.regeneration()
        print("reward = ",avg_reward)
        f.write(str(avg_reward)+"\n")
        f.flush()
        pool.mutation()


