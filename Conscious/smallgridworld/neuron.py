import numpy as np
import math
from smallgridworld.env_small_demo import GridWorldTiny

env = GridWorldTiny()
neurons = []


class Chunk:
    def __init__(self, address, info, strength, intensity, emotion):
        self.emotion = emotion
        self.weight = strength
        self.intensity = intensity
        self.gist = info
        self.address = address

    @staticmethod
    def compete(a, b):
        # f: chunk -> |weight|
        # it is unlikely in this environment that a "+10*1 and -1*90" scenario happens.
        if abs(a.weight) > abs(b.weight):
            return Chunk(a.address, a.gist, a.weight, a.intensity + b.intensity, a.emotion + b.emotion)
        else:
            return Chunk(b.address, b.gist, b.weight, a.intensity + b.intensity, a.emotion + b.emotion)

class Neuron:
    def __init__(self, n, num, initial_strength):
        self.n, self.id = n, num
        self.matrix = np.zeros((n, n, 3))
        self.bias = np.random.random() - 0.5
        self.strength = initial_strength
        a, b = np.random.random(), np.random.random()
        self.mood = b - 0.5 # mood is between (-0.5, 0.5)
        self.fixed_mood = True  # if false, then the mood is influenced by links
        self.age = 0

    def try_activate(self, obs, mask=None, gists=None, modifier=1):
        N, M, tot_fired = obs.shape[0], obs.shape[1], 0
        if mask is None: mask = np.ones_like(obs)
        s = np.sum(mask * (self.matrix * obs + self.bias * np.sum(mask)))
        if s > 0: chunk = self.generate_chunk(self.strength * modifier, gists)
        else: chunk = (0, "null chunk") # null chunk
        return tot_fired, chunk

    def generate_chunk(self, strength, info):
        return Chunk(self.id, info, strength, abs(strength), self.mood)


class Args:
    def __init__(self):
        self.decay_factor = 0.95
        self.new_neurons_per_turn = 5
        self.new_mutation = 5
        self.neuron_upper_limit = 128
        self.strength_upper_limit = 10
        self.recover_factor = 2
        self.CTM_temporal_decay = 0.1
        self.mutation_ratio = 0.05
        self.link_frequency_per_step = 10
        self.eval_steps = 15

args = Args()

def generate_new_neurons(num, n):
    neuron = Neuron(n, num, 1)
    neuron.matrix = np.random.random((n, n, 3)) - 0.5
    neuron.bias = np.random.random((n, n, 3)) - 0.5
    return neuron


if __name__ == "__main__":
    env.reset()
    for t in range(1000):
        obs, mp = env.get_array_obs(), env.get_observation()
        print(mp.shape)
        new_neurons = []
        for i in range(len(neurons)):
            # try to activate
            tot_fired, chunks = neurons[i].try_activate(mp)
            neurons[i].strength *= args.decay_factor / (args.decay_factor + neurons[i].age * 0.01) # will decay faster according to age
            neurons[i].strength = min(neurons[i].strength, args.strength_upper_limit)
            neurons[i].age += 1

            if neurons[i].strength >= 0.05:
                new_neurons.append(neurons[i])
        n = len(new_neurons)
        for i in range(min(args.new_neurons_per_turn, args.neuron_upper_limit - n)):
            new_neurons.append(generate_new_neurons(len(neurons), env.N))
        neurons = new_neurons
        # print([neurons[i].strength for i in range(len(neurons))])
        env.visualize()
        env.move(0)