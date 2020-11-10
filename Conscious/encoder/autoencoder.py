import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from env.env_demo import GridWorld
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image as img
foods_low, foods_high = [0xcc, 0x99, 0x99], [0xff, 0xcc, 0xcc]
animals_low, animals_high = [0x66, 0x66 ,0x66], [0x99, 0x99, 0x99]
monsters_low, monsters_high = [0, 0, 0], [0x4b, 0x4b, 0x4b]

# not satisfying for generating neurons.

class autoencoder(nn.Module):
    def __init__(self, n, batch_size):
        super(autoencoder, self).__init__()
        self.n, self.batch_size = n, batch_size
        self.fc1 = nn.Linear(n*n*3, 120).double()
        self.fc2 = nn.Linear(120, 40).double()
        self.fc3 = nn.Linear(40, 4).double()
        self.fc4 = nn.Linear(4, 40).double()
        self.fc5 = nn.Linear(40, 120).double()
        self.fc6 = nn.Linear(120, n*n*3).double()
        self.op = Adam(self.parameters())

    def forward(self, x):
        x = torch.from_numpy(np.array(x)).reshape(-1, self.n * self.n * 3).double()
        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = F.relu(self.fc3(fc2))
        fc4 = F.relu(self.fc4(fc3))
        fc5 = F.relu(self.fc5(fc4))
        fc6 = F.relu(self.fc6(fc5))
        return fc6.reshape(-1, self.n, self.n, 3)

    def get_loss(self, output, target, self_pos, foods, animals, monsters): # adjusted position
        # note the presence of special points!
        # MSELoss
        target = torch.from_numpy(np.array(target))
        loss = 0.1 * ((output - target) ** 2).sum()
        # classify crucial points
        #self_pos
        #foods

        for idx in range(self.batch_size):
            # print(self_pos[idx][0], self_pos[idx][1])
            o = output[idx, self_pos[idx][0], self_pos[idx][1]]
            loss += ((target[idx, self_pos[idx][0], self_pos[idx][1]] - o) ** 2).sum()
            for i in range(len(foods[idx])):
                o = output[idx, foods[idx][i].pos[0], foods[idx][i].pos[1], :]
                for j in range(3):
                    if o[j] < foods_low[j]:
                        loss += ((foods_low[j] - o[j]) ** 2).sum()  # is the right category
                    elif o[j] > foods_high[j]:
                        loss += ((foods_high[j] - o[j]) ** 2).sum()
            for i in range(len(animals[idx])):
                for x in range(2):
                    for y in range(2):
                        if animals[idx][i][0] + x >= self.n or animals[idx][i][1] + y >= self.n: continue
                        o = output[idx, animals[idx][i][0] + x, animals[idx][i][1] + y, :]
                        for j in range(3):
                            if o[j] < animals_low[j]:
                                loss += ((animals_low[j] - o[j]) ** 2).sum()
                            elif o[j] > animals_high[j]:
                                loss += ((animals_high[j] - o[j]) ** 2).sum()
            for i in range(len(monsters[idx])):
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if monsters[idx][i][0] + x >= self.n or monsters[idx][i][1] + y >= self.n: continue
                        if monsters[idx][i][0] + x < 0 or monsters[idx][i][1] + y < 0: continue
                        o = output[idx, monsters[idx][i][0] + x, monsters[idx][i][1] + y, :]
                        for j in range(3):
                            if o[j] < animals_low[j]:
                                loss += ((animals_low[j] - o[j]) ** 2).sum()
                            elif o[j] > animals_high[j]:
                                loss += ((animals_high[j] - o[j]) ** 2).sum()

        return loss

    def prepare_batch(self, batch):
        # real_LU, see_color, inner_animals, inner_monsters, inner_foods, self.player_pos
        mp, foods, animals, monsters, self_pos = [], [], [], [], []
        for i in range(self.batch_size):
            mp.append(batch[i][1])
            animals.append([])
            monsters.append([])
            foods.append([])
            for j in range(len(batch[i][2])):
                # animals[-1].append((batch[i][2][j].pos_LU[0] - batch[i][0][0], batch[i][2][j].pos_LU[1] - batch[i][0][1]))
                animals[-1].append(batch[i][2][j].pos_LU)
            for j in range(len(batch[i][3])):
                # monsters[-1].append((batch[i][3][j].pos_centre[0] - batch[i][0][0], batch[i][3][j].pos_centre[1] - batch[i][0][1]))
                monsters[-1].append(batch[i][3][j].pos_centre)
            for j in range(len(batch[i][4])):
                # foods[-1].append((batch[i][4][j].pos_centre[0] - batch[i][0][0], batch[i][4][j].pos_centre[1] - batch[i][0][1]))
                foods[-1].append(batch[i][4][j].pos)
            self_pos.append((int(batch[i][5][0] - batch[i][0][0]), int(batch[i][5][1] - batch[i][0][1]))) # real self_pos - real_LU
        return mp, foods, animals, monsters, self_pos

    def study(self, batch):
        mp, foods, animals, monsters, self_pos = self.prepare_batch(batch)
        output = AE(mp)
        loss = AE.get_loss(output, mp, self_pos, foods, animals, monsters)
        self.op.zero_grad()
        loss.backward()
        self.op.step()
        # print(loss.item())
        return loss.item(), mp[0], output[0]

env = GridWorld()
batch_size = 16
AE = autoencoder(env.player_sight, batch_size)
memory = []
if __name__ == "__main__":
    env.initialize()
    losses = []
    plt.ion()
    t = 0
    current = 0
    while True:
        batch = []
        for i in range(batch_size):
            local_obs = env.get_local_observation()
            local_obs = list(local_obs)
            local_obs[1] /= 256
            local_obs = tuple(local_obs)

            if len(memory) < 1000000:
                memory.append(local_obs)  # real_LU, see_color, inner_animals, inner_monsters, inner_foods, self.player_pos
            else: memory[current] = local_obs
            current = (current + 1) % 1000000
            env.move(0)
        idx = np.random.choice(len(memory), batch_size)
        batch = []
        for _ in range(batch_size): batch.append(memory[idx[_]])
        loss, mp, output = AE.study(batch)
        if t % 100 == 99:
            losses.append(loss)
            print(t)
            plt.subplot(221)
            mp[:, :, 0], mp[:, :, 2] = mp[:, :, 2], mp[:, :, 0]
            output = output.detach().numpy()
            output[:, :, 0], output[:, :, 2] = output[:, :, 2], output[:, :, 0]
            im = img.fromarray(np.uint8(mp*255))
            plt.imshow(im)
            plt.subplot(222)
            im = img.fromarray(np.uint8(output*255))
            plt.imshow(im)
            plt.subplot(212)
            plt.plot([j for j in range(99, t+1, 100)], losses)
            plt.show()
            plt.pause(0.001)
        t += 1