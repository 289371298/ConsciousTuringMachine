import numpy as np
import math
import cv2

class Animal:
    num = 0
    def __init__(self, pos):
        self.pos, self.id = pos, Animal.num
        self.color = [0xcc, 0xcc, 0x99]
        Animal.num += 1

    def __eq__(self, other):
        return self.id == other.id

class Monster:
    num = 0
    def __init__(self, pos):
        self.pos, self.id = pos, Monster.num
        self.color = [0x66, 0x66, 0x66]
        Monster.num += 1

    def __eq__(self, other):
        return self.id == other.id

class GridWorldTiny:
    def __init__(self):
        self.N = 8
        self.player_pos = (self.N // 2, self.N // 2)
        self.animals, self.monster = [], []
        self.hp, self.animal_cure, self.monster_harm = 0, 5, 20

    def reset(self):
        self.hp = 0
        self.player_pos = (self.N // 2, self.N // 2)
        self.animals = [Animal((np.random.randint(0, self.N) , np.random.randint(0, self.N))) for i in range(5)]
        self.monster = [Monster((np.random.randint(0, self.N), np.random.randint(0, self.N))) for i in range(3)]

    def move(self, action):
        dx, dy = [0, 1, 0, -1, 0], [-1, 0, 1, 0, 0]  # left, down, right, up, stay
        after_pos = (max(min(self.player_pos[0] + dx[action], self.N-1), 0), max(min(self.player_pos[1] + dy[action], self.N-1), 0))
        self.player_pos = after_pos
        bumped_into = []
        for i in range(len(self.animals)):
            seed = np.random.random()
            if seed >= 0.5:
                d = np.random.randint(4)
                print("animal", i, "moved!")
                after_pos = (max(min(self.animals[i].pos[0] + dx[d], self.N-1), 0), max(min(self.animals[i].pos[1] + dy[d], self.N-1), 0))
            else: after_pos = self.animals[i].pos
            self.animals[i].pos = after_pos
            if self.player_pos == after_pos:
                bumped_into.append(True)
                self.hp += self.animal_cure
            else: bumped_into.append(False)
        new_animals = []
        for i in range(len(self.animals)):
            if not bumped_into[i]: new_animals.append(self.animals[i])
        self.animals = new_animals
        for i in range(len(self.monster)):
            seed = np.random.random()
            if seed < 0.4:  # 怪物只有60%的概率移动
                d = np.random.randint(4)
                print("monster", i, "moved!")
                after_pos = (max(min(self.monster[i].pos[0] + dx[d], self.N-1), 0), max(min(self.monster[i].pos[1] + dy[d], self.N-1), 0))
            else: after_pos = self.animals[i].pos
            self.monster[i].pos = after_pos
            if self.player_pos == after_pos:
                self.hp -= self.monster_harm

    def get_array_obs(self):
        obs = np.zeros((self.N, self.N, 3))
        for i in range(len(self.animals)):
            X, Y = self.animals[i].pos
            obs[X, Y, 0] = 1
        for i in range(len(self.monster)):
            X, Y = self.monster[i].pos
            obs[X, Y, 1] = 1
        X, Y = self.player_pos
        obs[X, Y, 2] = 1
        return obs

    def get_observation(self):
        see_color = np.ones((self.N, self.N, 3)) * 0xee
        for i in range(len(self.animals)):
            X, Y = self.animals[i].pos
            see_color[X, Y, :] = self.animals[i].color
        for i in range(len(self.monster)):
            X, Y = self.monster[i].pos
            see_color[X, Y, :] = self.monster[i].color

        # color the player
        see_color[int(self.player_pos[0]), int(self.player_pos[1]), :] = np.array([255, 0, 0])
        return see_color

    def visualize(self, waitkey=False, image=None):
        if image is None: image = self.get_observation()
        new_image = np.zeros((image.shape[0] * 10, image.shape[1] * 10, 3))
        image[:, :, 2], image[:, :, 0] = image[:, :, 0], image[:, :, 2]
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(3):
                    new_image[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10, k] = image[i, j, k]
        cv2.imshow('map', new_image.astype('uint8'))
        if waitkey: cv2.waitKey(0)
        else: cv2.waitKey(100)