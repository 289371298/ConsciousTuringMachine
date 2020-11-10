
        # 一个单元的基本属性包括：(R,G,B,sound)

        # 光源（暂时不加入）

        # 黑暗区（暂时不加入）

        # 草 静止 [00-33, CC-FF, 00-33]

        # 泥土 静止 [CC-FF, 88-AA, 33-66]

        # 岩石 静止 [96-DD, 96-DD, 96-DD]

        # 怪物 3*3 [00-4B, 00-4B, 00-4B] 会追击一定距离内的agent，否则随机游走。接触到怪物的agent会迅速减少生命值。怪物的移动速度比agent慢。怪物会产生很大的声音。

        # 食物 1*1 [CC-FF, 99-CC, 99-CC]。食物可以增加agent的生命值。

        # agent最终被选择繁衍下一代的概率是和生命值成正比的。

        # 每一代繁衍时，继承上一代的部分LTM。（如何保证关键的东西被保存下来？）

        # 动物 2*2 [66-99, 66-99, 66-99] 会在区块内随机游走。攻击动物（位置与动物重合）会令其消失并增加比食物更多的生命值。动物也会产生声音，但是音色和怪物不同。
import numpy as np
import math
import cv2

class Food:
            num = 0
            def __init__(self, pos):
                self.pos, self.id = pos, Food.num
                Food.num += 1
                self.color = [np.random.randint(0xcc, 0xff), np.random.randint(0x99, 0xcc), np.random.randint(0x99, 0xcc)]
            def __eq__(self, other):
                return self.id == other.id
class Animal:
            num = 0
            def __init__(self, pos):
                self.pos_LU, self.id = pos, Animal.num
                Animal.num += 1
                self.color = [np.random.randint(0x66, 0x99), np.random.randint(0x66, 0x99), np.random.randint(0x66, 0x99)]
            def __eq__(self, other):
                return self.id == other.id
class Monster:
            num = 0
            def __init__(self, pos):
                self.pos_centre, self.id = pos, Monster.num
                Monster.num += 1
                self.color = [np.random.randint(0x0, 0x4b), np.random.randint(0x0, 0x4b), np.random.randint(0x0, 0x4b)]
            def __eq__(self, other):
                return self.id == other.id

class GridWorld:
            def __init__(self):
                self.N = 63
                self.mp, self.color = np.zeros((self.N, self.N)), np.zeros((self.N, self.N, 3))
                self.animals, self.monster, self.food = [], [], []
                self.player_pos = (self.N//2, self.N//2)
                # below are params
                self.grass_mud_density, self.animal_density = 5, 48
                self.food_density, self.monster_density, self.monster_sight, self.player_sight = 50, 32, 32, 11
                self.hp, self.food_cure, self.animal_cure, self.monster_harm = 0, 1, 5, 20
                # monster sight is bigger than player.
                self.mp_local_obs = {
                    'real_LU': 0,
                    'see_color': 1,
                    'inner_animals': 2,
                    'inner_monsters': 3,
                    'inner_foods': 4,
                    'self_pos': 5
                }

            def reset(self):
                self.hp = 0
                self.player_pos = (self.N//2, self.N//2)
                self.initialize()

            def debug_show_graph(self, waitkey=False,image=None):
                # 展示当前地图
                if image is None: image = self.get_observation()
                new_image = np.zeros((image.shape[0]*4,image.shape[1]*4,3))
                image[:, :, 2], image[:, :, 0] = image[:, :, 0], image[:, : ,2]
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        for k in range(3):
                            new_image[i*4:(i+1)*4,j*4:(j+1)*4,k]=image[i,j,k]
                cv2.imshow('map', new_image.astype('uint8'))
                if waitkey: cv2.waitKey(0)
                else: cv2.waitKey(100)

            def initialize(self):
                self.mp, self.animals, self.monster, self.food, self.color = self.generate_mp(0, self.N-1, 0, self.N-1, ((self.N-1)//2, (self.N-1)//2))
                # self.debug_show_graph(waitkey=True)

            # 给一个方块确定颜色
            def coloring(self, H, W, mode):
                color = np.zeros((H, W, 3))
                assert mode in ['stone', 'grass', 'mud'], "Error!"
                if mode == 'stone': lowred, highred, lowgreen, highgreen, lowblue, highblue = 0x96, 0xDD, 0x96, 0xDD, 0x96, 0xDD
                elif mode == 'grass': lowred, highred, lowgreen, highgreen, lowblue, highblue = 0, 0x33, 0xCC, 0xFF, 0, 0x33
                else: lowred, highred, lowgreen, highgreen, lowblue, highblue = 0xCC, 0xFF, 0x88, 0xAA, 0x33, 0x66
                color[:, :, 0] = np.random.randint(lowred, highred, size=(H, W))
                color[:, :, 1] = np.random.randint(lowgreen, highgreen, size=(H, W))
                color[:, :, 2] = np.random.randint(lowblue, highblue, size=(H, W))
                return color

            # 没有必要在这里就开始涂色；这里只负责产生地形和物品，最后在给出observation时统一渲染即可。
            def generate_mp(self, up, down, left, right, player_pos):  # [up, down] * [left, right]
                # people_pos is the corresponding position in this map.
                H, W = down - up + 1, right - left + 1
                new_mp = np.zeros((H, W))  # 一开始都是石头
                color_of_ground = self.coloring(H, W, 'stone')

                # mp 只包括基础的三种地形（泥土，草地，石头）
                # grass = 1, mud = 2, stone = 3
                # 在给定的四个坐标围成的矩形里生成新地形。
                for i in range(self.grass_mud_density):
                    # generate mud
                    len_x, len_y = np.random.randint(1, (H + 1)//2), np.random.randint(1, (W + 1)//2)
                    upleft_x, upleft_y = np.random.randint(H - len_x), np.random.randint(W - len_y)
                    # don't worry about time complexity; you will have to go O(n) steps to get a O(kn^2) cost, where k is grass_mud_density.
                    new_mp[upleft_x: upleft_x + len_x, upleft_y: upleft_y + len_y] = 2  # this will be reversed later.
                    color_of_ground[upleft_x: upleft_x + len_x, upleft_y: upleft_y + len_y] = self.coloring(len_x, len_y, 'mud')
                    # generate grass
                    len_x, len_y = np.random.randint(1, (H + 1)//2), np.random.randint(1, (W + 1)//2)
                    upleft_x, upleft_y = np.random.randint(H - len_x), np.random.randint(W - len_y)
                    # don't worry about time complexity; you will have to go O(n) steps to get a O(kn^2) cost, where k is grass_mud_density.
                    new_mp[upleft_x: upleft_x + len_x, upleft_y: upleft_y + len_y] = 1  # this will be reversed later.
                    color_of_ground[upleft_x: upleft_x + len_x, upleft_y: upleft_y + len_y] = self.coloring(len_x, len_y, 'grass')
                # 首先随机产生一些草和泥土，然后剩下的部分是石头。
                new_mp = 3 - new_mp  # vectorization!
                animals, monster, food = [], [], []
                for i in range(math.floor(self.food_density * (H * W) / (self.N * self.N))):
                    while True:
                        animal_LU = (np.random.randint(1, H), np.random.randint(1, W))
                        if animal_LU == player_pos: continue
                        animals.append(Animal(animal_LU))
                        break
                for i in range(math.floor(self.monster_density * (H * W) / (self.N * self.N))):
                    while True:
                        monster_centre = (np.random.randint(1, H), np.random.randint(1, W))
                        if monster_centre == player_pos: continue
                        monster.append(Monster(monster_centre))
                        break
                # 这里先不管碰撞的问题，假设动物和怪物都可以重叠。但是怪物总是在重叠的上层,而食物在更上层。玩家永远在重叠的顶层。
                # generate food
                for i in range(math.floor(self.food_density * (H * W) / (self.N * self.N))):
                    food_pos = (np.random.randint(1, H), np.random.randint(1, W))
                    food.append(Food(food_pos))
                return new_mp, animals, monster, food, color_of_ground

            def delete_all_items(self, u, d, l, r):
            #这些区域内的物品因为超出界限而被删除了。
                new_animals, new_monsters, new_foods = [], [], []
                for i in range(len(self.animals)):
                    animal = self.animals[i]
                    if animal.pos_LU[0] >= u and animal.pos_LU[0] <= d and animal.pos_LU[1] >= l and animal.pos_LU[1] <= r:
                        pass# print("animal removed!")
                    else: new_animals.append(animal)
                for i in range(len(self.monster)):
                    monster = self.monster[i]
                    if monster.pos_centre[0] >= u and monster.pos_centre[0] <= d and monster.pos_centre[1] >= l and monster.pos_centre[1] <= r:
                        pass# print("monster removed!")
                    else: new_monsters.append(monster)
                for i in range(len(self.food)):
                    food = self.food[i]
                    if food.pos[0] >= u and food.pos[0] <= d and food.pos[1] >= l and food.pos[1] <= r:
                        pass# print("food removed!")
                    else: new_foods.append(food)
                self.animals, self.monster, self.food = new_animals, new_monsters, new_foods
            def mov_items(self, dir):
                assert dir in ['left', 'right', 'up', 'down'], "invalid direction."
                direction = {'left': (0, (self.N + 1) // 2), 'right':(0, -(self.N+1)//2), 'up':((self.N + 1) // 2, 0), 'down':(-(self.N + 1) // 2, 0)}
                for i in range(len(self.animals)):
                    d = direction[dir]
                    self.animals[i].pos_LU = (self.animals[i].pos_LU[0] + d[0], self.animals[i].pos_LU[1] + d[1])
                for i in range(len(self.monster)):
                    d = direction[dir]
                    self.monster[i].pos_centre = (self.monster[i].pos_centre[0] + d[0], self.monster[i].pos_centre[1] + d[1])
                for i in range(len(self.food)):
                    d = direction[dir]
                    self.food[i].pos = (self.food[i].pos[0] + d[0], self.food[i].pos[1] + d[1])

            def mov_mp(self, dir):
                # 判断所处位置，移动地图
                # 这里两个轴是分开处理的，所以要假设每个时间步只能向四个方向移动1格。
                assert dir in ['left', 'right', 'up', 'down'], "invalid direction."
                # print("moving_map:", dir)
                a, b, c = None, None, None
                if dir == 'left':
                    assert self.player_pos[0] == (self.N - 1) // 2, "Error!"
                    # self.N = 255          127 + 1 : 255            0 : 127
                    self.mp[:, (self.N - 1) // 2 + 1:self.N] = self.mp[:, 0:(self.N - 1) // 2].copy()  # 抛弃右半边的地图
                    self.color[:, (self.N - 1)//2+1:self.N] = self.color[:, 0:(self.N - 1) // 2].copy()
                    self.delete_all_items(0, self.N - 1, (self.N - 1) // 2, self.N - 1)  # 127
                    # 0:128
                    self.mp[:, 0:(self.N - 1) // 2 + 1], a, b, c, self.color[:, 0:(self.N - 1) // 2 + 1] = self.generate_mp(0, self.N - 1, 0, (self.N - 1) // 2,
                                                                                   self.player_pos)
                elif dir == "right":
                    assert self.player_pos[0] == (self.N - 1) // 2, "Error!"
                    #          0 : 127
                    self.mp[:, 0:(self.N - 1) // 2] = self.mp[:, (self.N - 1) // 2 + 1:self.N].copy()
                    self.color[:, 0:(self.N - 1) // 2] = self.color[:, (self.N - 1) // 2 + 1:self.N].copy()
                    self.delete_all_items(0, self.N - 1, 0, (self.N - 1) // 2)
                    self.mp[:, (self.N - 1) // 2:self.N], a, b, c, self.color[:, (self.N - 1) // 2:self.N] = self.generate_mp(0, self.N - 1, (self.N - 1) // 2,
                                                                                    self.N - 1, (self.player_pos[0],
                                                                                                 0))  # player is on the left of the new map.
                elif dir == "up":
                    assert self.player_pos[1] == (self.N - 1) / 2, "Error!"
                    self.mp[(self.N - 1) // 2 + 1:self.N, :] = self.mp[0:(self.N - 1) // 2, :].copy()  # 抛弃右半边的地图
                    self.color[(self.N - 1) // 2 + 1:self.N, :] = self.color[0:(self.N - 1) // 2, :].copy()
                    self.delete_all_items((self.N - 1) // 2, self.N - 1, 0, self.N - 1)  # 127
                    self.mp[0:(self.N - 1) // 2 + 1, :], a, b, c, self.color[0:(self.N - 1) // 2 + 1, :] = self.generate_mp(0, (self.N - 1) // 2, 0, self.N - 1,
                                                                                   self.player_pos)
                elif dir == "down":
                    assert self.player_pos[1] == (self.N - 1) / 2, "Error!"
                    #          0 : 127
                    self.mp[0:(self.N - 1) // 2, :] = self.mp[(self.N - 1) // 2 + 1:self.N, :].copy()
                    self.color[0:(self.N - 1) // 2, :] = self.color[(self.N - 1) // 2 + 1:self.N, :].copy()
                    self.delete_all_items(0, (self.N - 1) // 2, 0, self.N - 1)
                    self.mp[(self.N - 1) // 2:self.N, :], a, b, c, self.color[(self.N - 1) // 2:self.N, :] = self.generate_mp((self.N - 1) // 2,
                                                                                    self.N - 1, 0, self.N - 1, (0, self.player_pos[
                            1]))  # player is on the upmost of the new map.
                # 移动此时剩余的所有物品
                self.mov_items(dir)
                self.animals += a
                self.monster += b
                self.food += c

            def move(self, action):
                dx, dy = [0, 1, 0, -1, 0], [-1, 0, 1, 0, 0] # left, down, right, up, stay
                # up, right, down, left
                # agent移动
                after_pos = (self.player_pos[0] + dx[action], self.player_pos[1] + dy[action])
                # 越界判定。此时不会有动物或怪物超过边界。
                if after_pos[0] < 0:
                    self.player_pos = (after_pos[0] + (self.N + 1) / 2, after_pos[1])
                    self.mov_mp('up')
                elif after_pos[0] >= self.N:
                    self.player_pos = (after_pos[0] - (self.N + 1) / 2, after_pos[1])
                    self.mov_mp('down')
                elif after_pos[1] < 0:
                    self.player_pos = (after_pos[0], after_pos[1] + (self.N + 1) / 2)
                    self.mov_mp('left')
                elif after_pos[1] >= self.N:
                    self.player_pos = (after_pos[0], after_pos[1] - (self.N + 1) / 2)
                    self.mov_mp('right')
                else:
                    self.player_pos = after_pos
                # 其他生物移动

                for i in range(len(self.animals)):
                    seed = np.random.random()
                    if seed < 0.5: continue  # 动物只有50%的概率移动
                    # 动物总是随机游走
                    dir = np.random.randint(4)
                    after_pos_LU = (self.animals[i].pos_LU[0] + dx[dir], self.animals[i].pos_LU[1] + dy[dir])
                    if after_pos_LU[0] < 0 or after_pos_LU[0] >= self.N or after_pos_LU[1] < 0 or after_pos_LU[
                        1] >= self.N: continue
                    # 只判定越界，越界则移动无效。这里先不管碰撞的问题，假设动物和怪物都可以重叠。但是怪物总是在重叠的上层。玩家永远在重叠的顶层。
                    self.animals[i].pos_LU = after_pos_LU

                for i in range(len(self.monster)):
                    seed = np.random.random()
                    if seed < 0.4: continue  # 怪物只有60%的概率移动
                    # 首先判定怪物是否能看见玩家
                    if min(abs(self.monster[i].pos_centre[0] - self.player_pos[0]),
                           abs(self.monster[i].pos_centre[1] - self.player_pos[1])) < self.monster_sight:
                        # 能看见玩家
                        # up, right, down, left
                        if abs(self.monster[i].pos_centre[0] - self.player_pos[0]) < abs(
                                self.monster[i].pos_centre[1] - self.player_pos[1]):
                            if self.monster[i].pos_centre[1] < self.player_pos[1]: dir = 2  # right
                            else: dir = 0  # left
                        else:
                            if self.monster[i].pos_centre[0] < self.player_pos[0]: dir = 1  # down
                            else: dir = 3  # up
                    else: dir = np.random.randint(4) # 不能看见玩家
                    after_pos_centre = (self.monster[i].pos_centre[0] + dx[dir], self.monster[i].pos_centre[1] + dy[dir])
                    if after_pos_centre[0] < 0 or after_pos_centre[0] >= self.N or after_pos_centre[1] < 0 or \
                            after_pos_centre[1] >= self.N: continue
                    self.monster[i].pos_centre = after_pos_centre
                # 是否撞到了动物
                for animal in self.animals:
                    subX, subY = animal.pos_LU[0] - self.player_pos[0], animal.pos_LU[1] - self.player_pos[1]
                    if subX >= 0 and subX <= 1 and subY >= 0 and subY <= 1:
                        self.hp += self.animal_cure
                        self.animals.remove(animal)
                        # print("bumping into animal!")
                # 是否撞到了怪物
                for monster in self.monster:
                    subX, subY = monster.pos_centre[0] - self.player_pos[0], monster.pos_centre[1] - self.player_pos[1]
                    if abs(subX) <= 1 and abs(subY) <= 1:
                        self.hp -= self.monster_harm
                        self.monster.remove(monster)
                        # print("bumping into monster!")
                # 是否撞到了食物
                for food in self.food:
                    if food.pos == self.player_pos:
                        self.hp += self.food_cure
                        self.food.remove(food)

            def get_observation(self):
                see_color = self.color.copy()
                for i in range(len(self.animals)):
                    X, Y = self.animals[i].pos_LU
                    see_color[X:min(X+2, self.N), Y:min(Y+2, self.N), :] = self.animals[i].color
                for i in range(len(self.monster)):
                    X, Y = self.monster[i].pos_centre
                    see_color[max(X-1,0):min(X+2,self.N), max(Y-1, 0):min(Y+2, self.N), :] = self.monster[i].color
                for i in range(len(self.food)):
                    X, Y = self.food[i].pos
                    see_color[X, Y, :] = self.food[i].color
                # color the player
                see_color[int(self.player_pos[0]), int(self.player_pos[1]), :] = np.array([255,0,0])
                return see_color

            def partial_get_observation(self, LU):
                see_color = self.color[LU[0]:LU[0]+self.player_sight, LU[1]:LU[1]+self.player_sight].copy()
                inner_animals, inner_monsters, inner_foods = [], [], []
                for i in range(len(self.animals)):
                    flag = 0
                    for x in range(2):
                        for y in range(2):
                            X, Y = self.animals[i].pos_LU[0] + x - LU[0], self.animals[i].pos_LU[1] + y - LU[1]
                            if X >= 0 and X < self.player_sight and Y >= 0 and Y < self.player_sight:
                                see_color[X, Y, :], flag = self.animals[i].color, 1
                    if flag: inner_animals.append(self.animals[i])
                for i in range(len(self.monster)):
                    flag = 0
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            X, Y = self.monster[i].pos_centre[0] + x - LU[0], self.monster[i].pos_centre[1] + y - LU[1]
                            if X >= 0 and X < self.player_sight and Y >= 0 and Y < self.player_sight:
                                see_color[X, Y, :], flag = self.monster[i].color, 1
                    if flag: inner_monsters.append(self.monster[i])
                for i in range(len(self.food)):
                    flag = 0
                    X, Y = self.food[i].pos[0] - LU[0], self.food[i].pos[1] - LU[1]
                    if X >= 0 and X < self.player_sight and Y >= 0 and Y < self.player_sight:
                        see_color[X, Y, :] = self.food[i].color
                    if flag: inner_foods.append(self.food[i])
                see_color[int(self.player_pos[0] - LU[0]), int(self.player_pos[1] - LU[1]), :] = np.array([255, 0, 0])
                return see_color, inner_animals, inner_monsters, inner_foods

            def get_local_observation(self):
                LU = (self.player_pos[0] - (self.player_sight - 1) // 2, self.player_pos[1] - (self.player_sight - 1) // 2)
                RD = (self.player_pos[0] + (self.player_sight - 1) // 2, self.player_pos[1] + (self.player_sight - 1) // 2)
                # note that player_sight must be smaller than the map(self.N)!
                real_LU = list(LU)
                if LU[0] < 0: real_LU[0] = 0
                elif RD[0] >= self.N - 1: real_LU[0] = self.N - self.player_sight
                if LU[1] < 0: real_LU[1] = 0
                elif RD[1] >= self.N - 1: real_LU[1] = self.N - self.player_sight
                real_LU = (int(real_LU[0]), int(real_LU[1]))
                see_color, inner_animals, inner_monsters, inner_foods = self.partial_get_observation(real_LU)
                return real_LU, see_color, inner_animals, inner_monsters, inner_foods, self.player_pos
from pynput.keyboard import Listener
current_action = None
def press(key):
    if key == 'l': action = 0
    elif key == 'd': action = 1
    elif key == 'w': action = 2
    elif key == 'r': action = 3
    else: action = 4
    if action < 4: env.move(current_action)
    env.debug_show_graph()

env = GridWorld()

def step():
    #a = input()
    #key = a[0]
    #if key == 'w':action = 0
    #elif key == 'r':action = 1
    #elif key == 'd':action = 2
    #elif key == 'l':action = 3
    #else:action = 4
    #if action < 4: env.move(current_action)
    if min(min(env.player_pos[0], env.N - env.player_pos[0]), min(env.player_pos[1], env.N - env.player_pos[1])) < 15:
        env.debug_show_graph(env.get_local_observation()[env.mp_local_obs['see_color']])
    env.move(0)

if __name__ == "__main__":
    env.initialize()
    print("initialization complete.")
    # while True:
    while True:
        step()
    #while True:
    #    with Listener(on_press=press) as listener:
    #        listener.join()

