"""
Q-learning agent's brain
用来实现Qlearning算法
"""
import numpy as np
import pandas as pd
from simpleQLearning import env

class QTable(object):
    def __init__(self,actions,learning_rate,gamma,epsilon):
        """
        定义Qlearning的state-action value table
        :param actions: agent执行的动作
        :param learning_rate: 学习率
        :param gamma: value function的折扣率
        :param epsilon: epsilon-greedy
        """
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # 定义一个pd二维表，列名columns、行名index
        # 行是state，列是action，数据是float型的reward
        self.table = pd.DataFrame(columns=self.actions,dtype=np.float32)

    def policy(self,observation):
        """
        决策函数，根据observation得到的state来选择相应的action
        :param observation:从环境中获得的state
        :return:action
        """
        if observation not in self.table.index:
            # 将该state下对应的action以及其value值加入table
            # pd.DataFrame.append是在列方向上扩充table，即列数不变，行数增加
            self.table = self.table.append(
                # pd.Series是一个一维的pandas数据结构
                pd.Series(
                    [0]*len(self.actions), # 长度是actions的个数，初始化全为0
                    index=self.table.columns, # index是actions
                    name=observation
                )
            )
            print("observation:", observation," 被添加入table中")
        _random = np.random.uniform(low=0.0,high=1.0)
        if _random<self.epsilon:
            # 如果_random小于epsilon就采取随机策略（从动作空间中随机选择一个动作）
            action = np.random.choice(self.actions)
        else:
            # 使用贪心策略，选择当前最优动作(即可以获得最优值函数的动作）
            # 根据观测到的state获得表中所有该state对应的action的value
            state_actions = self.table.loc[observation,:] # 根据observation来查看一行的数据，loc表示observation是index
            # 随机选择一个value最大的动作
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        return action

    def learn(self,s,a,r,s_):
        """
        qlearning学习函数
        :param s: state
        :param a: actions
        :param r: 当前的reward
        :param s_: 新的state(next state)
        :return:
        """
        # 判断_s是否存在于table中，如果不存在就将该状态以及其对应的所有action和value加入table
        if s_ not in self.table.index:
            # 将该state下对应的action以及其value值加入table
            # pd.DataFrame.append是在列方向上扩充table，即列数不变，行数增加
            self.table = self.table.append(
                # pd.Series是一个一维的pandas数据结构
                pd.Series(
                    [0]*len(self.actions), # 长度是actions的个数，初始化全为0
                    index=self.table.columns, # index是actions
                    name=s_
                )
            )
            print("s_:", s_," 被添加入table中")
        # 策略评估
        predict = self.table.loc[s,a] # 获得该state和action下的对value的评价
        if s_ != 'terminal':
            # 如果s_不是terminal状态,就更新目标-动作状态值函数
            # 就是当前最优的state和action
            target = r+self.gamma*self.table.loc[s_,:].max()
        else:
            # 如果当前的s_是terminal状态
            target = r
        # 就是用目标策略生成的value和行为策略实际在环境中执行获得的value作比较，希望行为策略接近目标策略
        # 目标策略是贪心的最优策略
        self.table.loc[s,a] += self.learning_rate*(target-predict) # 更新


def train(env,learning_rate=0.01,gamma=0.9,epsilon=0.1,episodeN=100):
    RL = QTable(actions=list(range(env.n_actions)),learning_rate=learning_rate,gamma=gamma,epsilon=epsilon)
    for episode in range(episodeN):
        print("episode:",episode)
        # reset环境
        observation = env.reset()
        while True:
            env.render() # 刷新env（用于图像化显示env）
            # 根据策略选择action
            action = RL.policy(str(observation))
            # 将action在环境中执行，获得next observation信息
            observation_ ,reward,done = env.step(action)
            RL.learn(str(observation),action,reward,str(observation_))
            observation = observation_
            if done:
                # 本条轨迹结束
                break
    print(RL.table)
    env.destroy()


def main(episodeN):
    env_ = env.Maze()
    env_.after(episodeN,train(env=env_,episodeN=episodeN))
    env_.mainloop()

if __name__ == "__main__":
    main(100)
