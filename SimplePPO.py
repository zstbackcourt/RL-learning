# -*- coding:utf-8 -*-
"""
    Author：ShenWeijie
    Date:2019/2/26
    功能：最简单的单线程PPO实现；
    reference："https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob
                    /master/contents/12_Proximal_Policy_Optimization/simply_PPO.py"
"""
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""定义default参数"""
EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001 # actor的learning_rate
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1 # 状态空间3维，动作空间1维
# 两种优化方式KL散度和Clip
METHOD = [
    dict(name='kl_pen', kl_target=0.01, beta=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1] # choose the method for optimization


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        # 定义输入state的placeholder
        self.tf_state = tf.placeholder(dtype=tf.float32,shape=[None,S_DIM],name="state")
        # 定义action的placeholder
        self.tf_action = tf.placeholder(dtype=tf.float32,shape=[None,A_DIM],name="action")
        # 定义advantage的placeholder
        self.tf_advantage = tf.placeholder(dtype=tf.float32,shape=[None,1],name="advantage")

        # 定义critic
        with tf.variable_scope('critic'):
            # l1是一个全连接层输units=100，用relu激活
            l1 = tf.layers.dense(self.tf_state, 100, tf.nn.relu)
            # 输出当前state所得到的value
            self.v = tf.layers.dense(l1, 1)
            # 经过了y折的reward
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], name = "y-discount_r")
            # advantage function
            self.advantage = self.tfdc_r - self.v
            # critic的loss就是advantage function的平方的均值
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            # critic的目的就是最小化closs，也就是最小化TD error
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # 定义actor
        # actor的目的就是根据advantage更新policy最大化J_PPO
        with tf.variable_scope('actor'):
            pi,pi_params = self.policy_net(name="pi",trainable = True)
            oldpi,oldpi_params = self.policy_net(name="old_pi",trainable=True)
            # 采样动作
            with tf.variable_scope('sample_action'):
                # pi是一个正态分布，也就是说这个action是服从一个正态分布的；
                # pi.sample()是从这个分布中随机抽样；
                # 随着学习的更新，这个正态分布会趋向于一个很小的区间，从这个区间抽样的时候就近似确定的一个action了
                self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
            # 更新策略
            with tf.variable_scope('update_oldpi'):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            # 定义PPO的loss
            with tf.variable_scope('loss'):
                with tf.variable_scope('surrogate_loss'):
                    # ratio = pi.log_prob(self.tf_action) / oldpi.log_prob(self.tf_action)
                    # prob()计算概率密度
                    ratio = pi.prob(self.tf_action) / oldpi.prob(self.tf_action)
                    # 论文公式(6)，即L_CPI
                    surr = ratio * self.tf_advantage
                # 如果采用kl散度的方法
                if METHOD['name'] == 'kl_pen':
                    # KL散度中的beta
                    self.tf_beta = tf.placeholder(dtype=tf.float32,shape=None,name="beta")
                    # 计算oldpi和pi的KL散度
                    kl = tf.distributions.kl_divergence(oldpi, pi)
                    self.kl_mean = tf.reduce_mean(kl)
                    # 公式(8)，最大化L_KLPEN就是最小化它的负数
                    self.finall_loss = -(tf.reduce_mean(surr - self.tf_beta * kl))
                else:  # 采用clip方法
                    # 公式(9)
                    self.finall_loss = -tf.reduce_mean(tf.minimum(surr,
                                                                  tf.clip_by_value(ratio, 1. - METHOD['epsilon'],
                                                                                   1. + METHOD['epsilon']) * self.tf_advantage))

        # 定义train_op
        self.train_op = tf.train.AdamOptimizer(A_LR).minimize(self.finall_loss)
        # 初始化全局变量
        self.sess.run(tf.global_variables_initializer())
    # 策略网路生成action，action是一个正太分布
    # 该实验中的action space中只有一个action
    def policy_net(self,name,trainable = True):
        with tf.variable_scope(name):
            # l1是一个全连接网络
            l1 = tf.layers.dense(self.tf_state, 100, tf.nn.relu, trainable=trainable)
            # 生成正太分布的μ（均值mean）
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            # 生成正态分布的θ（标准差stddev）
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            # 动作是服从正太分布的
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        # 把name域下的所有参数收集起来
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # 选择action
    def action_choose(self,state):
        state = state[np.newaxis, :] # 给state增加一维
        action = self.sess.run(self.sample_op, {self.tf_state: state})[0]
        return np.clip(action, -2, 2) # 把动作卡在-2到2之间

    # 获得当前state的value
    def get_value(self,state):
        if state.ndim < 2: state = state[np.newaxis, :]
        return self.sess.run(self.v, {self.tf_state: state})[0, 0]

    # 定义训练函数
    def train(self,state,action,reward):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tf_state: state, self.tfdc_r: reward})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # 更新actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.train_op, self.kl_mean],
                    {self.tf_state: state, self.tf_action: action, self.tf_advantage: adv, self.tf_beta: METHOD['beta']})
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['beta'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['beta'] *= 2
            METHOD['beta'] = np.clip(METHOD['beta'], 1e-4, 10)  # sometimes explode, this clipping is my solution
        else:  # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.train_op, {self.tf_state: state, self.tf_action: action, self.tf_advantage: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tf_state: state, self.tfdc_r: reward}) for _ in range(C_UPDATE_STEPS)]


def train_main():
    env = gym.make('Pendulum-v0').unwrapped
    ppo = PPO()
    all_ep_r = [] # 存储所有episode的rewards
    for ep in range(EP_MAX):
        # 重置环境
        state = env.reset()
        # 存放state，action和reward的buffer，因为是连续的action和state space，
        # 所以在一个episode中存放一组数据，然后送入PPO更新
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):  # in one episode
            # render()用来展现环境
            env.render()
            action = ppo.action_choose(state)
            # step()执行给定的动作，返回4个值：执行action之后的state，reward，done(实验是否结束),info
            s_, r, done, _ = env.step(action)
            buffer_s.append(state)
            buffer_a.append(action)
            buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
            state = s_
            ep_r += r

            # update ppo
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                v_s_ = ppo.get_value(s_)
                discounted_r = []
                # γ折reward
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse() # 将折扣reward反向排序，把大的放前面（因为大的先被加入list)
                # 将bs，ba，br用numpy数组返回
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.train(bs, ba, br)
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|beta: %.4f" % METHOD['beta']) if METHOD['name'] == 'kl_pen' else '',
        )

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()

if __name__ == "__main__":
    train_main()

