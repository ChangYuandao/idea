"""
这段代码定义了一个强化学习框架中非常核心的抽象接口类 —— IVecEnv。
IVecEnv 类定义了一组方法，这些方法是所有具体环境类必须实现的接口。通过继承 IVecEnv，具体的环境类可以确保它们提供了统一的功能和行为，从而使得强化学习算法能够与不同的环境进行交互，而不需要关心环境的具体实现细节。
"""

class IVecEnv:
    
    # 抽象方法，表示在环境中执行一步操作
    # 子类必须实现此方法，否则调用时会抛出 NotImplementedError
    def step(self, actions):
        raise NotImplementedError

    # 抽象方法，表示重置环境到初始状态
    # 子类必须实现此方法，否则调用时会抛出 NotImplementedError
    def reset(self):
        raise NotImplementedError

    # 检查环境是否有 action masks（动作掩码）功能
    # action masks 用于指示在某些状态下哪些动作是不可用的
    # 默认返回 False，表示没有 action masks
    def has_action_masks(self):
        return False

    # 获取环境中智能体的数量
    # 默认返回 1，表示单智能体环境
    def get_number_of_agents(self):
        return 1

    # 获取环境的详细信息
    # pass 表示该方法没有具体实现，子类需要重写此方法，如果子类没有实现，则调用时不会报错但也不会有任何效果
    def get_env_info(self):
        pass
    
    # 设置环境的随机种子
    def seed(self, seed):
        pass

    # 将训练信息传递给环境
    # env_frames：已训练的环境步数，用于度量训练进度
    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        pass
    
    # 获取环境的可序列化状态，以便保存到检查点
    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None

    # 将保存的环境状态重新载入（恢复训练时使用
    # 输入参数 env_state 通常是 get_env_state() 的返回值
    def set_env_state(self, env_state):
        pass
