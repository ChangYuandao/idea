from rl_games.common import object_factory
import rl_games.algos_torch
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import models

NETWORK_REGISTRY = {}
MODEL_REGISTRY = {}

def register_network(name, target_class):
    NETWORK_REGISTRY[name] = lambda **kwargs: target_class()

def register_model(name, target_class):
    MODEL_REGISTRY[name] = lambda  network, **kwargs: target_class(network)


class NetworkBuilder:
    def __init__(self):
        
        # 创建一个网络工厂对象，用于动态创建网络实例
        self.network_factory = object_factory.ObjectFactory()
        
        # 设置网络工厂的构建器。NETWORK_REGISTRY 是网络的注册表，通常包含已定义的网络类型
        self.network_factory.set_builders(NETWORK_REGISTRY)
        
        # 注册不同的网络类型，每个类型对应一个 lambda 函数，用于创建特定的网络类实例
        self.network_factory.register_builder('actor_critic', lambda **kwargs: network_builder.A2CBuilder())
        self.network_factory.register_builder('resnet_actor_critic',
                                              lambda **kwargs: network_builder.A2CResnetBuilder())
        self.network_factory.register_builder('rnd_curiosity', lambda **kwargs: network_builder.RNDCuriosityBuilder())
        self.network_factory.register_builder('soft_actor_critic', lambda **kwargs: network_builder.SACBuilder())

    def load(self, params):
        
        # 从 train 配置字典中提取网络的名称，路径是 network.name 
        # AntPPO 对应的网络名称是 actor_critic
        network_name = params['name']
        
        # 使用网络工厂创建相应的网络对象，根据网络名称选择对应的网络构建器
        network = self.network_factory.create(network_name)
        
        # 使用配置字典中的参数加载网络的具体配置
        network.load(params)

        # 返回创建并加载的网络对象
        return network


class ModelBuilder:
    def __init__(self):
        # 创建一个模型工厂对象，用于动态创建模型实例
        self.model_factory = object_factory.ObjectFactory()
        
        # 设置模型工厂的构建器。MODEL_REGISTRY 是模型的注册表，通常包含已定义的模型类型
        self.model_factory.set_builders(MODEL_REGISTRY)
        
        # 注册不同的模型类型，每个类型对应一个 lambda 函数，用于创建特定的模型类实例
        self.model_factory.register_builder('discrete_a2c', lambda network, **kwargs: models.ModelA2C(network))
        self.model_factory.register_builder('multi_discrete_a2c',
                                            lambda network, **kwargs: models.ModelA2CMultiDiscrete(network))
        self.model_factory.register_builder('continuous_a2c',
                                            lambda network, **kwargs: models.ModelA2CContinuous(network))
        self.model_factory.register_builder('continuous_a2c_logstd',
                                            lambda network, **kwargs: models.ModelA2CContinuousLogStd(network))
        self.model_factory.register_builder('soft_actor_critic',
                                            lambda network, **kwargs: models.ModelSACContinuous(network))
        self.model_factory.register_builder('central_value',
                                            lambda network, **kwargs: models.ModelCentralValue(network))
        self.model_factory.register_builder('continuous_a2c_tanh',
                                            lambda network, **kwargs: models.ModelA2CContinuousTanh(network))
        # 创建一个网络构建器，用于构建神经网络结构
        self.network_builder = NetworkBuilder()
        
    # 获取网络构建器对象的方法
    def get_network_builder(self):
        return self.network_builder

    def load(self, params):
        # 根据配置字典 params 创建并返回一个模型
        # 从 params 中获取模型的名称
        # 以 Ant 的训练配置为例，返回的 model_name 是 continuous_a2c_logstd
        model_name = params['model']['name']
        
        # 使用网络构建器加载网络配置
        network = self.network_builder.load(params['network'])
        
        # 使用模型工厂创建对应的模型实例，传入网络作为参数
        model = self.model_factory.create(model_name, network=network)
        
        # 返回创建的模型
        return model
