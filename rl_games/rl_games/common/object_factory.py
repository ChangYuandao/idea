# 动态创建对象（algos、players、models 等）
# 注册对象构造器（builder），然后根据名称创建实例

class ObjectFactory:
    """General-purpose class to instantiate some other base class from rl_games. Usual use case it to instantiate algos, players etc.

    The ObjectFactory class is used to dynamically create any other object using a builder function (typically a lambda function).

    """

    # 初始化一个私有字典 _builders 用于存储
    def __init__(self):
        """Initialise a dictionary of builders with keys as `str` and values as functions.

        """
        self._builders = {}

    # 向 _builders 字典注册一个对象构造器
    def register_builder(self, name, builder):
        """Register a passed builder by adding to the builders dict.

        Initialises runners and players for all algorithms available in the library using `rl_games.common.object_factory.ObjectFactory`

        Args:
            name (:obj:`str`): Key of the added builder.
            builder (:obj `func`): Function to return the requested object

        """
        self._builders[name] = builder

    # 直接用一个字典覆盖已有的 _builders
    # 适用于一次性注册多个构造器
    def set_builders(self, builders):
        self._builders = builders
    
    # 据 name 查找对应的构造器
    # 调用构造器并传入任意关键字参数 **kwargs
    # 返回对象实例
    def create(self, name, **kwargs):
        """Create the requested object by calling a registered builder function.

        Args:
            name (:obj:`str`): Key of the requested builder.
            **kwargs: Arbitrary kwargs needed for the builder function

        """
        builder = self._builders.get(name)
        if not builder:
            raise ValueError(name)
        return builder(**kwargs)