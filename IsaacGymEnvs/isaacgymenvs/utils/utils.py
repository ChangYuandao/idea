# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# python
#import pwd
import getpass
import tempfile
import time
from collections import OrderedDict
from os.path import join

import numpy as np
import torch
import random
import os


def retry(times, exceptions):
    """
    Retry Decorator https://stackoverflow.com/a/64030200/1645784
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param exceptions: Lists of exceptions that trigger a retry attempt
    :type exceptions: Tuple of Exceptions
    """
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(f'Exception thrown when attempting to run {func}, attempt {attempt} out of {times}')
                    time.sleep(min(2 ** attempt, 30))
                    attempt += 1

            return func(*args, **kwargs)
        return newfn
    return decorator


def flatten_dict(d, prefix='', separator='.'):
    res = dict()
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value

    return res


def set_np_formatting():
    """ formats numpy print """
    # NumPy 内置函数，用于控制数组打印时的格式
    # edgeitems=30：在打印大数组时，前后各显示 30 个元素
    # infstr='inf'：正无穷大显示为 'inf'
    # linewidth=4000：单行最大字符数，超出换行。4000 表示几乎不换行
    # nanstr='nan'：NaN 值显示为 'nan'
    # precision=2：浮点数保留小数点后 2 位
    # suppress=False：是否禁止科学计数法
    # threshold=10000：大于 10000 元素时才会使用省略号 ...
    # formatter=None：可以自定义打印函数，默认 None
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    # Python 标准库 random 模块的随机数种子
    random.seed(seed)
    # Numpy 随机数种子
    np.random.seed(seed)
    # Pytorch CPU 随机数种子
    torch.manual_seed(seed)
    # Python 哈希随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 当前 GPU 随机种子
    torch.cuda.manual_seed(seed)
    # 所有 GPU 随机种子
    torch.cuda.manual_seed_all(seed)

    # 固定算法，可复现，但是会比较慢
    if torch_deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    # 非确定性算法，速度更快
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

def nested_dict_set_attr(d, key, val):
    pre, _, post = key.partition('.')
    if post:
        nested_dict_set_attr(d[pre], post, val)
    else:
        d[key] = val
    
def nested_dict_get_attr(d, key):
    pre, _, post = key.partition('.')
    if post:
        return nested_dict_get_attr(d[pre], post)
    else:
        return d[key]

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def safe_ensure_dir_exists(path):
    """Should be safer in multi-treaded environment."""
    try:
        return ensure_dir_exists(path)
    except FileExistsError:
        return path


def get_username():
    uid = os.getuid()
    try:
        return getpass.getuser()
    except KeyError:
        # worst case scenario - let's just use uid
        return str(uid)


def project_tmp_dir():
    tmp_dir_name = f'ige_{get_username()}'
    return safe_ensure_dir_exists(join(tempfile.gettempdir(), tmp_dir_name))

# EOF
