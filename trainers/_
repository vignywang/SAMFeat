#
# Created  on 2020/8/25
#


def get_trainer(name):
    f_name, c_name = name.split('.')
    mod = __import__('{}.{}'.format(__name__, f_name), fromlist=[''])
    return getattr(mod, c_name)



