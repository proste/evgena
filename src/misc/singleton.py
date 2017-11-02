from abc import ABCMeta


class Singleton(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        elif len(args) + len(kwargs) != 0:
            raise TypeError('Singleton already instantiated')

        return cls._instance


class SingletonABCMeta(Singleton, ABCMeta):
    pass
