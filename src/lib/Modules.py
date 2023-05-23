from collections import namedtuple

Bounds = namedtuple("Bounds", "lower upper")
Optim = namedtuple("Optim", "start lower upper", defaults=[0, None, None])


class ParametricModule:
    def __init__(self, name="", **kwargs):
        self.name = name
        self._params = dict()
        self.bounds = dict()
        for k, v in kwargs.items():
            if isinstance(v, Optim):
                self.bounds[k] = Bounds(lower=v.lower, upper=v.upper)
                # self.set_params(**{k: np.mean(v)})  # starting value the center
                self.set_params(**{k: v.start})  # starting value the center
            else:
                self.set_params(**{k: v})
        self.losses = dict()
        self.calibrated = False

    @property
    def params(self):
        return self._params

    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def __str__(self):
        return str(self.__class__.__name__) + self.name
