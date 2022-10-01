class VARModel:
    """
    A $k$-variate VAR(p) model of order-$p$.
    """

    def __init__(self, k, p):
        self.p = p
        self.k = k

    def _scenario_random(self):
        return [self.p.random() for _ in range(self.k)]

    def generate(self):
        pass

    def __call__(self):
        pass
