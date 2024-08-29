class BasicInvDynamic:

    def predict(self, **kwargs):
        raise NotImplementedError

    def __call__(self, **kwargs):
        return self.predict(**kwargs)