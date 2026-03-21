from dataclasses import fields


def builder(cls):
    class DataclassBuilder:
        def __init__(self):
            self._kwargs = {}

        def build(self):
            return cls(**self._kwargs)

    for field in fields(cls):

        def make_setter(name):
            def setter(self, value):
                self._kwargs[name] = value
                return self

            return setter

        setattr(DataclassBuilder, field.name, make_setter(field.name))

    cls.builder = staticmethod(lambda: DataclassBuilder())

    return cls
