
class ExampleRegistry(object):

    _example_models = {
        'simple_rbc': 'pynare/examples/pynare/simple_rbc.mod',
        'simple_linear': 'pynare/examples/pynare/simple_linear.mod',
        'optim_ck': 'pynare/examples/pynare/optim_ck.mod',
        'basic_nk': 'pynare/examples/pynare/basic_nk.mod',
        'four_eq_nk': 'pynare/examples/pynare/four_eq_nk.txt'
    }

    def __init__(self):
        pass

    @classmethod
    def get_example(cls, ex_name):
        try:
            return cls._example_models[ex_name]
        except KeyError:
            # remove any '.x' from ex_name string and try again
            try:
                cleaned_name = ex_name.split('.')[0]
                return cls._example_models[cleaned_name]
            except KeyError:
                raise ValueError(f"unrecognized example name: {ex_name}")
