from neuroforge.tensor import Parameter


class Module:
    def parameters(self):
        params = []

        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)

            elif isinstance(value, Module):
                params.extend(value.parameters())

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Parameter):
                        params.append(item)

                    elif isinstance(item, Module):
                        params.extend(item.parameters())

        return params

    def state_dict(self):
        state = {}

        def collect(module, prefix=""):
            for name, value in module.__dict__.items():
                key = f"{prefix}{name}"

                if isinstance(value, Parameter):
                    state[key] = value.data

                elif isinstance(value, Module):
                    collect(value, key + ".")

                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        item_key = f"{key}.{index}"

                        if isinstance(item, Module):
                            collect(item, item_key + ".")

                        elif isinstance(item, Parameter):
                            state[item_key] = item.data

        collect(self)
        return state

    def load_state_dict(self, state):
        current_state = self.state_dict()

        if set(current_state.keys()) != set(state.keys()):
            raise ValueError("State dictionary keys do not match")

        parameters = self.parameters()

        for parameter, key in zip(parameters, state.keys()):
            parameter.data = state[key]