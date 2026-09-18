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
