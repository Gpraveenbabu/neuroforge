class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0.0


class Adam:
    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = [0.0 for _ in parameters]
        self.v = [0.0 for _ in parameters]
        self.step_count = 0

    def step(self):
        self.step_count += 1

        for i, parameter in enumerate(self.parameters):
            gradient = parameter.grad

            self.m[i] = (
                self.beta1 * self.m[i]
                + (1 - self.beta1) * gradient
            )

            self.v[i] = (
                self.beta2 * self.v[i]
                + (1 - self.beta2) * gradient ** 2
            )

            m_corrected = (
                self.m[i] / (1 - self.beta1 ** self.step_count)
            )

            v_corrected = (
                self.v[i] / (1 - self.beta2 ** self.step_count)
            )

            parameter.data -= (
                self.learning_rate
                * m_corrected
                / (v_corrected ** 0.5 + self.epsilon)
            )

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0.0