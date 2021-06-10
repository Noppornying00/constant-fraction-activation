import torch
from torch import nn
from torch.nn.functional import relu


def secant_method(f, x0, x1, iterations):
    """Return the root calculated using the secant method."""
    for i in range(iterations):
        f_x1 = f(x1)
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f(x0) + 1e-9).float()
        x0, x1 = x1, x2
    return x2


def regula_falsi(func, a, b, iterations):
    f_a = func(a, -1)
    f_b = func(b, -1)

    if torch.any(f_a * f_b >= 0):
        # breakpoint()
        print(f_a * f_b >= 0)
        raise Exception(
            "You have not assumed right initial values in regula falsi")

    c = a  # Initialize result

    break_indices = torch.zeros_like(a).bool()

    for i in range(iterations):

        # Find the point that touches x axis
        c = (a * f_b - b * f_a) / (f_b - f_a)

        f_c = func(c, i)

        # Check if the above found point is root
        break_indices[f_c == 0] = True
        # if f_c == 0:
        #     break

        # Decide the side to repeat the steps
        b_eq_c_indices = ((f_c*f_a) < 0) & ~break_indices
        b[b_eq_c_indices] = c[b_eq_c_indices]
        a_eq_c_indices = ~(b_eq_c_indices | break_indices)
        a[a_eq_c_indices] = c[a_eq_c_indices]
        # elif f_c * f_a < 0:
        #     b = c
        # else:
        #     a = c

    return c


class relu_constant_fraction(nn.Module):
    def __init__(self, nb_channels):
        super(relu_constant_fraction, self).__init__()
        # not registering it as a parameter on purpose
        self.biases = nn.Parameter(torch.zeros(nb_channels))
        self.biases.requires_grad = False

    def forward(self, x):
        return relu(x-self.biases.view(1, -1, 1, 1))

    def adjust_bias(self, desired_fraction, prev_layer_outputs):

        if desired_fraction > 1-1e-3:
            self.biases.data = -10*torch.ones_like(self.biases)
            return

        def get_fraction_deviation(biases, j):

            activations = relu(prev_layer_outputs-biases.view(1, -1, 1, 1))
            ratios = (activations > 1e-3).float().mean(dim=(0, 2, 3))

            return ratios-desired_fraction

        with torch.no_grad():
            self.biases.data = regula_falsi(
                get_fraction_deviation, -1*torch.ones_like(self.biases), 1*torch.ones_like(self.biases), 20)

    def get_activation_fractions(self, prev_layer_outputs):

        activations = relu(prev_layer_outputs-self.biases.view(1, -1, 1, 1))
        ratios = (activations > 1e-3).float().mean(dim=(0, 2, 3))

        return ratios

    def show_trajectory(self, prev_layer_outputs):
        import numpy as np
        import matplotlib.pyplot as plt

        bias_values = np.linspace(-10, 10, 1000)
        fractions = np.zeros((1000, self.biases.shape[0]))

        for j, bias in enumerate(bias_values):

            cumulative_ratios = torch.zeros_like(self.biases)
            batch_size = 1000

            for i in range(0, len(prev_layer_outputs), batch_size):
                data = prev_layer_outputs[i:i +
                                          batch_size].to(self.biases.device)
                activations = relu(data-bias)
                cumulative_ratios += (activations >
                                      1e-3).float().mean(dim=(0, 2, 3))*len(data)

            fractions[j] = (cumulative_ratios /
                            len(prev_layer_outputs)).detach().cpu().numpy()

        plt.plot(bias_values, fractions)
        plt.show()


class surviving_K_relu(nn.Module):
    def __init__(nb_channels):
        super().__init__()
        pass

    def forward():
        pass
