import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def coupled_system(y, t, params):
    A, B, C = y
    I = 1
    k1, k2, k3, k4, k5, k6, ka, J1, J2, J3, J4, J5, J6, Ja, n1, n2, n3, na = \
        params['k1'], params['k2'], params['k3'], params['k4'], params['k5'], params['k6'], \
        params['ka'], \
        params['J1'], params['J2'], params['J3'], params['J4'], params['J5'], params['J6'], \
        params['Ja'], \
        params['n1'], params['n2'], params['n3'], \
        params['na']

    dA_dt = I * k1 * (1 - A) / ((1 - A) + J1) - C * k4 * A / (A + J4**n1) + A * ka * (1-A)/((1-A) + Ja**na)
    dB_dt = k2 * (1 - B) / ((1 - B) + J2) - A * k5 * B / (B + J5**n2)
    dC_dt = k3 * (1 - C) / ((1 - C) + J3) - B * k6 * C / (C + J6**n3)
    
    return [dA_dt, dB_dt, dC_dt]
def simulate_system(params, y0, t):
    result = odeint(coupled_system, y0, t, args=(params,))
    return result
def check_oscillation(time_series, threshold=0.05):
    std_dev = np.std(time_series, axis=0)

    if all(std_dev > threshold):
        return True
    else:
        return False

params_base = {'k1': 1, 'k2': 0.5, 'k3': 1, 'k4': 2, 'k5': 2, 'k6': 2, 'ka' : 0.5,
        'J1': 0.2, 'J2': 0.2, 'J3': 0.3, 'J4': 0.2, 'J5': 0.2, 'J6': 0.3, 'Ja': 0.5,
        'n1': 2, 'n2': 2, 'n3': 2, 'na':2}

t = np.linspace(0, 100, 1000)

y0 = [0.5, 0.5, 0.5]

# param_sweeps = {
#     'k1': [0.5, 1, 2, 3, 4, 5],
#     'k2': [0.5, 1, 2, 3, 4, 5],
#     'k3': [0.5, 1, 2, 3, 4, 5],
#     'k4': [0.5, 1, 2, 3, 4, 5],
#     'k5': [0.5, 1, 2, 3, 4, 5],
#     'k6': [0.5, 1, 2, 3, 4, 5],
#     'J1': [0.1, 0.2, 0.4, 0.6, 0.8, 1],
#     'J2': [0.1, 0.2, 0.4, 0.6, 0.8, 1],
#     'J3': [0.1, 0.2, 0.4, 0.6, 0.8, 1],
#     'J4': [0.1, 0.2, 0.4, 0.6, 0.8, 1],
#     'J5': [0.1, 0.2, 0.4, 0.6, 0.8, 1],
#     'J6': [0.1, 0.2, 0.4, 0.6, 0.8, 1],
#     'n1': [1, 2],
#     'n2': [1, 2],
#     'n3': [1, 2]
# }

param_sweeps = {
    'k1': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'k2': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'k3': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'k4': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'k5': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'k6': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'J1': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'J2': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'J3': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'J4': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'J5': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'J6': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    'n1': [1, 2],
    'n2': [1, 2],
    'n3': [1, 2]
}

oscillation_results = []

for param_name, param_values in param_sweeps.items():
    for value in param_values:
        params = params_base.copy()
        params[param_name] = value
        result = simulate_system(params, y0, t)
        oscillates = check_oscillation(result)
        oscillation_results.append((param_name, value, oscillates))
oscillation_results = np.array(oscillation_results, dtype=[('param', 'U10'), ('value', 'f4'), ('oscillates', 'bool')])

rows = 3
cols = 5

fig, axs = plt.subplots(rows, cols, figsize=(15, 8))

if rows == 1:
    axs = np.array([axs])

param_groupings = {
    'J_col1': ['J1', 'J2', 'J3'],
    'J_col2': ['J4', 'J5', 'J6'],
    'K_col1': ['k1', 'k2', 'k3'],
    'K_col2': ['k4', 'k5', 'k6'],
    'N': ['n1', 'n2', 'n3']
}

colors = {
    'J_col1': "blue",
    'J_col2': "blue",
    'K_col1': "green",
    'K_col2': "green",
    'N': "red"
}


linestyle = {
    'J_col1': "-",
    'J_col2': "--",
    'K_col1': "-",
    'K_col2': "--",
    'N': "-"
}

for i, group in enumerate(param_groupings.keys()):
    params_to_plot = param_groupings[group]
    for j, param_name in enumerate(params_to_plot):
        if "1" in param_name or "4" in param_name:
            row = 0
        if "2" in param_name or "5" in param_name:
            row = 1
        if "3" in param_name or "6" in param_name:
            row = 2
        col = i
        values = oscillation_results[oscillation_results['param'] == param_name]
        axs[row, col].plot(values['value'], values['oscillates'], marker="o", linestyle=linestyle[group], color = colors[group])
        axs[row, 0].set_ylabel('Oscillates (1 = True, 0 = False)')
        axs[2, col].set_xlabel(f'Value')
        axs[row, col].grid(True)
        axs[row, col].legend({param_name}, loc = "center right")
        
        if param_name in ['k1', 'k2', 'k3', 'k4', 'k5', 'k6']:
            axs[row, col].set_xscale('log')
        if param_name in ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']:
            axs[row, col].set_xscale('log')

plt.tight_layout()
plt.savefig("plots/oscillatory_behavior_A_positive_log_new_equations.pdf")
# plt.savefig("plots/oscillatory_behavior_A_positive_new_equations.pdf")
plt.show()
