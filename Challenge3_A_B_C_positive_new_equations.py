import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def coupled_system(y, t, k1, k2, k3, k4, k5, k6, ka, kb, kc, J1, J2, J3, J4, J5, J6, Ja, Jb, Jc, n1, n2, n3, na, nb, nc):
    A, B, C = y
    I = 1
    dA_dt = I * k1 * (1 - A) / ((1 - A) + J1) - C * k4 * A / (A + J4**n1) + A * ka * (1-A)/((1-A) + Ja**na)
    dB_dt = k2 * (1 - B) / ((1 - B) + J2) - A * k5 * B / (B + J5**n2) + B * kb * (1-B)/((1-B) + Jb**nb)
    dC_dt = k3 * (1 - C) / ((1 - C) + J3) - B * k6 * C / (C + J6**n3) + C * kc * (1-C)/((1-C) + Jc**nc)
    
    A = np.clip(A, 0.01, 0.99)
    B = np.clip(B, 0.01, 0.99)
    C = np.clip(C, 0.01, 0.99)
    
    return [dA_dt, dB_dt, dC_dt]

def integrate_system(y0, time_steps, params):
    result = odeint(
        lambda y, t: coupled_system(y, t, params['k1'], params['k2'], params['k3'], 
                                     params['k4'], params['k5'], params['k6'],
                                     params['ka'], params['kb'], params['kc'],
                                     params['J1'], params['J2'], params['J3'], 
                                     params['J4'], params['J5'], params['J6'],
                                     params['Ja'], params['Jb'], params['Jc'],
                                     params['n1'], params['n2'], params['n3'],
                                     params['na'], params['nb'], params['nc']),
        y0, time_steps
    )
    return result

def parameter_sweep(y0, time_steps, param_sets):
    results = []
    for params in param_sets:
        result = integrate_system(y0, time_steps, params)
        results.append(result)
    return results

def main():
    y0 = [0.5, 0.5, 0.5]
    time_steps = np.linspace(0, 100, 1000)

    param_sets = [
        
        {'k1': 1, 'k2': 0.5, 'k3': 1, 'k4': 2, 'k5': 2, 'k6': 2, 'ka': 0.5, 'kb': 0.2, 'kc': 0.1,
        'J1': 0.2, 'J2': 0.2, 'J3': 0.3, 'J4': 0.2, 'J5': 0.2, 'J6': 0.3, 'Ja': 0.5, 'Jb': 0.5, 'Jc': 0.2,
        'n1': 2, 'n2': 2, 'n3': 2, 'na':2, 'nb':2, 'nc':2},  
        
    ]
    
    all_results = parameter_sweep(y0, time_steps, param_sets)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for i, (params, result) in enumerate(zip(param_sets, all_results)):
        A_values = result[:, 0]
        B_values = result[:, 1]
        C_values = result[:, 2]

        axs[0].plot(time_steps, A_values, label=f'A', color = 'C4')
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel('Concentration of A')
        
        axs[1].plot(time_steps, B_values, label=f'B', color = 'C1')
        axs[1].set_ylim(0, 1)
        axs[1].set_ylabel('Concentration of B')

        axs[2].plot(time_steps, C_values, label=f'C', color = 'C6')
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel('Concentration of C')

    axs[0].set_title('Protein Concentration over Time')
    axs[2].set_xlabel('Time')
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    

    plt.ylim(0, 1)
    
    axs[0].legend(loc = 'upper left')
    axs[1].legend(loc = 'upper left')
    axs[2].legend(loc = 'upper left')
    

    for ax in axs:
        ax.grid(True)
    plt.tight_layout()
    
    plt.savefig('plots/concentration_over_time_A_B_C_positive_new_equations.pdf')
    plt.show()

if __name__ == "__main__":
    main()
