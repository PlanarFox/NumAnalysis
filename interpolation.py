import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 1000 # TODO: replace with real N
I0 = 1 # TODO: replace with real I0
r_beta = 1

def load_table():
    df = pd.read_csv('LUT.csv')
    t = df['t'].values
    I = df['I'].values
    return t, I

def hermite_func(t, tk, I):
    # t, I: given t_k and I(t_k)
    I_div = I - I * I / N
    I_hat = t.copy()
    i = 0
    for k in range(tk.size-1):
        inc_tk = tk[k+1] - tk[k]
        I_tk =I[k]
        I_tk_1=I[k+1]
        if t[i] == tk[k]:
            I_hat[i] = I[k]
            i += 1
            if i > t.size-1:
                return I_hat
        while t[i] < tk[k+1]:
            ti_tk = t[i]-tk[k]
            ti_tk1 = t[i]-tk[k+1]
            I_hat[i] = (ti_tk1/(-inc_tk))*(ti_tk1/(-inc_tk))*((1+2*ti_tk/inc_tk)*I[k]+ti_tk*I_div[k])
            I_hat[i] += (ti_tk/(inc_tk))*(ti_tk/(inc_tk))*((1+2*ti_tk1/(-inc_tk))*I[k+1]+ti_tk1*I_div[k+1])
            i += 1
            if i > t.size-1:
                return I_hat
    return I_hat

def func(t):
    # calculate I(t)
    It =  N * I0 / (I0 + (N - I0) * np.exp(-r_beta * t)) # TODO: replace with the real I(t)
    return It


if __name__ == "__main__":
    tk, I_tk = load_table()
    t = np.arange(0, 15, 0.1)
    # calculate
    I = func(t)
    I_hat = hermite_func(t, tk, I_tk)
    print(np.abs(I_hat - I).max())

# git learning
