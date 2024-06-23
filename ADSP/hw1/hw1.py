import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

FILTER_LEN = 17
DELTA = 1e-4
FILTER_PASS = (0, 0.225)
WEIGHT = ((0, 0.2, 10), (0.25, 0.5, 6))

def weight_function(x):
    if WEIGHT[0][0] <= x <= WEIGHT[0][1]:
        return WEIGHT[0][2]
    elif WEIGHT[1][0] <= x <= WEIGHT[1][1]:
        return WEIGHT[1][2]
    else:
        return 0

def freq_response(F, S):
    out = 0
    k = int((FILTER_LEN-1)/2)
    for n in range(k+1):
        out += S[n]*math.cos(2*math.pi*n*F)
    return out

def designed_filter(x):
    if FILTER_PASS[0] <= x <= FILTER_PASS[1]:
        return 1
    else:
        return 0

def err(F, S):
    return (freq_response(F, S) - designed_filter(F)) * weight_function(F)


def main():

    k = int((FILTER_LEN-1)/2)
    print(f"k = {k}")
    N_EXP = k+2
    print(f"Number of extreme point = {N_EXP}")


    max_err_last = math.inf
    F_m = [0, 0.05, 0.1, 0.15, 0.19, 0.26, 0.3, 0.35, 0.4, 0.45]
    while True:
        # Step 2, get A ,b 
        A = []
        for m in range(k+2):
            row = []
            for n in range(k+2):
                if n == 0:
                    row.append(1)
                elif n == k+1: 
                    row.append( ( (-1)**m ) / weight_function(F_m[m]) )
                else:
                    row.append(math.cos(2*n*math.pi*F_m[m]))
            A.append(row)

        A = np.array(A)
        A_inv = inv(A)

        b = []
        for m in range(k+2):
            b.append(designed_filter(F_m[m]))
        b = np.array(b)

        S = np.matmul(A_inv, b)

        new_extreme = []
        max_err = -1
        f1 = -1
        f2  = -1
        for i in range(int(0.5 / DELTA)+2):
            if i == 0:
                continue
            if i == 1:
                f1 = 0
                f2  = err(0*DELTA, S)
            if i == int(0.5 / DELTA)+1:
                F = 0
            else:
                F = err(i*DELTA, S)

            if  f2 - F > 0 and f2 - f1 > 0 or \
                f2 - F < 0 and f2 - f1 < 0:
                new_extreme.append((i-1)*DELTA)
                if max_err < abs(f2):
                    max_err = abs(f2)

            f1 = f2
            f2 = F

        print(f"new_extreme = {new_extreme}")
        print(f"max_err = {max_err}")
        
        if 0 <= max_err_last - max_err <= DELTA:
            print(f"final max_err = {max_err}")
            print(f"final S = {S}")
            break

        max_err_last = max_err
        F_m = new_extreme[:k+2]

    impulse_resp = []
    for i in range(FILTER_LEN):
        if i < k:
            impulse_resp.append(S[k-i]/2)
        elif i == k:
            impulse_resp.append(S[0])
        else:
            impulse_resp.append(S[i-k]/2)

    print(f"h = {impulse_resp}")

    plt.figure(figsize=(10,5))

    t = np.arange(0.0, 0.5, DELTA)
    plt.plot(t, np.array([ freq_response(t[i], S) for i in range(len(t)) ]), lw=2)
    plt.plot(t, np.array([ designed_filter(t[i])    for i in range(len(t)) ]), lw=2)
    plt.legend(['FIR Filter', 'Desire Filter']) 
    plt.title("Frequency Response")
    plt.show()

    plt.figure(figsize=(10,5))
    t = np.arange(0, FILTER_LEN)
    plt.stem(t, np.array([ impulse_resp[i] for i in range(len(t)) ]))
    plt.legend(['h[n]']) 
    plt.title("Impluse Response")
    plt.show()

if __name__ == "__main__":
    main()
