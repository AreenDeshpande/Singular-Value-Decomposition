#@title Singular Value Decomposition

import numpy as np
from IPython.display import display, Latex
from scipy.linalg import null_space


def array_to_list(array):
    mlist = []
    for row in array:
        rlist = []
        if row.size == 1:
            rlist.append(row)
            mlist.append(rlist)
            continue
        for element in row:
            rlist.append(element)
        mlist.append(rlist)
    return mlist

def append_to_matrix(array, appendee):
    mT = np.transpose(array)
    aT = np.transpose(appendee)
    mTlist = array_to_list(mT)
    aTlist = array_to_list(aT)
    mTlist.extend(aTlist)
    mT = np.array(mTlist)
    return np.transpose(mT)

# Define a function to create Latex code from arrays to display as matrices.

def array_to_bmatrix(array):
    """Makes Latex code for the array given as parameter."""
    begin = '\\begin{bmatrix} \n'
    data = ''
    for line in array:
        if line.size == 1:
            if round(line) == round(line, 3):
                line = round(line)
                t = 'd'
            else:
                t = '.3f'
            data = data + f' %{t} &'%line
            data = data + r' \\'
            data = data + '\n'
            continue
        for element in line:
            if round(element) == round(element, 3):
                element = round(element)
                t = 'd'
            else:
                t = '.3f'
            data = data + f' %{t} &'%element

        data = data + r' \\'
        data = data + '\n'
    end = '\end{bmatrix}'
    return (begin + data + end)

def round_lows(ar):
    for i in range(len(ar)):
        if ar[i].size == 1:
            if 0 > ar[i] > -1e-3:
                ar[i] = 0
        else:
            for j in range(len(ar[i])):
                if 0 > ar[i][j] > -1e-3:
                    ar[i][j] = 0
    return ar

def printable(mat):
    return array_to_bmatrix(round_lows(mat))

# Take input from user for the matrix

print()
print("Enter the order of Matrix :")
m, n = tuple(map(int, input("m,n : ").split(',')))

print()
print("Enter the entries row by row, separated by commas :\n")

matrix_list = []

for row_number in range(1, m+1):
    current_row = list(map(int, input(f"Row {row_number} : ").split(',')))
    matrix_list.append(current_row)

# Convert the list of matrix into an array
A = np.array(matrix_list).reshape(m, n)

# transpose of A
AT = np.transpose(A)

ATxA = np.matmul(AT, A)
AxAT = np.matmul(A, AT)

# Get sigma and V

eigATxA, v = np.linalg.eig(ATxA)

for vi in range(len(v)):
    v[vi] *= -1

orgl_eig = eigATxA[:]

for i in range(len(eigATxA)):
    if eigATxA[i] < 0:
        eigATxA[i] = abs(eigATxA[i])
    eigATxA[i] = round(eigATxA[i], 3)

sqrteig = np.sqrt(eigATxA)
sorted_indices = np.argsort(sqrteig)[::-1]
sortedsqrt = np.sort(sqrteig)[::-1]
V = v[:, sorted_indices]
VT = np.transpose(V)

# Creating m×n sigma matrix

sigmalist = []
sqrtlist = array_to_list(sortedsqrt)

for i in range(m):
    sigmalist.append([])
    for j in range(n):
        if i == j:
            sigmalist[i].append(sqrtlist[i][0])
        else:
            sigmalist[i].append(0)

sigma = np.array(sigmalist)

# Find U matrix from V

iterator = iter(np.transpose(np.matmul(A, V)))
uTlist = []
for element in sortedsqrt:
    if element != 0:
        uTlist.append((1/element) * next(iterator))

uT = np.array(uTlist)
u = np.transpose(uT)

if m > n:
    almost_U, _ = np.linalg.qr(u)
    UT = np.transpose(u)

if len(uT) < m:
    B = null_space(AxAT)*-1  # null space
    U = append_to_matrix(u, B)

else:
    U = u

# Display as a matrix using Latex

print()
display(Latex(r" A  = "+ printable(A)))
print()
display(Latex(r"A^T = " + printable(AT)))
print()
display(Latex(r"A^T A = " + printable(ATxA)))
print()
display(Latex(r"Eigen \space values \space of \space A^T A = "
              + printable(orgl_eig)))
print()
display(Latex(r"Square \space root \space of \space eigen \space values = "
              + printable(sqrteig)))
print()
if eigATxA.all() != orgl_eig.all():
    display(Latex(r"Absolute \space values \space of \space \
    square \space roots =" + printable(eigATxA)))
    print()
display(Latex(r"Diagonal \space elements \space of \space \Sigma = "
              + printable(sortedsqrt)))
print()
display(Latex(r"\Sigma = " + printable(sigma)))
print()
display(Latex(r"Orthonormal \space eigenvectors \space of \space A^TA:"))
print()

for i in range(len(VT)):
    display(Latex(f"v_{i+1}" + "=" + printable(np.transpose(VT[i]))
    + r", \space" + f"\\sigma_{i+1}" + "="
                  + str(float(round(sortedsqrt[i], 3)))))
    print()

display(Latex(r"V = " + printable(V)))
print()
display(Latex(r"V^T = " + printable(VT)))
print()

if m <= n:
    display(Latex(r"By \space using \space u_i = Av_i / \sigma_i,"))
else:
    display(Latex(r"Through \space Gram-schmidt \space process, \
    \space column \space vectors \space of \space U:"))
print()

x = 0
for value in sortedsqrt:
    if value != 0:
        display(Latex(f"u_{x+1}" + "=" + printable(np.transpose(uT[x]))))
        x += 1
        print()

if len(uT) < m:
    display(Latex(r"Remaining \space eigenvectors \space of \space U \
    \space using \space null \space space:"))
    print()

    for vector in np.transpose(B):
        display(Latex(f"u_{x+1}" + "=" + printable(np.transpose(vector))))
        x += 1
        print()

display(Latex(r"U = " + printable(U)))
print()

# Cross check the code by multiplying the three matrices
product = np.matmul(np.matmul(U, sigma), VT)

display(Latex(r"U \Sigma V^T =" + printable(U) + printable(sigma)
        + printable(V) + "^T=" + printable(product)))
print()
