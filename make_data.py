
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License Version 3 as
#published by the Free Software Foundation.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.

#IMPORTS

import sys
import pandas as pd
import numpy as np
from joblib import dump

#VARIABLES

N = 10000 #NUMBER OF SAMPLES
X_E = [2, 3, 1, 1, 3] #X VALUE LIMITS/MULTIPLIERS
FACTOR = 0.35 #GAUSSIAN NOISE FACTOR
FUNCTION = 'exp' #GENERATOR FUNCTION - USE 'sin' OR 'exp'
FILE_NAME = f"./{FUNCTION}.data" #OUTPUT FILENAME

#FUNCTIONS

def generator_function_exp(a, b, c, d, e):
    y = np.multiply(a, np.exp(np.add(b, np.multiply(c, d))))
    z = np.subtract(d, np.exp(np.add(a, e)))
    return y, z
def generator_function_sin(a, b, c, d, e):
    y = np.multiply(a, np.sin(np.add(b, np.multiply(c, d))))
    z = np.subtract(d, np.sin(np.add(a, e)))
    return y, z

#MAIN

x_e = []
z_e = []

for i in range(len(X_E)):
    x_e.append(np.random.rand(N) * X_E[i])

if FUNCTION == 'exp':
    y, z = generator_function_exp(x_e[0], x_e[1], x_e[2], x_e[3], x_e[4])
elif FUNCTION == 'sin':
    y, z = generator_function_sin(x_e[0], x_e[1], x_e[2], x_e[3], x_e[4])
else:
    print(f"Invalid function! Exiting...")
    sys.exit(0)

noise = np.random.randn(N)
z_e.append(y + (FACTOR * noise))
z_e.append(z + (FACTOR * noise))

for i in range(len(x_e)):
    x_e[i] = x_e[i].reshape(-1,1)
for i in range(len(z_e)):
    z_e[i] = z_e[i].reshape(-1,1)

X_e = np.hstack((x_e[0], x_e[1], x_e[2], x_e[3], x_e[4]))
y_e = np.hstack((z_e[0], z_e[1]))

v=int(N/2)

print(f"Sample: {v}, in: {X_e[v]}, out: {y_e[v]}")

columns = ['a', 'b', 'c', 'd', 'e', 'y', 'z', 'y_act', 'z_act']

y = np.array(y).reshape(-1,1)
z = np.array(z).reshape(-1,1)

all_e = np.hstack((X_e, y_e, y, z))

df = pd.DataFrame(all_e, columns=columns)

print(type(df))
print(df.head(3))

uin = input("Save data? [Y/n]\n")
if uin == 'Y':
    dump(df, FILE_NAME)
