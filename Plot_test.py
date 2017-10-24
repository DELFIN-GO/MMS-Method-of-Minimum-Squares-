import numpy as np
import matplotlib.pyplot as plt


def f(a):
    return [a[0], a[1], 0, 0]


data = open("input.txt")
plot_data = open("datafile.dat", "w")

input_data = [[float(i) for i in line.split()]
              for line in data.read().split("\n")
              if len(line) != 0]

calculated_input = [f(cur_x_y) for cur_x_y in input_data]
print_result = "\n".join([" ".join([str(i)
                                    for i in cur_xy])
                          for cur_xy in calculated_input])

plot_data.write(print_result)

data.close()
plot_data.close()

# Апроксимация(МНК)
x = np.array(calculated_input)[:, 0]
y = np.array(calculated_input)[:, 1]
k = int(input("Enter the polinomial power: "))

Aproximate_lin_sys_A = np.array([[sum([xi ** (i + j) for xi in x])
                                  for j in range(k + 1)]
                                 for i in range(k + 1)])
Aproximate_lin_sys_B = np.array([sum([y[j] * x[j] ** i for j in range(len(x))])
                                 for i in range(k + 1)])
polynomial_coefficients = np.linalg.solve(Aproximate_lin_sys_A, Aproximate_lin_sys_B)


def polinomial_func(val):
    ans = np.zeros(val.shape)
    for i in range(len(polynomial_coefficients)):
        ans += polynomial_coefficients[i] * val ** i
    return ans


# Построение графика

def calc_lag(arr, step=100):
    return (max(arr) - min(arr)) / step


def calc_limits(arr, limits_size=0.15):
    min_element = min(arr)
    max_element = max(arr)
    delta_element = max_element - min_element
    return (min_element - delta_element * limits_size,
            max_element + delta_element * limits_size)


x_plot = np.arange(*calc_limits(x), calc_lag(x))
y_plot = polinomial_func(x_plot)

plt.scatter(x, y)
plt.plot(x_plot, y_plot)
plt.grid(True)


# подсчёт областей рисования
def local_max_min(arr):
    return (np.array([arr[i]
                     for i in range(1, len(arr) - 1)
                     if max(arr[i-1], arr[i], arr[i+1]) == arr[i]]),
            np.array([arr[i]
                      for i in range(1, len(arr) - 1)
                      if min(arr[i - 1], arr[i], arr[i + 1]) == arr[i]]))


plt.xlim(*calc_limits(x))
plt.ylim(*calc_limits(np.hstack((y, *local_max_min(y_plot)))))

plt.show()
