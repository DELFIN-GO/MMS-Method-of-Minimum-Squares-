import numpy as np
import matplotlib.pyplot as plt


def users_func(arr):
    return [arr[0] ** 2, 2 * 721.3 * 39.737 * arr[1] ** 2, 0.2, 112]


data = open("input.txt")
plot_data = open("datafile.dat", "w")


def convert_to_float(value_str):
    value_arr = value_str.split("^")
    if len(value_arr) == 1:
        value_arr += [1]
    return float(value_arr[0]) ** float(value_arr[1])


input_data = [[convert_to_float(i)
               for i in line.split() if len(i) != 0]
              for line in data.read().split("\n")
              if len(line) != 0]

calculated_input = [users_func(cur_x_y)
                    for cur_x_y in input_data]
print_result = "\n".join([" ".join([str(i)
                                    for i in cur_xy])
                          for cur_xy in calculated_input])

plot_data.write(print_result)

data.close()
plot_data.close()

# Апроксимация(МНК)
Aproximate_data_x = np.array(calculated_input)[:, 0]
Aproximate_data_y = np.array(calculated_input)[:, 1]
polinomial_power = input("Enter the polinomial power: ")
while not polinomial_power.isdigit():
    print("Incorrect input")
    polinomial_power = input("Enter the polinomial power: ")
polinomial_power = int(polinomial_power)

Aproximate_lin_sys_A = np.array([[sum([xi ** (i + j) for xi in Aproximate_data_x])
                                  for j in range(polinomial_power + 1)]
                                 for i in range(polinomial_power + 1)])
Aproximate_lin_sys_B = np.array([sum([Aproximate_data_y[j] * Aproximate_data_x[j] ** i
                                      for j in range(len(Aproximate_data_x))])
                                 for i in range(polinomial_power + 1)])
polynomial_coefficients = np.linalg.solve(Aproximate_lin_sys_A, Aproximate_lin_sys_B)


def polinomial_func(val_arr):
    ans = np.zeros(val_arr.shape)
    for i in range(len(polynomial_coefficients)):
        ans += polynomial_coefficients[i] * val_arr ** i
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


x_plot = np.arange(*calc_limits(Aproximate_data_x), calc_lag(Aproximate_data_x))
y_plot = polinomial_func(x_plot)

plt.scatter(Aproximate_data_x, Aproximate_data_y)
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


plt.xlim(*calc_limits(Aproximate_data_x))
plt.ylim(*calc_limits(np.hstack((Aproximate_data_y, *local_max_min(y_plot)))))

plt.show()
