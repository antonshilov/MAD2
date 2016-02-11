from math import sqrt
from random import random

import numpy
import scipy.misc

lin = 'lin'
quad = 'quad'


def build_model(object_type, model_type, var_number, measure_number, base_points, intervals, function, border_cochran,
                border_fisher, border_student, m, d):
    exp_num = experiment_number(object_type, model_type, var_number)
    reg_coef_num = regression_coef_number(object_type, model_type, var_number)
    plan_matrix = get_plan_matrix(object_type, model_type, var_number)
    y_object, result_object = object_y_params(exp_num, var_number, measure_number, plan_matrix, base_points, intervals,
                                              m, d, function)
    object_disp = object_dispersion(y_object, result_object)
    if not check_kochran(border_cochran, object_disp):
        return 'disp_not_uniform'
    object_disp_eval = sum(object_disp) / len(object_disp)
    b_coef = regression_coef_model(reg_coef_num, plan_matrix, result_object)
    if model_type == quad and object_type == quad:
        b_coef[0] = ckp(b_coef, exp_num, var_number, plan_matrix)
    # Тут должно быть цкп
    if not check_student(b_coef, object_disp_eval, exp_num, border_student):
        return 'coef_insign'
    model_function = build_model_function(object_type, model_type, var_number, base_points, intervals, b_coef)
    result_model = model_y_params(exp_num, var_number, plan_matrix, base_points, intervals, model_function)

    adeq_disp = adequacy_dispersion(result_object, result_model, exp_num, var_number, measure_number)
    is_model_adeq, fish = check_fisher(border_fisher, adeq_disp, object_disp_eval)
    return {'model_function': model_function, 'adeq_disp': adeq_disp, 'object_disp_eval': object_disp_eval,
            'is_model_adeq': is_model_adeq, 'f': fish}


def ckp(bcoef, exp_num, var_number, plan_matrix):
    res = bcoef[0]
    for i in range(1, var_number + 1):
        tmp = 0
        for j in range(exp_num):
            tmp += plan_matrix[j][i] ** 2
        tmp /= exp_num
        tmp *= bcoef[len(bcoef) - var_number - 1 + i]
        res -= tmp
    return res


def build_model_function(object_type, model_type, var_number, base_points, intervals, b_coef):
    if var_number == 2:
        if object_type == lin:
            return '{0} + {1} * (u1 - {2}) + {3} * (u2 - {4})'.format(b_coef[0], round(b_coef[1] / intervals[0], 2),
                                                                      base_points[0],
                                                                      round(b_coef[2] / intervals[1], 2),
                                                                      base_points[1])
        elif object_type == quad and model_type == lin:
            return '{0} + {1} * (u1 - {2}) + {3} * ' \
                   '(u2 - {4}) + {5} *' \
                   ' (u1 - {2})*(u2 - {4})'.format(b_coef[0], round(b_coef[1] / intervals[0], 2), base_points[0],
                                                   round(b_coef[2] / intervals[1], 2),
                                                   base_points[1],
                                                   round(b_coef[3] / (intervals[0] * intervals[1]), 2))
        elif model_type == quad and object_type == quad:
            return '{0} + {1} * (u1 - {2}) + {3} * ' \
                   '(u2 - {4}) + {5} *' \
                   ' (u1 - {2})*(u2 - {4})' \
                   '+ {6} * (u1 - {2}) ** 2' \
                   '+ {7} *(u2 - {4}) ** 2'.format(b_coef[0], round(b_coef[1] / intervals[0], 2), base_points[0],
                                                   round(b_coef[2] / intervals[1], 2),
                                                   base_points[1],
                                                   round(b_coef[3] / (intervals[0] * intervals[1]), 2),
                                                   round(b_coef[4] / (intervals[0] ** 2), 2),
                                                   round(b_coef[5] / (intervals[1] ** 2), 2))
    elif var_number == 3:
        if object_type == lin:
            return '{0} + {1} * (u1 - {2}) ' \
                   '+ {3} * (u2 - {4})' \
                   ' + {5} * (u3 - {6})'.format(b_coef[0],
                                                round(b_coef[1] / intervals[0], 2),
                                                base_points[0],
                                                round(b_coef[2] / intervals[1], 2),
                                                base_points[1],
                                                round(b_coef[3] / intervals[2], 2),
                                                base_points[2])
        elif object_type == quad:
            if model_type == lin:
                return '{0} + {1} * (u1 - {2}) ' \
                       '+ {3} * (u2 - {4})' \
                       '+ {5} * (u3 - {6})' \
                       '+ {7} * (u1 - {2})*(u2 - {4})' \
                       '+ {8} * (u2 - {4})*(u3 - {6})' \
                       '+ {9} + (u1 - {2})*(u3 - {6})'.format(b_coef[0],
                                                              round(b_coef[1] / intervals[0], 2),
                                                              base_points[0],
                                                              round(b_coef[2] / intervals[1], 2),
                                                              base_points[1],
                                                              round(b_coef[3] / intervals[2], 2),
                                                              base_points[2],
                                                              round(b_coef[4] / (intervals[0] * intervals[1]), 2),
                                                              round(b_coef[5] / (intervals[1] * intervals[2]), 2),
                                                              round(b_coef[6] / (intervals[0] * intervals[2]), 2),
                                                              )
            elif model_type == quad:
                return '{0} + {1} * (u1 - {2}) ' \
                       '+ {3} * (u2 - {4})' \
                       '+ {5} * (u3 - {6})' \
                       '+ {7} * (u1 - {2})*(u2 - {4})' \
                       '+ {8} * (u2 - {4})*(u3 - {6})' \
                       '+ {9} * (u1 - {2})*(u3 - {6})' \
                       '+ {10} * (u1 - {2}) ** 2' \
                       '+ {11} * (u2 - {4}) ** 2' \
                       '+ {12} * (u3 - {6}) ** 2'.format(b_coef[0],
                                                         round(b_coef[1] / intervals[0], 2),
                                                         base_points[0],
                                                         round(b_coef[2] / intervals[1], 2),
                                                         base_points[1],
                                                         round(b_coef[3] / intervals[2], 2),
                                                         base_points[2],
                                                         round(b_coef[4] / (intervals[0] * intervals[1]), 2),
                                                         round(b_coef[5] / (intervals[1] * intervals[2]), 2),
                                                         round(b_coef[6] / (intervals[0] * intervals[2]), 2),
                                                         round(b_coef[7] / (intervals[0] ** 2), 2),
                                                         round(b_coef[8] / (intervals[1] ** 2), 2),
                                                         round(b_coef[9] / (intervals[2] ** 2), 2),
                                                         )


def experiment_number(object_type, model_type, var_number):
    if model_type == 'quad' or object_type == 'quad':
        return (2 ** var_number) + var_number * 2 + 1
    else:
        return (2 ** var_number)


# Кол-во коэф. регрессии
def regression_coef_number(object_type, model_type, var_number):
    res = 1
    for i in range(1, var_number + 1):
        combin = scipy.misc.comb(var_number, i)
        if object_type != 'quad' and i != 1:
            combin = 0
        res += combin
    if object_type == 'quad' and model_type == 'quad':
        res += var_number
    return res


def get_plan_matrix(object_type, model_type, var_number):
    # матрица планирования эксперимента линейного объекта от 2х переменных
    plan_lin_2 = [
        [1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [1, -1, -1]]

    # матрица планирования эксперимента линейного объекта от 3х переменных


    plan_lin_3 = [
        [1, 1, 1, 1],
        [1, 1, 1, -1],
        [1, 1, -1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, 1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
        [1, -1, -1, -1]
    ]

    # матрица планирования эксперимента квадратичного объекта 2 переменные
    plan_quad_2 = [
        [1, 1, 1, 1, 1.0 / 3.0, 1.0 / 3.0],
        [1, -1, 1, -1, 1.0 / 3.0, 1.0 / 3.0],
        [1, 1, -1, -1, 1.0 / 3.0, 1.0 / 3.0],
        [1, -1, -1, 1, 1.0 / 3.0, 1.0 / 3.0],
        [1, 1, 0, 0, 1.0 / 3.0, -2.0 / 3.0],
        [1, -1, 0, 0, 1.0 / 3.0, -2.0 / 3.0],
        [1, 0, 1, 0, -2.0 / 3.0, 1.0 / 3.0],
        [1, 0, -1, 0, -2.0 / 3.0, 1.0 / 3.0],
        [1, 0, 0, 0, -2.0 / 3.0, -2.0 / 3.0]
    ]
    # матрица планирования эксперимента квадратичного объекта c 3 переменными
    plan_quad_3 = [
        [1, -1, -1, -1, 1, 1, 1, -1, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1, 1, -1, -1, -1, -1, 1, 1, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1, -1, 1, -1, -1, 1, -1, 1, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1, 1, 1, -1, 1, -1, -1, -1, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1, -1, -1, 1, 1, -1, -1, 1, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1, 1, -1, 1, -1, 1, -1, -1, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1, -1, 1, 1, -1, -1, 1, -1, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1, -1, 0, 0, 0, 0, 0, 0, 1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
        [1, 1, 0, 0, 0, 0, 0, 0, 1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
        [1, 0, -1, 0, 0, 0, 0, 0, -2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0],
        [1, 0, 1, 0, 0, 0, 0, 0, -2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0],
        [1, 0, 0, -1, 0, 0, 0, 0, -2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
        [1, 0, 0, 1, 0, 0, 0, 0, -2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
        [1, 0, 0, 0, 0, 0, 0, 0, -2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]
    ]

    if var_number == 2:
        if model_type == lin:
            if object_type == quad:
                return plan_quad_2
            else:
                return plan_lin_2
        else:
            return plan_quad_2
    elif model_type == lin:
        if object_type == quad:
            return plan_quad_3
        else:
            return plan_lin_3
    else:
        return plan_quad_3


def interference(m, d):
    val = 10
    hid = 0
    for i in range(val):
        hid += random() - 0.5
    hid *= sqrt(d) * sqrt(12 / val)
    hid += m
    return hid


def calc_function(vals, f):
    u1 = u2 = u3 = 0
    if len(vals) == 3:
        u1 = vals[0]
        u2 = vals[1]
        u3 = vals[2]
    else:
        u1 = vals[0]
        u2 = vals[1]
    return eval(f)


def object_y_params(exp_number, var_number, measure_number, plan_matrix, base_points, intervals, m, d, f):
    y_object = numpy.zeros((exp_number, measure_number))
    result_object = []
    for i in range(exp_number):
        vals = []
        for j in range(var_number):
            vals.append(base_points[j] + plan_matrix[i][j + 1] * intervals[j])
        for k in range(measure_number):
            res = calc_function(vals, f)
            res += interference(m, d)
            y_object[i][k] = res
        result_object.append(sum(y_object[i]) / measure_number)
    return y_object, result_object


def model_y_params(exp_number, var_number, plan_matrix, base_points, intervals, f):
    result_model = []
    for i in range(exp_number):
        vals = []
        for j in range(var_number):
            vals.append(base_points[j] + plan_matrix[i][j + 1] * intervals[j])
        result_model.append(calc_function(vals, f))
    return result_model


def object_dispersion(y_object, result_object):
    object_disp = []

    for i in range(len(result_object)):
        sm = 0
        for j in range(len(y_object[0])):
            sm += (y_object[i][j] - result_object[i]) ** 2
        object_disp.append(sm / (len(y_object[0]) - 1))
    return object_disp


def adequacy_dispersion(result_object, result_model, exp_number, var_number, measure_number):
    adeq_disp = 0
    for i in range(exp_number):
        adeq_disp += (result_object[i] - result_model[i]) ** 2
    return adeq_disp / (exp_number * measure_number - var_number - 1)  # TODO:FIX


def regression_coef_model(n, plan_matrix, result_object):
    b_coef = []
    for i in range(int(n)):
        matrix_sum = 0
        coef_sum = 0
        for j in range(len(result_object)):
            coef_sum += plan_matrix[j][i] * result_object[j]
            matrix_sum += plan_matrix[j][i] ** 2
        b_coef.append(coef_sum / matrix_sum)
    return b_coef


def check_kochran(border, object_disp):
    if (max(object_disp) / sum(object_disp)) > border:
        return False
    return True


def check_student(b_coef, object_disp_eval, exp_number, border):
    insignificant = 0
    for i in range(len(b_coef)):
        t = abs(b_coef[i]) / sqrt(object_disp_eval / exp_number)
        if t < border:
            b_coef[i] = 0  # Возможно не зануляется
            insignificant += 1
    if insignificant == len(b_coef) - 1:
        return False
    return True


def check_fisher(border, adeq_disp, object_disp_eval):
    fish = adeq_disp / object_disp_eval
    if fish > border:
        return False, fish
    else:
        return True, fish

# print(build_model_function(quad, lin, 2, [0., 0., 0.], [5., 5., 5.], [3.2, 1.5, 3.2, 20]))
# print(build_model(lin, quad, 2, 50, [0., 0., 0.], [5., 5., 5.], '2+ 3*u1 + 2* u2', 0.68, 1.43, 6.2, 0.1, 1.))
