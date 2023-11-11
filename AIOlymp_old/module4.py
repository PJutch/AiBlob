import pandas as pd


# Работа со словарём


def count_numbers_solution(list_of_data):
    """
    Принимает на вход список целочисленных значений и подсчитывает, сколько
    раз в списке встретилось каждое из значений.

    Аргументы:
        list_of_data: Список целочисленных значений, котрые нужно пересчитать.

    Возвращаемое значение:
        Словарь, в котором каждому значению из исходного списка сопоставлено
        количество раз, которое это значение встречается в списке.
    """
    counts = {}
    for number in list_of_data:
        counts.setdefault(number, 0)
        counts[number] += 1
    return counts


def count_numbers_tests():
    def compare_dicts(d_calc, d_real):
        if len(d_calc) != len(d_real):
            raise RuntimeError(
                f'Число ключей в словарях различается: должно быть {len(d_real)}, а получилось {len(d_calc)}.')

        for key, value in d_calc.items():
            if d_real[key] != value:
                raise RuntimeError(
                    f'Значение для ключа {key} получилось неправильное: должно быть {d_real[key]}, а получилось {value}.')

    example_1_list = [1, 2, 3, 2, 3, 1, 2, 2, 2]
    example_1_res = {1: 2, 2: 5, 3: 2}

    compare_dicts(count_numbers_solution(example_1_list), example_1_res)

    example_2_list = [8, 2, 10, 5, 7, 0, 1, 9, 5, 9, 1, 1, 2, 10, 5, 3, 6, 2, 2, 9, 5, 7, 10, 2, 6, 8, 7, 7, 7, 10, 5,
                      2, 8, 9, 2, 8, 8, 2, 6, 4, 6, 3, 0, 3, 6, 3, 4, 1, 2, 8]
    example_2_res = {8: 6, 2: 9, 10: 4, 5: 5, 7: 5, 0: 2, 1: 4, 9: 4, 3: 4, 6: 5, 4: 2}

    compare_dicts(count_numbers_solution(example_2_list), example_2_res)

    example_3_list = [5, 2, 3, 2, 5, 1, 4, 7, 5, 5, 5, 5, 8, 2, 5, 3, 5, 6, 1, 3, 6, 6, 6, 3, 6, 2, 4, 5, 1, 4, 6, 6, 1,
                      3, 3, 4, 4, 3, 1, 7, 5, 3, 6, 4, 4, 3, 2, 7, 1, 4, 2, 3, 7, 8, 7, 7, 7, 5, 9, 3, 3, 4, 2, 2, 3, 6,
                      8, 5, 4, 4, 4, 7, 1, 11, 6, 5, 6, 5, 6, 4, 4, 5, 4, 8, 3, 4, 3, 1, 4, 5, 5, 4, 4, 6, 4, 6, 3, 4,
                      7, 8]
    example_3_res = {5: 17, 2: 8, 3: 16, 1: 8, 4: 21, 7: 9, 8: 5, 6: 14, 9: 1, 11: 1}

    compare_dicts(count_numbers_solution(example_3_list), example_3_res)

    print('Все тесты прошли успешно!')


# count_numbers_tests()


# Кодирование факторов с помощью новых бинарных факторов


def new_columns_solution(column):
    """
    Принимает на вход колонку со значениями категориального фактора
    и производит его кодирование, представляя каждую категорию в виде
    колонки со значениями 0 и 1.

    Аргументы:
        column: Колонка со значениями категориального фактора.

    Возвращаемое значение:
        Словарь, в котором каждой категории исходной колонки сопоставлен список из 0 и 1.
        Если на позиции i в этом списке находится 1, то это значит, что в исходной колонке
        в i-ой строке было записано название соответствующей категории.
        Если на позиции i в списке находится 0, то в i-ой строке исходной колонки была
        записана другая категория.
    """
    res = {}
    for category in list(column.unique()):
        res[category] = []

    for value in column:
        for category in res:
            if category == value:
                res[category].append(1)
            else:
                res[category].append(0)

    return res


def new_columns_tests():
    def compare_dicts(d_calc, d_real):
        if len(d_calc) != len(d_real):
            raise RuntimeError(
                f'Число ключей в словарях различается: должно быть {len(d_real)}, а получилось {len(d_calc)}.')

        for key, value in d_calc.items():
            if d_real[key] != value:
                raise RuntimeError(
                    f'Значение для ключа {key} получилось неправильное: должно быть {d_real[key]}, а получилось {value}.')

    example_1_column = pd.Series(['a', 'aa', 'aaa', 'aa'])
    example_1_res = {
        'a': [1, 0, 0, 0],
        'aa': [0, 1, 0, 1],
        'aaa': [0, 0, 1, 0]
    }

    compare_dicts(new_columns_solution(example_1_column), example_1_res)

    example_2_column = pd.Series(['улица', 'помещение', 'помещение', 'улица', 'улица'])
    example_2_res = {
        'улица': [1, 0, 0, 1, 1],
        'помещение': [0, 1, 1, 0, 0]
    }

    compare_dicts(new_columns_solution(example_2_column), example_2_res)

    example_3_column = pd.Series(['--', '++', '--', '^^', '~~', '--', '^^'])
    example_3_res = {
        '--': [1, 0, 1, 0, 0, 1, 0],
        '++': [0, 1, 0, 0, 0, 0, 0],
        '^^': [0, 0, 0, 1, 0, 0, 1],
        '~~': [0, 0, 0, 0, 1, 0, 0],
    }

    compare_dicts(new_columns_solution(example_3_column), example_3_res)

    print('Все тесты прошли успешно!')


# new_columns_tests()


def categorical_to_binary_solution(data, columns):
    """
    Принимает на вход таблицу и производит кодирование представленных
    в ней категориальных факторов. Каждый фактор представляется в виде
    набора бинарных факторов, соответствующих отдельным категориям исходного фактора.

    Аргументы:
        data: Таблица, категориальные факторы которой необходимо закодировать.
        columns: Список колонок, в которых представлены факторы, которые необходимо закодировать.

    Возвращаемое значение:
        Исходная таблица, в которой для каждого из указанных категориальных факторов
        добавлено несколько бинарных факторов — по одному на каждую категорию.

        Для какого-то объекта бинарный фактор принимает значение 1 в том случае,
        если исходный фактор для данного объекта принимал значение соответствующей категории.
        В противном случае значение бинарного фактора равно 0.
    """
    for column in columns:
        new_columns = new_columns_solution(data[column])
        for category in new_columns:
            data[category] = new_columns[category]
    return data


def categorical_to_binary_tests():
    def compare_tables(table_calc, table_real, factors):
        for factor in factors:
            vals = list(table_real[factor].unique())

            for val in vals:
                assert list(table_calc[val]) == list(table_real[val])

    example_1_columns = ['A', 'B']
    example_1_data = [
        ['a', 3],
        ['aa', 5],
        ['aaa', 8],
        ['aa', 13]
    ]
    example_1_table = pd.DataFrame(columns=example_1_columns, data=example_1_data)
    example_1_factors = ['A']

    example_1_res_columns = ['A', 'B', 'a', 'aa', 'aaa']
    example_1_res_data = [
        ['a', 3, 1, 0, 0],
        ['aa', 5, 0, 1, 0],
        ['aaa', 8, 0, 0, 1],
        ['aa', 13, 0, 1, 0]
    ]
    example_1_res_table = pd.DataFrame(columns=example_1_res_columns, data=example_1_res_data)

    compare_tables(categorical_to_binary_solution(example_1_table, example_1_factors),
                   example_1_res_table,
                   example_1_factors)

    example_2_columns = ['номер наблюдения', 'время суток', 'погода']
    example_2_data = [
        [1, 'день', 'дождь'],
        [2, 'ночь', 'снег'],
        [3, 'день', 'солнце'],
        [4, 'день', 'солнце'],
        [5, 'сумерки', 'дождь'],
    ]
    example_2_table = pd.DataFrame(columns=example_2_columns, data=example_2_data)
    example_2_factors = ['время суток', 'погода']

    example_2_res_columns = ['номер наблюдения', 'время суток', 'погода',
                             'день', 'ночь', 'сумерки',
                             'солнце', 'дождь', 'снег']
    example_2_res_data = [
        [1, 'день', 'дождь', 1, 0, 0, 0, 1, 0],
        [2, 'ночь', 'снег', 0, 1, 0, 0, 0, 1],
        [3, 'день', 'солнце', 1, 0, 0, 1, 0, 0],
        [4, 'день', 'солнце', 1, 0, 0, 1, 0, 0],
        [5, 'сумерки', 'дождь', 0, 0, 1, 0, 1, 0],
    ]
    example_2_res_table = pd.DataFrame(columns=example_2_res_columns, data=example_2_res_data)

    compare_tables(categorical_to_binary_solution(example_2_table, example_2_factors),
                   example_2_res_table,
                   example_2_factors)

    print('Все тесты прошли успешно!')


# categorical_to_binary_tests()


# Целевое кодирование


def round_to_2(x):
    """
    Принимает число и возвращает результат его округления
    до 2 знаков после запятой.

    Аргументы:
        x: Число.

    Возвращаемое значение:
        Результат округления числа до 2 знаков после запятой.
    """

    return round(x, 2)


def target_coding_solution(data, factor_column, target_column):
    """
    Принимает таблицу и производит в ней целевое кодирование заданного фактора.

    Аргументы:
        data: Исходная таблица.
        factor_column: Название категориального фактора исходной таблицы, который нужно закодировать.
        target_column: Название колонки, в которой содержится предсказываемое по данным значение.

    Возвращаемое значение:
        Функция ничего не возвращает. Однако в результате её запуска в исходной таблице должна появиться
        новая колонка с названием "encoded", в которой представлен результат кодирования
        фактора "factor_column".
    """
    means = data.groupby(factor_column).aggregate({target_column: 'mean'})[target_column]
    data['encoded'] = [means[val] for val in data[factor_column]]


def target_coding_tests():
    def compare_tables(table_calc, table_real):
        for factor in factors:
            vals = list(table_real[factor].unique())

            for val in vals:
                assert list(table_calc['encoded']) == list(table_real[val])

    example_1_columns = ['Фактор 1', 'Фактор 2', 'Предсказание']
    example_1_data = [
        ['a', 'значение 1', 1],
        ['a', 'значение 2', 3],
        ['aa', 'значение 3', 3]
    ]
    example_1_table = pd.DataFrame(columns=example_1_columns, data=example_1_data)
    example_1_factor = 'Фактор 1'
    example_1_target_factor = 'Предсказание'

    target_coding_solution(example_1_table, example_1_factor, example_1_target_factor)

    example_1_res = [2.0, 2.0, 3.0]

    assert list(example_1_table['encoded']) == example_1_res

    example_2_columns = ['A', 'B', 'C', 'D', 'E']
    example_2_data = [
        ['a', 'bb', 'ccc', 'dd', 1],
        ['aa', 'b', 'ccc', 'ddd', 7],
        ['a', 'bb', 'ccc', 'dd', 6],
        ['aa', 'b', 'ccc', 'd', 5],
        ['a', 'bb', 'ccc', 'd', 1]
    ]
    example_2_table = pd.DataFrame(columns=example_2_columns, data=example_2_data)
    example_2_factor = 'D'
    example_2_target_factor = 'E'

    target_coding_solution(example_2_table, example_2_factor, example_2_target_factor)

    example_2_res = [3.5, 7.0, 3.5, 3.0, 3.0]

    assert list(example_2_table['encoded']) == example_2_res

    print('Все тесты прошли успешно!')


# target_coding_tests()


# Стандартизация факторов


def standard_deviation_solution(column):
    """
    Вычисляет разброс значений в колонке.

    Аргументы:
        column: Колонка с численными значениями.

    Возвращаемое значение:
        Величина разброса значений в колонке, округлённая до 2 знаков после запятой.
    """
    return round_to_2((((column - column.mean()) ** 2).sum() / len(column)) ** .5)


def standard_deviation_tests():
    example_1_column = pd.Series([1, 2, 1, 3, 4, 5])
    example_1_res = 1.49

    assert standard_deviation_solution(example_1_column) == example_1_res

    example_2_column = pd.Series([5, 5, 5, 5, 5])
    example_2_res = 0.0

    assert standard_deviation_solution(example_2_column) == example_2_res

    print('Все тесты прошли успешно!')


# standard_deviation_tests()


def standartize_column_solution(column):
    """
    Производит стандартизацию значений в колонке.

    Аргументы:
        column: Колонка с численными значениями.

    Возвращаемое значение:
        Возвращает колонку со стандартизированными значениями.
        То есть из каждого из исходных значений вычтено среднее, после чего разница разделена на значение разброса.

        Каждое значение в итоговой колонке должно быть округлено до 2 знаков после запятой.
    """
    mean = column.mean()
    deviation = standard_deviation_solution(column)

    if deviation == 0:
        return pd.Series([0.0] * len(column))

    res = pd.Series(round_to_2((value - mean) / deviation) for value in column)
    return res


def standartize_column_tests():
    example_1_column = pd.Series([1, 2, 1, 3, 4, 5])
    example_1_res = [-1.12, -0.45, -1.12, 0.22, 0.89, 1.57]

    assert list(standartize_column_solution(example_1_column)) == example_1_res

    example_2_column = pd.Series([5, 5, 5, 5, 5])
    example_2_res = [0.0, 0.0, 0.0, 0.0, 0.0]

    assert list(standartize_column_solution(example_2_column)) == example_2_res

    print('Все тесты прошли успешно!')


# standartize_column_tests()


# data = pd.Series([33.0, 98.0, 72.0, 117.0, 75.0])
# print(data.mean())
# print(standard_deviation_solution(data))
# print(standartize_column_solution(data))


# Корреляция


def correlation_solution(column_x, column_y):
    """
    Вычисляет корреляцию значений двух колонок.

    Аргументы:
        column_x: Первая колонка с численными значениями.
        column_y: Вторая колонка с численными значениями.

    Возвращаемое значение:
        Возвращает корреляцию значений двух колонок, округлённую до 2 знаков после запятой.
    """
    return round_to_2(((column_x - column_x.mean()) * (column_y - column_y.mean())).sum()
                      / (((column_x - column_x.mean()) ** 2).sum()
                         * ((column_y - column_y.mean()) ** 2).sum()) ** .5)


def correlation_tests():
    example_1_column_x = pd.Series([1, 2, 1, 3, 4, 5])
    example_1_column_y = pd.Series([0, 5, 3, 8, 9, 1])
    example_1_res = 0.29

    assert correlation_solution(example_1_column_x, example_1_column_y) == example_1_res

    example_2_column_x = pd.Series([1, 2, 1, 3, 4, 5])
    example_2_column_y = pd.Series([1, 2, 1, 3, 4, 5])
    example_2_res = 1.0

    assert correlation_solution(example_2_column_x, example_2_column_y) == example_2_res

    print('Все тесты прошли успешно!')


# correlation_tests()


def correlation_table_solution(table):
    """
    Принимает на вход таблицу с данными, а возвращает таблицу со значениями
    корреляции колонок исходной таблицы друг с другом.

    Аргументы:
        table: Таблица с численными значениями в формате pandas.

    Возвращаемое значение:
        Возвращает таблицу корреляции между колонками исходной таблицы в формате списка списков.
        В ячейке таблицы с координатами ij (где i — индекс строки, а j — индекс колонки)
        должна быть записана корреляция значений i-ой колонки исходной таблицы с j-ой колонкой
        исходной таблицы. Значение корреляции должно быть округлено до 2 знаков после запятой.
    """
    res = []
    for column1 in table.columns:
        res.append([])
        for column2 in table.columns:
            res[-1].append(correlation_solution(table[column1], table[column2]))
    return res


def correlation_table_tests():
    example_1_table = [[1, -1],
                       [2, 7],
                       [3, 9],
                       [4, 5]]
    example_1_df = pd.DataFrame(data=example_1_table, columns=['A', 'B'])
    example_1_res = [[1.0, 0.6],
                     [0.6, 1.0]]

    assert correlation_table_solution(example_1_df) == example_1_res

    example_2_table = [[17, 81, 19],
                       [21, 23, 94],
                       [55, 71, 83],
                       [53, 82, 64],
                       [25, 62, 72]]
    example_2_df = pd.DataFrame(data=example_2_table, columns=['D', 'E', 'Q'])
    example_2_res = [[1.0, 0.42, 0.33],
                     [0.42, 1.0, -0.67],
                     [0.33, -0.67, 1.0]]

    assert correlation_table_solution(example_2_df) == example_2_res

    print('Все тесты прошли успешно!')


# correlation_table_tests()
