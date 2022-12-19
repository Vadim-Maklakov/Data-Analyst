# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import os, sys


wrk_dir = os.getcwd() + "/"


df_gender_train = pd.read_csv(wrk_dir + 'gender_train.csv')
df_tr_mcc_codes = pd.read_csv(wrk_dir + 'tr_mcc_codes.csv', sep=";")
df_tr_types = pd.read_csv(wrk_dir + 'tr_types.csv', sep=";")

# Простые задачи
# 1. Считать файл transactions только первые 1000000 строк.
df_transactions_raw = pd.read_excel(wrk_dir+"transactions.xlsx", nrows=1000000)

# EDA для  transactions_raw, оставляем копию что бы была под рукой
df_transactions_raw_isna = df_transactions_raw.isna().sum()
# 422 143 NaN values in column term_id

df_transactions_raw_row_dupl = df_transactions_raw.duplicated().sum()
# 3 335 duplicated rows

# - заполнить Nan значение знаком '-'
df_transactions = df_transactions_raw.fillna('-')

# 2. изменить название столбца term_id на term_id_1
df_transactions = df_transactions.rename(columns = {'term_id': 'term_id_1' }) 


# 3. Просуммировать значения в столбце amount, вывести минимальное 
# и максимальное значение, просуммировать значения больше 0

# решение конвертим столбец amount в str
df_transactions["amount"] = df_transactions["amount"].astype(str)

# столбец содержит ошибочные данные в datetime подобрые '2029-06-01 00:00:00'
# удаляем олибочные данные
date_pat =r"([12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]))"
df_transactions = df_transactions[df_transactions.amount.str.contains(date_pat)== False]

# конвертим во float
df_transactions.amount = df_transactions.amount.astype(float)

# общая сумма
amount_total = df_transactions.amount.sum()

# максимальное и минимальное значение amount
amount_max = df_transactions.amount.max()
amount_min = df_transactions.amount.min()

# сумма положительных элементов
amount_plus_total = df_transactions[df_transactions.amount > 0].amount.sum()


# 4. В столбце tr_datetime отделить время [0 10:23:26, 1 10:19:29] -> [10:23:26, 10:19:29]
# решение - определям паттерн для первых цифр и пробела и заменяем на ''
fnum_wsp_pat = r"(^\d+\s+)"
df_transactions.tr_datetime = df_transactions.tr_datetime.str.replace(fnum_wsp_pat, '')


# 5. В столбце tr_datetime имеются ошибочные данные в секундах, заменить 
# 60 на 59 [06:27:60] -> [06:27:59].  Сделать двумя способами, в одном из них 
# использовать try - except

# простой pd.apply
def correct_sec(row):
    if row[6:8] == '60':
        row = row[:6] + '59'
    return row

# фиксим неправильные секунды и конвертим tr_datetime во время
df_transactions.tr_datetime = df_transactions.tr_datetime.apply(correct_sec)
df_transactions.tr_datetime = pd.to_datetime(df_transactions.tr_datetime)
        
# c try-except перехват ошибки при неправильном значении аргумента функции и вывод
try:
  correct_sec(12697)  
except Exception as X:
    print("Проверьте правильность формата и типа данных! {}".format(X))         
# Проверьте правильность формата и типа данных! 'int' object is not subscriptable

# c try-except ниже значение t5 для корректного типа данных функции
try:
  t5 = correct_sec('12:12:60')  
except Exception as X:
    print("Проверьте правильность формата и типа данных! {}".format(X))         
# t5 = '12:12:59' после вызова correct_sec

# Пытаемся конвертировать в datetime tr_datetime  в исходном df_transactions_raw 
try:
    df_transactions_raw.tr_datetime = pd.to_datetime(df_transactions_raw.tr_datetime)  
except Exception as X:
    print("Проверьте правильность формата и типа данных! {}".format(X))         
# Проверьте правильность формата и типа данных! day is out of range for month: 0 10:23:26 present at position 0


# 6.Отфильтровать датафрейм по столбцу mcc_code > 6000 и столбцу 
# tr_datetime позже 10:00:00, посчитать среднее значение столбца amount

# convert str to int in column mcc_code 
df_transactions.mcc_code = df_transactions.mcc_code.astype(int) 

# Применяем фильтр к новому df_transactions_filtered 
df_transactions_filtered = df_transactions[
    (df_transactions.tr_datetime > '2022-12-15 10:00:00') 
    &  (df_transactions.mcc_code > 6000)] 


# 7.Вывести дф в обратном порядке, вывести каждую 10 строку дф.

# обратный порядок
df_transactions_revers = df_transactions.iloc[::-1].reset_index(drop = True);

#Выводим каждую 10-ю строку в реврерсивном df index начинается с 0 поэтому + 1 к idx
df_transactions_revers_one_to_ten = df_transactions_revers.iloc[
    lambda x : (x.index+1)%10 == 0]  


# 8. Сгруппировать по customer_id, создать новый дф из строк в которых 
# среднее значение больше 0.

# Решение - группируем по customer_id и 'amount':'mean' 
df_transactions_gb_amount = df_transactions.groupby('customer_id')\
    .agg({'amount':'mean'}) 

# Выводим custoer_id с mean amount > 0
df_transactions_gb_amount_over_zero = df_transactions_gb_amount[
    df_transactions_gb_amount.amount > 0]


# 9.  добавить в новый дф столбец из рандомных чисел от 0 до 100
df_transactions["rand_int"] = np.random.randint(0, high = 101, 
                                                size = df_transactions.shape[0] )

# 10. Используя регулярные выражения отфильтровать по столбцу customer_id 
# строки в которых содержатся цифры 3, 6, 8.

pat_368 = r"(3|6|8)"
df_transactions['customer_id_str'] = df_transactions.customer_id.astype(str)

df_transactions_368 = df_transactions[
    df_transactions.customer_id_str.str.contains(pat_368) == True]
# итого 941 563 строк содержат цифру либо 3 либо 6 либо 8 в комбинации от
# одной и выше штук



# Задание 1
# удаляем дупли
df_transactions = df_transactions[~df_transactions.duplicated()] 

# Удаляем ненужные  customer_id_str,  rand_int 
df_transactions = df_transactions.drop(columns=['customer_id_str', 'rand_int' ])

# В датафрейме transactions задайте столбец customer_id в качестве индекса.
df_transactions =  df_transactions.set_index('customer_id')

# Выделите клиента с максимальной суммой транзакции 
# (то есть с максимальным приходом на карту). (*)
max_income_client = df_transactions[df_transactions.amount 
                                    == df_transactions.amount.max()]

# Найдите у него наиболее часто встречающийся модуль суммы приходов/расходов. (**)

max_income_client_val_count = df_transactions.iloc[
    lambda x: x.index == max_income_client.index[0]].amount.value_counts()
# =============================================================================
# Извлекаем ручками, правильный ответ
# Три чаще всего повторяемых платежа сортировка desc по количеству повторяемых 
# платежей
# - 3) 22459.16 — 25 раз
# - 2) 15721.41 — 12 раз
# - 1) 1122957.89  - 5 раз
# Самые редкие платежи
# - 4) 13475494.63  - 2 раза
# - 5) 107407.78 — всего 1 раз 
# - 6) 65019.26 — всего 1 раз
# =============================================================================

# Cоединение df
# =============================================================================
# Соедините transactions с всеми остальными таблицами (tr_mcc_codes, 
# tr_types, gender_train). Причём с gender_train необходимо смёрджиться с 
# помощью left join, а с оставшимися датафреймами - через inner. 
# После получения результата таблицы gender_train, tr_types, tr_mcc_codes 
# можно удалить. В результате соединения датафреймов должно получиться 
# 999584 строки.
# =============================================================================

df_merge = pd.merge(df_transactions, df_gender_train,  how="left", 
                    on = 'customer_id')
df_merge = pd.merge(df_merge, df_tr_mcc_codes,  how="inner", 
                    on = 'mcc_code')
df_merge = pd.merge(df_merge, df_tr_types,  how="inner", 
                    on = 'tr_type')
# У меня получилось  998 676 строк так как я удалил  вначале дубли которые 
# надо удалять так как они портят реальную картину


# =============================================================================
# Задание 2:
# •	Выделите из поля tr_datetime относительный день tr_day (первое число до 
# точного времени). (*)
# •	Отфильтруйте строки таким образом, чтобы оставить только те транзакции, 
# у которых в соответствующий относительный день tr_day количество уникальных 
# MCC кодов при транзакциях было больше 75 (можно воспользоваться функцией nunique())
# =============================================================================

# По поводу выделения отностиельного дня  не совсем понятно, так как с учетом картчек
# оплаты с мобильного платеж может быть в любое время суток особенно если 
# сервер платежной БД стоит в другом часовом поясе - оставляю все как есть так 
# как не понятно что брать за начало и конец интервала
# по факту же 
# df_merge.tr_datetime.min() = Timestamp('2022-12-15 00:00:00')
# df_merge.tr_datetime.max() =  Timestamp('2022-12-15 23:59:59'

# Фильтрацию строк делаю через группировку счетчика MCC  кодов > 75 по customer_id
df_merge_mcc_cnt_gb = df_merge.groupby('customer_id', as_index = False)\
    .agg({'mcc_code':'count'}) 
df_merge_mcc_cnt_gb = df_merge_mcc_cnt_gb[df_merge_mcc_cnt_gb.mcc_code > 75] 
custmoer_id_lst = list(df_merge_mcc_cnt_gb.customer_id.values)

# df_merge_filt где количество счетчика MCC  кодов > 75 для каждого клиента
df_merge_filt =  df_merge[df_merge.customer_id.isin(custmoer_id_lst)]


# Группируем что бы ответить на вопросы по заданию 2
df_merge_filt_gb = df_merge_filt.groupby(['mcc_description', 'gender'], as_index = False)\
    .agg({'mcc_code':'count', 'amount':['sum', 'mean'] }) 
# =============================================================================
# Вопросы и ответы задания 2
# Сгруппируйте полученный отфильтрованный датафрейм по MCC коду и полу, 
# после чего, пронализировав результат, выберите верные варианты ответов ниже 
# •1) gender == 0 - женщины, gender == 1 — мужчины — правильный  ответ исходя 
# из автомобильных запчастей, шин и аксессуаров для женщин
#
# •2) gender == 1 - женщины, gender == 0 — мужчины — не правильный ответ
#
# •3) Абсолютное значение медианы с типом "Флористика" (расходов/приходов) 
# у мужчин выше той же медианы у женщин — правильный ответ 
# 
# •4) Абсолютное значение медианы женских трат (расходов/приходов) на ценные 
# бумаги выше мужских — неправильный ответ,  модуль медианы трат на ценные бумаги 
# выше у мужчин
# 
# •5) Абсолютное значение медианы женских трат (расходов/приходов) 
# в категории "Бары, коктейль-бары, дискотеки, ночные клубы и таверны — 
# места продажи алкогольных напитков" ниже мужских - правильный ответ 
# =============================================================================


