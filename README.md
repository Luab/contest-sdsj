# Sberbank Data Science Contest

Решение соревнования от Сбербанка (https://contest.sdsj.ru). 

| # | Участник |  Общий балл | Задача A | Задача B | Задача C | 
|----------|----------|----------|----------|----------|----------|
| 77 | mrk-andreev (DMIA) | 358.8919 | 0.896297 (169.4049) | 1.575759 (189.4871) | 1.647985 (0.0000) |
| (11%)  | | |  14 (2%) | 59 (8%) | 194 (27%) |

## Описание задач

**Задача A**. Для клиентов, у которых неизвестен пол (которых нет в обучающей выборке ```customers_gender_train.csv```, но которые есть в ```transactions.csv```), необходимо предсказать вероятность быть мужского пола (значение ```1```).
Качество оценивается как площадь под ROC кривой (AUC-ROC) между прогнозами и истинными значениями.

**Задача B**. Необходимо предсказать объем трат по каждой из 184 категорий на каждый день следующего месяца. Итоговый файл должен содержать предсказания по 184 * 30 = 5520 объектам. Объем трат в конкретной категории считается как сумма всех расходных транзакций в текущей категории по всем пользователям.
Качество оценивается метрикой RMSLE (со смещением 500) между прогнозами и истинными значениями во всех категориях и днях.

**Задача C**. Необходимо предсказать объем трат в следующем месяце в каждой из 184 категорий для каждого ```customer_id```, которого нет в обучающей выборке ```customers_gender_train.csv```, но есть в ```transactions.csv```. Итоговый файл должен содержать предсказания по 184 * 3 000 = 552 000 объектам.
Объем трат пользователя в конкретной категории считается как сумма всех расходных транзакций этого пользователя в текущей категории.
Качество оценивается метрикой RMSLE (со смещением 1) между прогнозами и истинными значениями по всем категориям и клиентам.
