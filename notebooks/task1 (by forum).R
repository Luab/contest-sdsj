#
# Konstantin Ivanin (@const) выложил на форум этот скрипт
# В моем решении "task1_solution_by_const.csv" смешивается с моим решением
# Результат автора 0.888662, мое решение, в которое вмешено это решение, дает 0.896297 :)
#
# На форуме: https://contest.sdsj.ru/#/posts/show/41?_k=2fvn0t
# В gist: (https://gist.github.com/ivaninkv/35e0f45dc1039d8719cc5d58a4b22034)

rm(list = ls())
gc()

# загрузка библиотек
library(data.table)
library(ggplot2)
library(xgboost)
library(caret)
library(Metrics)

rate <- pi^exp(1) #22.4591577184

# загрузка данных
tran <- fread('transactions.csv')
cust <- fread('customers_gender_train.csv')
mcc <- fread('tr_mcc_codes.csv')
tr_type <- fread('tr_types.csv')

# добавляем признаки по дням недели и продолжительности дней, сколько человек являлся клиентом (dur)
mcc$mcc_code <- paste('mcc', as.character(mcc$mcc_code), sep = '_')
tran[, amount := round(amount / rate, 2)]
tran[, c('day', 'time') := tstrsplit(tr_datetime, ' ', fixed = TRUE, type.convert = TRUE)]
tran[, c('hh', 'mm', 'ss') := tstrsplit(time, ':', fixed = TRUE, type.convert = TRUE)]
tran[, c('tr_datetime', 'time', 'mm', 'ss', 'term_id') := NULL] 
tran[, dw := day %% 7]
tran[, day := day + 1]
tran[, dw := dw + 1]
tran[, dur := max(day) - min(day), by = customer_id]
tran$l_day <-
  ifelse(tran$hh %in% 6:10,
         'morning',
         ifelse(
           tran$hh %in% 11:18,
           'light_day',
           ifelse(tran$hh %in% 19:23, 'evening', 'night')
         ))

# l_day, траты по времени суток
tmp <- unique(tran[, .(N = (.N / dur)), by = c('customer_id', 'l_day')])
l_day <- dcast(tmp, customer_id ~ l_day, value.var = 'N', fill = 0)
colnames(l_day)[2:length(colnames(l_day))] <- paste('l_day', colnames(l_day)[2:length(colnames(l_day))], sep = '_')

# dw, пишем среднее кол-во транзакций клиента в определенный день недели, усреднение по времени "жизни" клиента
tmp <- unique(tran[, .(N = (.N / dur)), by = c('customer_id', 'dw')])
dw <- dcast(tmp, customer_id ~ dw, value.var = 'N', fill = 0)
colnames(dw)[2:length(colnames(dw))] <- paste('dw', colnames(dw)[2:length(colnames(dw))], sep = '_')

# money, аналогично предыдущему куску, выносим в отдельные переменные положительные суммы и отрицательные
money <- tran[, .(rich = sum(amount)), by = customer_id]
m_plus <- unique(tran[amount > 0, .(money_plus = sum(amount) / dur), by = customer_id])
m_minus <- unique(tran[amount < 0, .(money_minus = sum(amount) / dur), by = customer_id])
money <- merge(money, m_plus, by = 'customer_id', all.x = T)
money <- merge(money, m_minus, by = 'customer_id', all.x = T)
sum(is.na(money))
money[is.na(money)] = 0
money[, rich := NULL]
rm(list = c('m_plus', 'm_minus'))

# фичи по комбинации customer_id, mcc_code, tr_type
tmp <- unique(tran[, .(mean_val = .N / dur), by = .(customer_id, mcc_code, tr_type)])
pred <- dcast(tmp, customer_id ~ mcc_code + tr_type, value.var = 'mean_val', fill = 0)
rm(list = c('tmp', 'tran'))

# сливаем все вместе
colnames(pred)[2:length(colnames(pred))] <- paste('mcc_tr', colnames(pred)[2:length(colnames(pred))], sep = '_')
pred <- merge(pred, money, by = 'customer_id', all.x = T)
pred <- merge(pred, cust, by = 'customer_id', all.x = T)
pred <- merge(pred, dw, by = 'customer_id', all.x = T)
pred <- merge(pred, l_day, by = 'customer_id', all.x = T)

# удаление столбцов с маленькой суммой
tmp <- colSums(pred)
n_col <- names(tmp[abs(tmp) < 0.01])
pred[, (n_col) := NULL]

# делаем трейн и тест
X <- pred[!is.na(gender)]
y <- X$gender
X[, gender := NULL]
X_pred <- pred[is.na(gender), -c('gender'), with = FALSE]
c_id <- X_pred$customer_id
X[, customer_id := NULL]
X_pred[, customer_id := NULL]
dp <- createDataPartition(as.factor(y), p = 0.7)$Resample1
X_train <- X[dp]
y_train <- y[dp]
X_test <- X[-c(dp)]
y_test <- y[-c(dp)]
rm(list = c('cust', 'pred', 'money', 'l_day'))

# scale, ухудшает результат
#preProc <- preProcess(X, method=c("center", "scale"))
#X <- predict(preProc, X)
#X_pred <- predict(preProc, X_pred)

# тюнинг xgboost, тут уже итоговые параметры
xgbGrid <- expand.grid(
  eta = 0.2,
  nrounds = 150, #OK
  max_depth = 5, #OK
  colsample_bytree = 0.4, #OK
  min_child_weight = 9, #OK
  gamma = 4 #OK
)
fitControl <- trainControl(method = 'repeatedcv', number = 5, repeats = 3, verboseIter = TRUE)

m1 <- train(X, as.factor(y),
            method = 'xgbTree',
            trControl = fitControl,
            metric = "AUC",
            maximize = T, 
            tuneGrid = xgbGrid
)
m1$bestTune

# строим итоговую модель
k <- 256 # дальнейшее увеличение не улучшает модель
param <- list( 
  max_depth = 5,
  eta = 0.2/k,
  gamma = 4,
  colsample_bytree = 0.4,
  min_child_weight = 9,
  subsample = 0.7, 
  objective = 'binary:logistic',
  eval_metric = 'auc'
)

# проверка на отложенной выборке
model <- xgboost(data = as.matrix(X_train), label = y_train, params = param, nrounds = 150*k, print_every_n = 500, early_stopping_rounds = 100)
auc(y_test, predict(model, as.matrix(X_test)))
# cv
folds <- createFolds(y, 5)
for (i in 1:length(folds)) {
  print(auc(y[folds[[i]]], predict(model, as.matrix(X[folds[[i]]]))))
}

# финальная модель
model <- xgboost(data = as.matrix(X), label = y, params = param, nrounds = 150*k, print_every_n = 500, early_stopping_rounds = 100)
f_imp <- xgb.importance(feature_names = colnames(as.matrix(X)), model = model)
xgb.plot.importance(f_imp[Gain > 0.01])

res <- predict(model, as.matrix(X_pred))
ans <- data.frame(c_id, res)
colnames(ans) <- c('customer_id', 'gender')
write.csv(ans, '../data/raw/task1_solution_by_const.csv', quote = F, row.names = F)