trace(utils:::unpackPkgZip, edit=TRUE) #-- line 140 sys.sleep(0.5) --> changed to sys.sleep(2)

oldW <- getOption("warn")
options(warn = -1)

#install.packages('stringr')
#install.packages('purrr')
#install.packages('tidyr')
install.packages("tseries")
#install.packages("astsa")

install.packages("tidyr")
install.packages("dplyr")
install.packages("data.table")
install.packages("ggplot2")
install.packages("forecast")
install.packages('lubridate')
install.packages('fpp')
install.packages('prophet')
#install.packages('DescTools')
install.packages('MLmetrics')

#library(purrr)
#library(stringr)
#library(magrittr)
#library(imputeTS)

library(data.table)
library(tseries)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(forecast)
library(fpp)
library(prophet)
#library(DescTools)
library(MLmetrics)
#library(astsa)

# Create functions
f.fcCombined1 <- function(fc1, fc2, fc3){ 
  fcc1 <- (fc1+fc2+fc3)/3
  return (fcc1) 
}

f.fcCombined2 <- function(fc1, fc2, fc3, v1, v2, v3){ 
  fcc2 <- ((v2+v3)/(v1+v2+v3))*fc1+((v1+v3)/(v1+v2+v3))*fc2+((v1+v2)/(v1+v2+v3))*fc3
  return (fcc2) 
}


f.MAE <- function(act, pred){ 
  err <- abs(act - pred)
  err <- ifelse(is.na(act), NA, err)                    # omit if act is NA
  err <- ifelse(is.na(pred) & !is.na(act), NA, err)     # max error if pred is NA and act is available
  MAE <- mean(err)
  return (MAE) 
}

# Function to calculate error variance
f.var_err <- function(act, pred){ 
  sqerr <- abs(act - pred)^2
  sqerr <- ifelse(is.na(act), NA, sqerr)                    # omit if act is NA
  sqerr <- ifelse(is.na(pred) & !is.na(act), NA, sqerr)     # max error if pred is NA and act is available
  v.sqerr <- mean(sqerr)
  return (v.sqerr) 
}

f.residual <- function(act, pred){ 
  res <- act - pred
  res <- ifelse(is.na(act), NA, sqerr)                    # omit if act is NA
  res <- ifelse(is.na(pred) & !is.na(act), NA, sqerr)     # max error if pred is NA and act is available
  return (res) 
}
# f = vector with forecasts, y = vector with actuals
f.MASE <- function(f,y) {
  if(length(f)!=length(y)){ stop("Vector length is not equal") }
  n <- length(f)
  return(mean(abs((y - f) / ((1/(n-1)) * sum(abs(y[2:n]-y[1:n-1]))))))
}


df$err <- f.err(df$act,df$pred)
err.var <- var(df$err, na.rm=TRUE)


f.SMAPE <- function(act, pred){ 
  sm <- 200 * abs(act - pred) / (abs(act) + abs(pred)) # normal formula
  sm <- ifelse(is.na(act), NA, sm)                     # omit if act is NA
  sm <- ifelse(is.na(pred) & !is.na(act), 200, sm)     # max error if pred is NA and act is available
  sm <- ifelse(pred==0 & act==0, 0, sm)                # perfect (arbitrary 0/0=0)
  smape <- mean(sm, na.rm = TRUE)
  return (smape) 
}  

###

# Read data
setwd("D:\\RStudio\\Kaggle\\WebTraffic")
setwd("C:\\JP_TEMP\\CKME136")
#key_2 <- fread("key_2.csv")
dt.data <- fread("train_2.csv")
dt.data <- fread("top10.csv")

# Dimension of dataset --> # of rows (website) x # of columns (daily visits)
c(nrow(dt.data),ncol(dt.data))

dt.data %>% colnames() %>% head(15)
dt.data %>% colnames() %>% tail(5)

# Missing values compared to all data (%)
sum(is.na(dt.data))/((ncol(dt.data)-1)*nrow(dt.data))

# The number of data series with missing values for each row, ordered by number of missing NAs
dt.dataNA <- cbind((dt.data[!complete.cases(dt.data),][,1]),(as.data.table(rowSums(is.na(dt.data[!complete.cases(dt.data),])),row.names = NULL)))
setnames(dt.dataNA, c("Page","NA_Count"))

dt.dataNA <- arrange(dt.dataNA, desc(NA_Count))

dataNA_tmp <- dt.dataNA[1:10,1]
dt.dataNA_top10 <- dt.data[dt.data$Page %in% dataNA_tmp,]

# Pick 10 pages with the highest number of page visits for data exploration
dt.data_tmp <- data.frame(dt.data$Page)
dt.data_tmp$sum = rowSums(dt.data[,2:804], na.rm = TRUE)
data_tmp <- arrange(dt.data_tmp, desc(sum))[1:10,1]

dt.data_top10 <- dt.data[dt.data$Page %in% data_tmp,]

# Plot
plot(t(dt.data_top10[1,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")

# to see a weekly seasonal pattern
plot(t(dt.data_top10[1,2:122]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")

plot(t(dt.data_top10[2,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")
plot(t(dt.data_top10[3,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")
plot(t(dt.data_top10[4,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")
plot(t(dt.data_top10[5,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")
plot(t(dt.data_top10[6,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")
plot(t(dt.data_top10[7,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")
plot(t(dt.data_top10[8,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")
plot(t(dt.data_top10[9,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")
plot(t(dt.data_top10[10,-1]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation", type = "l")

#df.top10 <- data.frame(t(dt.data_top10))
#ggplot(df.top10, aes(row.names,X1) + geom_line() + scale_x_date('month') + ylab("Daily page visits") +  xlab(""))
#ggplot(df.top1, aes(row.names,2)) + geom_line() + scale_x_date('month') + ylab("Daily page visits") +  xlab("")


# Count NAs
any(is.na(dt.data_top10))

# Check stationarity
# The Augmented Dickey-Fuller (ADF) t-statistic test
#   small p-values (< 0.05) suggest the data is stationary and doesn't need to be transformed.

for (i in 1:nrow(dt.data_top10)){
  adf_out <- adf.test(unlist(dt.data_top10[i,-1],use.names = FALSE), alternative = "stationary")
  cat("\nPage:", unlist(dt.data_top10[i,1]))
  cat("\nADF test p-value: ", unlist(adf_out["p.value"]))
  if ((adf_out["p.value"]) > 0.05) {
    cat("\n")
    print("This is a non-stationary time series.")
    cat("\n")
  } else {
    cat("\n")
  }
}


# Create TS object

df.top10 <- data.frame(t(dt.data_top10[,-1]), stringsAsFactors = FALSE)


# Split 70% for training and 30% hold-out for test
# Training: 0.7 * 803 = 562 ==> row c(1:562) ==> From '2015-07-01' until '2017-01-13'
# Test    : 803 - 562 = 241 ==> row c(563:803) ==> From '2017-01-13' until '2017-09-10'

df.top10.train <- df.top10[1:562,]
df.top10.test <- df.top10[563:803,]

# Create empty forecast.metric data frame
df.fc.metrics <- data.frame(Page=character(),
                            ETS_MAE=double(),
                            ETS_MAPE=double(),
                            ETS_MASE=double(),
                            ETS_eVar=double(),
                            ETS_method=character(),
                            ARIMA_MAE=double(),
                            ARIMA_MAPE=double(),
                            ARIMA_MASE=double(),
                            ARIMA_eVar=double(),
                            ARIMA_method=character(),
                            Prophet_MAE=double(),
                            Prophet_MAPE=double(),
                            Prophet_MASE=double(),
                            Prophet_eVar=double(),
                            C1_MAE=double(),
                            C1_MAPE=double(),
                            C1_MASE=double(),
                            C2_MAE=double(),
                            C2_MAPE=double(),
                            C2_MASE=double())

df.fc.Combined1 <- data.frame(Page=character(),pred=integer())
df.fc.Combined2 <- data.frame(Page=character(),pred=integer())

i <- 1

rm(ts.train)
rm(ts.train_clean)
rm(ts.data)
rm(ts.data_clean)
rm(fc.ETS)

ts.train <- ts(df.top10.train[,i],frequency = 7)
autoplot(ts.train)

ts.train_clean <- tsclean(ts.train)
autoplot(ts.train_clean)

ts.test <- ts(df.top10.test[,i],frequency = 7)
ts.test_clean <- tsclean(ts.test)

ts.data <- ts(df.top10[1:803,i],frequency = 7)
ts.data_clean <- tsclean(ts.data)
autoplot(ts.data_clean)

# Using automatic forecasting method
fc.ETS <- forecast(ets(ts.train_clean), h = 241)
accuracy(fc.ETS, ts.data)
checkresiduals(fc.ETS)
plot(fc.ETS)

ETS.MASE <- f.MASE(as.numeric(round(fc.ETS$mean)),as.numeric(df.top10.test[,i]))
ETS.MAE <- MAE(round(fc.ETS$mean),df.top10.test[,i], na.rm = TRUE)
ETS.MAPE <- MAPE(round(fc.ETS$mean),df.top10.test[,i], na.rm = TRUE)
ETS.eVar <- f.var_err(round(fc.ETS$mean),df.top10.test[,i])

fc.ARIMA <- forecast(auto.arima(ts.train_clean), h = 241)
ARIMA.MASE <- f.MASE(as.numeric(round(fc.ARIMA$mean)),as.numeric(df.top10.test[,i]))
ARIMA.MAE <- MAE(round(fc.ARIMA$mean),df.top10.test[,i], na.rm = TRUE)
ARIMA.MAPE <- MAPE(round(fc.ARIMA$mean),df.top10.test[,i], na.rm = TRUE)
ARIMA.eVar <- f.var_err(round(fc.ARIMA$mean),df.top10.test[,i])
res.ARIMA <- checkresiduals(fc.ARIMA)
plot(fc.ARIMA)

fc.HW <- hw(ts.train_clean, seasonal = "additive", h = 241)
HW.MASE <- f.MASE(as.numeric(round(fc.HW$mean)),as.numeric(df.top10.test[,i]))
HW.MAE <- MAE(round(fc.HW$mean),df.top10.test[,i], na.rm = TRUE)
HW.MAPE <- MAPE(round(fc.HW$mean),df.top10.test[,i], na.rm = TRUE)
HW.eVar <- f.var_err(round(fc.HW$mean),df.top10.test[,i])

# Prophet
df.train <- data.frame(ds=rownames(df.top10.train),y=df.top10.train[,i])
m.prophet <- prophet(df.train, yearly.seasonality = "FALSE", weekly.seasonality = "auto")
future  <- make_future_dataframe(m.prophet, periods = 241)
fc.Prophet  <- predict(m.prophet, future)
plot(m.prophet,fc.Prophet)

Prophet.MASE <- f.MASE(as.numeric(round(fc.Prophet$yhat[563:803])),as.numeric(df.top10.test[,i]))
Prophet.MAE <- MAE(as.numeric(round(fc.Prophet$yhat[563:803])),as.numeric(df.top10.test[,i]), na.rm = TRUE)
Prophet.MAPE <- MAPE(as.numeric(round(fc.Prophet$yhat[563:803])),as.numeric(df.top10.test[,i]), na.rm = TRUE)
Prophet.eVar <- f.var_err(as.numeric(round(fc.Prophet$yhat[563:803])),as.numeric(df.top10.test[,i]))
plot(m.prophet,fc.Prophet)

rm(res.Prophet)
res.Prophet <- df.train$y - fc.Prophet$yhat[1:562]
checkresiduals(res.Prophet)

fc.C1 <- f.fcCombined1(as.numeric(round(fc.ETS$mean)), as.numeric(round(fc.ARIMA$mean)), as.numeric(round(fc.Prophet$yhat[563:803])))
C1.MASE <- f.MASE(as.numeric(round(fc.C1)),as.numeric(df.top10.test[,i]))
C1.MAE <- MAE(as.numeric(round(fc.C1)),as.numeric(df.top10.test[,i]), na.rm = TRUE)
C1.MAPE <- MAPE(as.numeric(round(fc.C1)),as.numeric(df.top10.test[,i]), na.rm = TRUE)

fc.C2 <- f.fcCombined2(as.numeric(round(fc.ETS$mean)), as.numeric(round(fc.ARIMA$mean)), as.numeric(round(fc.Prophet$yhat[563:803])),ETS.eVar,ARIMA.eVar,Prophet.eVar)
C2.MASE <- f.MASE(as.numeric(round(fc.C2)),as.numeric(df.top10.test[,i]))
C2.MAE <- MAE(as.numeric(round(fc.C2)),as.numeric(df.top10.test[,i]), na.rm = TRUE)
C2.MAPE <- MAPE(as.numeric(round(fc.C2)),as.numeric(df.top10.test[,i]), na.rm = TRUE)

# Check residuals


#accuracy(fc.prophet$yhat, df.top10[2:804,i])

# Decomposition
decomp = stl(ts.train_clean)
deseasonal_cnt <- seasadj(decomp)
plot(decomp)


# Model fitting
acf(ts.train_clean)
pacf(ts.train_clean)

acf2(ts.train_clean)

fit.SARIMA <- sarima(ts.train_clean, p=1,d=0,q=0, P=1,D=0,Q=1,S=7)

fc.ARIMA <- forecast(auto.arima(ts.train_clean, stepwise = FALSE),h=241)
fc.ARIMA2 <- forecast(auto.arima(ts.train_clean, stepwise = TRUE),h=241)
summary(fc.ARIMA)
summary(fc.ARIMA2)
autoplot(fc.ARIMA)
accuracy(fc.ARIMA, ts.data)
checkresiduals(fc.ARIMA)

fc.SES <- ses(ts.train_clean, h=241)
autoplot(fc.SES)
accuracy(fc.SES, ts.data)
checkresiduals(fc.SES)

fc.naive <- naive(ts.train_clean, h=241)
autoplot(fc.naive)
accuracy(fc.naive, ts.data)
checkresiduals(fc.naive)

fc.holt <- holt(ts.train_clean, h=241)
autoplot(fc.holt)
accuracy(fc.holt, ts.data)
checkresiduals(fc.holt)

#fc.HW <- forecast(HoltWinters(ts.train_clean, beta = FALSE, gamma = FALSE), h=241, fan=TRUE)
fc.HW <- forecast(HoltWinters(ts.train_clean), h=241, fan=TRUE)
autoplot(fc.HW)
accuracy(fc.HW, myTS)
checkresiduals(fc.HW)

#df.data_ph <- as.data.frame(t(dt.data_all[i,-1]))
df.train <- as.data.frame(t(dt.train[i,-1]))
setnames(df.train, "y")
df.train$ds <- colnames(dt.train[1,-1])

m.prophet <- prophet(df.train,yearly.seasonality = TRUE)
future       <- make_future_dataframe(m.prophet, periods = 241)
fc.prophet     <- predict(m.prophet, future)
plot(m.prophet,fc.prophet)

prophet_plot_components(m.prophet, fc.prophet)

ts.data1 <- ts(t(dt.data_top10[1,-1]),frequency = 365.25, start = c(2015, 7, 1))
ts.data2 <- ts(t(dt.data_top10[2,-1]),frequency = 365.25, start = c(2015, 7, 1))
ts.data3 <- ts(t(dt.data_top10[3,-1]),frequency = 365.25, start = c(2015, 7, 1))
ts.data4 <- ts(t(dt.data_top10[4,-1]),frequency = 365.25, start = c(2015, 7, 1))
ts.data5 <- ts(t(dt.data_top10[5,-1]),frequency = 365.25, start = c(2015, 7, 1))
ts.data6 <- ts(t(dt.data_top10[6,-1]),frequency = 365.25, start = c(2015, 7, 1))
ts.data7 <- ts(t(dt.data_top10[7,-1]),frequency = 365.25, start = c(2015, 7, 1))
ts.data8 <- ts(t(dt.data_top10[8,-1]),frequency = 365.25, start = c(2015, 7, 1))
ts.data9 <- ts(t(dt.data_top10[9,-1]),frequency = 365.25, start = c(2015, 7, 1))
ts.data10 <- ts(t(dt.data_top10[10,-1]),frequency = 365.25, start = c(2015, 7, 1))

autoplot(ts.data1)
autoplot(ts.data2)
autoplot(ts.data3)
autoplot(ts.data4)
autoplot(ts.data5)
autoplot(ts.data6)
autoplot(ts.data7)
autoplot(ts.data8)
autoplot(ts.data9)
autoplot(ts.data10)

ts.data1_clean <- tsclean(ts.data1[,1])
ts.data2_clean <- tsclean(ts.data2[,1])
ts.data3_clean <- tsclean(ts.data3[,1])
ts.data4_clean <- tsclean(ts.data4[,1])
ts.data5_clean <- tsclean(ts.data5[,1])
ts.data6_clean <- tsclean(ts.data6[,1])
ts.data7_clean <- tsclean(ts.data7[,1])
ts.data8_clean <- tsclean(ts.data8[,1])
ts.data9_clean <- tsclean(ts.data9[,1])
ts.data10_clean <- tsclean(ts.data10[,1])

autoplot(ts.data1_clean)
autoplot(ts.data2_clean)
autoplot(ts.data3_clean)
autoplot(ts.data4_clean)
autoplot(ts.data5_clean)
autoplot(ts.data6_clean)
autoplot(ts.data7_clean)
autoplot(ts.data8_clean)
autoplot(ts.data9_clean)
autoplot(ts.data10_clean)

# Decomposition
decomp = stl(ts.data1_clean, s.window="periodic")
deseasonal_cnt <- seasadj(decomp)
plot(decomp)

# Find out lag order. PACF => AR lag order, ACF => MA lag order

acf(ts.data1_clean)
pacf(ts.data1_clean)


# Split 70% for training and 30% hold-out for test
# Training: 0.7 * 803 = 562 ==> column c(2:563) ==> From '2015-07-01' until '2017-01-13'
# Test    : 803 - 562 = 241 ==> column c(564:803) ==> From '2017-01-13' until '2017-09-10'

i <- 1

ts.train <- ts.data1_clean[]
myTS <- ts(t(dt.data_all[i,-1]),frequency = 365.25, start = c(2015, 7, 1))


myTS <- ts.data1_clean

# Plotting
autoplot(myTS)


fc.ARIMA <- forecast(auto.arima(myTS),h=241)
autoplot(fc.ARIMA)
accuracy(fc.ARIMA, myTS)
checkresiduals(fc.ARIMA)

fc.SES <- ses(ts1, h=241)
autoplot(fc.SES)
accuracy(fc.SES, myTS)
checkresiduals(fc.SES)

fc.naive <- naive(ts1, h=241)
autoplot(fc.naive)
accuracy(fc.naive, myTS)
checkresiduals(fc.naive)

fc.holt <- holt(ts1, h=241)
autoplot(fc.holt)
accuracy(fc.holt, myTS)
checkresiduals(fc.holt)

fc.HW <- forecast(HoltWinters(ts1, beta = FALSE, gamma = FALSE), h=241, fan=TRUE)
autoplot(fc.HW)
accuracy(fc.HW, myTS)
checkresiduals(fc.HW)

#df.data_ph <- as.data.frame(t(dt.data_all[i,-1]))
df.train <- as.data.frame(t(dt.train[i,-1]))
setnames(df.train, "y")
df.train$ds <- colnames(dt.train[1,-1])

m.prophet <- prophet(df.train,yearly.seasonality = TRUE)
future       <- make_future_dataframe(m.prophet, periods = 241)
fc.prophet     <- predict(m.prophet, future)
plot(m.prophet,fc.prophet)

prophet_plot_components(m.prophet, fc.prophet)

# Transform


plot(t(dt.dataNA_top10[1,]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation")
plot(t(dt.dataNA_top10[2,]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation")
plot(t(dt.dataNA_top10[3,]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation")
plot(t(dt.dataNA_top10[4,]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation")
plot(t(dt.dataNA_top10[5,]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation")
plot(t(dt.dataNA_top10[6,]), na.rm=TRUE, ylab = "Page visits", xlab = "Daily observation")

# Try transform data

# Replace NAs with zero, leave outliers as is

#Testing with one row first

impute.zero <- function(x){
  x <- as.double(unlist(x))
  nas <- is.na(x)
  x[nas] <- 0
  as.list(x)
}

dt.data_all <- dt.data[,][, impute.zero(.SD), by=Page]
setnames(dt.data_all, names(dt.data))

# Pick 10 pages with the highest number of page visits for data exploration
dt.data_tmp <- data.frame(dt.data$Page)
dt.data_tmp$sum = rowSums(dt.data[,2:804])
data_tmp <- arrange(dt.data_tmp, desc(sum))[1:10,1]

dt.data_top10 <- dt.data[dt.data$Page %in% data_tmp,]

# Check stationarity
# The Augmented Dickey-Fuller (ADF) t-statistic test
#   small p-values suggest the data is stationary and doesn't need to be transformed.
adf.test(ts(t(dt.data_top10[1,-1]),frequency = 365.25, start = c(2015, 7, 1)))
# Decomposition

# Split 70% for training and 30% for test
# Training: 0.7 * 803 = 562 ==> column c(2:563) ==> From '2015-07-01' until '2017-01-13'
# Test    : 803 - 562 = 241 ==> column c(564:803) ==> From '2017-01-13' until '2017-09-10'

dt.train <- dt.data_all[,c(1:563)]
dt.test <- dt.data_all[,c(564:803)]

# Create TS object

i <- 2

ts1 <- ts(t(dt.train[i,-1]),frequency = 365.25, start = c(2015, 7, 1))
myTS <- ts(t(dt.data_all[i,-1]),frequency = 365.25, start = c(2015, 7, 1))

# Plotting
autoplot(ts1)


fc.ARIMA <- forecast(auto.arima(ts1),h=241)
autoplot(fc.ARIMA)
accuracy(fc.ARIMA, myTS)
checkresiduals(fc.ARIMA)

fc.SES <- ses(ts1, h=241)
autoplot(fc.SES)
accuracy(fc.SES, myTS)
checkresiduals(fc.SES)

fc.naive <- naive(ts1, h=241)
autoplot(fc.naive)
accuracy(fc.naive, myTS)
checkresiduals(fc.naive)

fc.holt <- holt(ts1, h=241)
autoplot(fc.holt)
accuracy(fc.holt, myTS)
checkresiduals(fc.holt)

fc.HW <- forecast(HoltWinters(ts1, beta = FALSE, gamma = FALSE), h=241, fan=TRUE)
autoplot(fc.HW)
accuracy(fc.HW, myTS)
checkresiduals(fc.HW)

#df.data_ph <- as.data.frame(t(dt.data_all[i,-1]))
df.train <- as.data.frame(t(dt.train[i,-1]))
setnames(df.train, "y")
df.train$ds <- colnames(dt.train[1,-1])

m.prophet <- prophet(df.train,yearly.seasonality = TRUE)
future       <- make_future_dataframe(m.prophet, periods = 241)
fc.prophet     <- predict(m.prophet, future)
plot(m.prophet,fc.prophet)

prophet_plot_components(m.prophet, fc.prophet)

#accuracy(fc.prophet, myTS)
#heckresiduals(fc.prophet)


#ts.top10 <- ts.top10[,][, impute.zero(.SD), by=Page]


class(ts.top10)


#ts.dates <- colnames(tsdata[1,2:804])



ts.top10[1,-1]

# Extract metadata from Page column. Possibly useful for data exploration.

str(tsdata)
head(tsdata)


tsdata <- tsdata[1:100,]
write.csv(tsdata, file="xx.csv", row.names = FALSE)

tsdata$article    <- gsub("^(.*)_.*.org.*","\\1",tsdata$Page)
tsdata$locale <- gsub(".*_(..).*.org_.*","\\1",tsdata$Page)
tsdata$project <- gsub(".*_...(.*.org)_.*","\\1",tsdata$Page)
tsdata$access  <- gsub(".*.org_(.*)_.*$","\\1",tsdata$Page)
tsdata$agent   <- gsub(".*_(.*)$","\\1",tsdata$Page)

# Reorder columns for convenience
setcolorder(tsdata, c("Page", "article", "locale", "project", "access", "agent", colnames(tsdata)[!(colnames(tsdata) %in% c("Page", "article", "locale", "project", "access", "agent"))])) 

str(tsdata)

tsdata[,c("Page","article","locale","project")]

write.csv(tsdata,"xx1.csv",row.names = FALSE)

# Create TS object

# Testing
inds <- seq(as.Date("2015-07-01"), as.Date("2017-09-10"), by = "day")

newdata <- t(tsdata[,7:809])
str(newdata)
head(newdata)

myts <- ts(newdata,
           start = c(2015, as.numeric(format(inds[1], "%j"))),
           frequency = 365.25)

myts
myts2 <- ts(newdata, start = decimal_date(as.Date("2015-07-01")), frequency = 365)

myts2

plot(myts[,5])

#Testing

tsdata <- tsdata[1:100,]

tmpts <- cbind(tsdata[,1],data.frame(Article="",Locale="",Site="",Access="",Agent=""))
class(tmpts)

write.csv(tmpts, file="tmpts.csv", row.names = FALSE)

tmpts

class(tsdata[1,1:4])
typeof(unlist(tsdata[1,1:4]))

f_ExtractPage <- function(X) {

#X <- unlist(X)
Idx1 <- nchar(X[,1])
PgIdx <- rev(unlist(gregexpr(pattern="_",X[,1])))
vAgent <- substr(X[,1],PgIdx[1]+1,Idx1)
vAccess <- substr(X[,1],PgIdx[2]+1,PgIdx[1]-1)
vLocale.Site <- substr(X[,1],PgIdx[3]+1,PgIdx[2]-1)
ArtIdx <- unlist(gregexpr(pattern="\\.",vLocale.Site))
vLocale <- substr(vLocale.Site,1,ArtIdx[1]-1)
vSite <- substr(vLocale.Site,ArtIdx[1]+1,nchar(vLocale.Site))
vArticle <- substr(X[,1],1,PgIdx[3]-1)

return(list(vArticle,vLocale,vSite,vAccess,vAgent))

}

#2

x1 <- tmpts[,1]
x2 <- tmpts$Page

class(x1)
class(x2)

typeof(x1)
typeof(x2)



f_ExtractPage(x2)


tmpts[,c("Article","Locale","Site","Access","Agent")] <- f_ExtractPage(tmpts[,])

# doesn't work -
for( i in 1:nrow(tmpts) ){
  vList <- f_ExtractPage(tmpts[i,])
  tmpts[i,"Article"] <- vList$Article
  tmpts[i,"Locale"] <- vList$Locale
  tmpts[i,"Site"] <- vList$Site
  tmpts[i,"Access"] <- vList$Access
  tmpts[i,"Agent"] <- vList$Agent
}

for( i in 1:2 ){
  vList <- f_ExtractPage(tmpts[i,])
  #print(vList)
}

f_ExtractPage(tmpts[1,])

# doesn't work - by(tmpts, 1:nrow(tmpts), f_ExtractPage(row))

apply(tmpts, 1, f_ExtractPage)

f_ExtractPage(tmpts[1,])

# Test
fTest <- function(x) {
  vPg <- x[1]
  vArt <- as.character(x[2])
  print(vPg)
}

apply(tmpts, 1, fTest)
# Test

apply(tmpts,2,f_ExtractPage(X))

rm(tmpts)
tmpts <- cbind(tsdata[,1],data.frame(article,locale,site,access,agent))
               
f_test2 <- function(X) {
  print(X[[1]])
}

dfTst1 <- tsdata[,c(1:4)][,f_test2(.SD),by=row.names(tsdata)]

f_test <- function(X) {
  
  Idx1 <- nchar(X)
  PgIdx <- rev(unlist(gregexpr(pattern="_",X)))
  vAgent <- substr(X,PgIdx[1]+1,Idx1)
  vAccess <- substr(X,PgIdx[2]+1,PgIdx[1]-1)
  vLocale.Article <- substr(X,PgIdx[3]+1,PgIdx[2]-1)
  ArtIdx <- unlist(gregexpr(pattern="\\.",vLocale.Article))
  vLocale <- substr(vLocale.Article,1,ArtIdx[1]-1)
  vArticle <- substr(vLocale.Article,ArtIdx[1]+1,nchar(vLocale.Article))
  
  X <- list(vAgent,vAccess,vLocale,vArticle)
  
  return(vPageList)
  
}


unlist(train_2[,1])
#1#SIMPLEST MODEL

# extract a lists  of pages and dates
train.date.cols = names(train_2[,-1])


par(mfrow = c(2,1))
plot(cmort)
plot(diff(cmort))


df <- data.frame(act = c(0,0,20, 50,NA,NA,10), 
                 pred= c(0,1,20,100,50,NA,NA))

#-----------------------------------------------------

f.fcCombined1 <- function(fc1, fc2, fc3){ 
  fcc1 <- (fc1+fc2+fc3)/3
  return (fcc1) 
}


f.fcCombined2 <- function(fc1, fc2, fc3, v1, v2, v3){ 
  fcc2 <- ((v2+v3)/(v1+v2+v3))*fc1+((v1+v3)/(v1+v2+v3))*fc2+((v1+v2)/(v1+v2+v3))*fc3
  return (fcc2) 
}


f.MAE <- function(act, pred){ 
  err <- abs(act - pred)
  err <- ifelse(is.na(act), NA, err)                    # omit if act is NA
  err <- ifelse(is.na(pred) & !is.na(act), NA, err)     # max error if pred is NA and act is available
  MAE <- mean(err)
  return (MAE) 
}

f.var_err <- function(act, pred){ 
  sqerr <- abs(act - pred)^2
  sqerr <- ifelse(is.na(act), NA, sqerr)                    # omit if act is NA
  sqerr <- ifelse(is.na(pred) & !is.na(act), NA, sqerr)     # max error if pred is NA and act is available
  v.sqerr <- mean(sqerr)
  return (v.sqerr) 
}

f.residual <- function(act, pred){ 
  res <- act - pred
  res <- ifelse(is.na(act), NA, sqerr)                    # omit if act is NA
  res <- ifelse(is.na(pred) & !is.na(act), NA, sqerr)     # max error if pred is NA and act is available
  return (res) 
}
# f = vector with forecasts, y = vector with actuals
f.MASE <- function(f,y) {
  if(length(f)!=length(y)){ stop("Vector length is not equal") }
  n <- length(f)
  return(mean(abs((y - f) / ((1/(n-1)) * sum(abs(y[2:n]-y[1:n-1]))))))
}


df$err <- f.err(df$act,df$pred)
err.var <- var(df$err, na.rm=TRUE)


f.smape <- function(act, pred){ 
  sm <- 200 * abs(act - pred) / (abs(act) + abs(pred)) # normal formula
  sm <- ifelse(is.na(act), NA, sm)                     # omit if act is NA
  sm <- ifelse(is.na(pred) & !is.na(act), 200, sm)     # max error if pred is NA and act is available
  sm <- ifelse(pred==0 & act==0, 0, sm)                # perfect (arbitrary 0/0=0)
  smape <- mean(sm, na.rm = TRUE)
  return (smape) 
}  

smape <- f.smape(df$act,df$pred)
#df$sm <- f.smape(df$act,df$pred)
#smape <- mean(df$sm, na.rm=TRUE)

mase <- function(act, pred){ 
  mse <- 200 * abs(act - pred) / (abs(act) + abs(pred)) # normal formula
  ms <- ifelse(is.na(act), NA, sm)                     # omit if act is NA
  sm <- ifelse(is.na(pred) & !is.na(act), 200, sm)     # max error if pred is NA and act is available
  sm <- ifelse(pred==0 & act==0, 0, sm)                # perfect (arbitrary 0/0=0)
  return (mse) 
}


options(warn = oldW)

