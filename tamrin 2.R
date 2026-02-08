############################################
# 0) Load data
############################################
df <- read.csv("Food_Delivery_Route_Efficiency_Dataset_HW2_without test.csv")


############################################
# 1) Data cleaning / recoding
############################################

# 1-1) Merge "Bike" and "Bicycle" into a single level "Bike"
df$delivery_mode[df$delivery_mode %in% c("Bike", "Bicycle")] <- "Bike"

# 1-2) Convert categorical variables to factor
df$traffic_level    <- as.factor(df$traffic_level)
df$delivery_mode    <- as.factor(df$delivery_mode)
df$weather          <- as.factor(df$weather)
df$restaurant_zone  <- as.factor(df$restaurant_zone)
df$customer_zone    <- as.factor(df$customer_zone)

# 1-3) Create different versions of hour (before split)
df$hour_num <- as.numeric(as.character(df$hour))  # numeric encoding
df$hour_fac <- as.factor(df$hour)                 # factor encoding


############################################
# 2) Train/Test split
############################################
set.seed(123)

n <- nrow(df)
train_index <- sample(1:n, size = 0.8 * n)

train <- df[train_index, ]
test  <- df[-train_index, ]

# SST for test-set R^2 (using train mean as baseline)
SST <- sum((test$delivery_time_min - mean(train$delivery_time_min))^2)


############################################
# 3) Compare hour encoding: numeric vs factor
############################################

# 3-1) Hour as numeric
m_num <- lm(
  delivery_time_min ~ route_length_km + distance_km + hour_num,
  data = train
)
pred_num <- predict(m_num, newdata = test)
R2_num <- 1 - sum((test$delivery_time_min - pred_num)^2) / SST

# 3-2) Hour as factor
m_fac <- lm(
  delivery_time_min ~ route_length_km + distance_km + hour_fac,
  data = train
)
pred_fac <- predict(m_fac, newdata = test)
R2_fac <- 1 - sum((test$delivery_time_min - pred_fac)^2) / SST

# Summaries + test R^2
summary(m_num)
summary(m_fac)
R2_num
R2_fac


############################################
# 4) EDA (plots)
############################################

# Distribution of delivery time
hist(
  df$delivery_time_min,
  breaks = 20,
  main = "Distribution of Delivery Time",
  xlab  = "Delivery Time (min)"
)

# Distance vs delivery time
plot(
  df$distance_km, df$delivery_time_min,
  main = "Distance vs Delivery Time",
  xlab = "Distance (km)",
  ylab = "Delivery Time (min)"
)

# Route length vs delivery time
plot(
  df$route_length_km, df$delivery_time_min,
  main = "Route Length vs Delivery Time",
  xlab = "Route Length (km)",
  ylab = "Delivery Time (min)"
)

# Delivery time by traffic level
boxplot(
  delivery_time_min ~ traffic_level,
  data = df,
  main = "Delivery Time by Traffic Level",
  xlab = "Traffic Level",
  ylab = "Delivery Time (min)"
)

# Delivery time by delivery mode
boxplot(
  delivery_time_min ~ delivery_mode,
  data = df,
  main = "Delivery Time by Delivery Mode",
  xlab = "Delivery Mode",
  ylab = "Delivery Time (min)"
)

# Delivery time by hour
boxplot(
  delivery_time_min ~ hour,
  data = df,
  main = "Delivery Time by Hour",
  xlab = "Hour of Day",
  ylab = "Delivery Time (min)"
)

# Average delivery time by hour (needs dplyr)
library(dplyr)

avg_by_hour <- df %>%
  group_by(hour) %>%
  summarise(mean_time = mean(delivery_time_min))

plot(
  avg_by_hour$hour, avg_by_hour$mean_time,
  type = "b",
  main = "Average Delivery Time by Hour",
  xlab = "Hour",
  ylab = "Avg Delivery Time (min)"
)


############################################
# 5) Correlation analysis (numeric variables)
############################################
num_vars <- df[, c("delivery_time_min", "distance_km", "route_length_km", "hour_num")]

cor_matrix <- cor(num_vars)
cor_matrix

# Heatmap of correlation matrix
heatmap(
  cor_matrix,
  Rowv = NA, Colv = NA,
  main = "Correlation Heatmap (Numeric Variables)",
  col  = heat.colors(256),
  scale = "none"
)

# Correlation plot (corrplot)
install.packages("corrplot")
library(corrplot)

corrplot(
  cor_matrix,
  method = "color",
  addCoef.col = "black",
  tl.cex = 0.9,
  number.cex = 0.7
)


############################################
# 6) Full model + Backward elimination
############################################

# 6-1) Full model
full_model <- lm(
  delivery_time_min ~ route_length_km + traffic_level +
    delivery_mode + weather + restaurant_zone + customer_zone + hour_num,
  data = train
)
summary(full_model)

# 6-2) Backward elimination
step_model <- step(full_model, direction = "backward")
summary(step_model)

# 6-3) Test R^2 comparison (full vs step)
pred_full <- predict(full_model, newdata = test)
pred_new  <- predict(step_model, newdata = test)

SSE_full <- sum((test$delivery_time_min - pred_full)^2)
SSE_new  <- sum((test$delivery_time_min - pred_new)^2)

SST <- sum((test$delivery_time_min - mean(train$delivery_time_min))^2)

R2_full <- 1 - (SSE_full / SST)
R2_new  <- 1 - (SSE_new  / SST)

R2_full
R2_new


############################################
# 7) Interaction models (m1 ... m11)
############################################

# m1: route_length_km * traffic_level
m1 <- lm(delivery_time_min ~ route_length_km * traffic_level, data = train)
summary(m1)
pred1 <- predict(m1, newdata = test)
SSE1 <- sum((test$delivery_time_min - pred1)^2)
R2_1 <- 1 - SSE1 / SST
R2_1

# m2: route_length_km * delivery_mode
m2 <- lm(delivery_time_min ~ route_length_km * delivery_mode, data = train)
summary(m2)
pred2 <- predict(m2, newdata = test)
SSE2 <- sum((test$delivery_time_min - pred2)^2)
R2_2 <- 1 - SSE2 / SST
R2_2

# m3: route_length_km + distance_km * weather
m3 <- lm(delivery_time_min ~ route_length_km + distance_km * weather, data = train)
summary(m3)
pred3 <- predict(m3, newdata = test)
SSE3 <- sum((test$delivery_time_min - pred3)^2)
R2_3 <- 1 - SSE3 / SST
R2_3

# m4: route_length_km + traffic_level * hour_num
m4 <- lm(delivery_time_min ~ route_length_km + traffic_level * hour_num, data = train)
summary(m4)
pred4 <- predict(m4, newdata = test)
SSE4 <- sum((test$delivery_time_min - pred4)^2)
R2_4 <- 1 - SSE4 / SST
R2_4

# m5: route_length_km * weather (kept duplicated exactly as you had it)
m5 <- lm(delivery_time_min ~ route_length_km * weather, data = train)
summary(m5)

m5 <- lm(delivery_time_min ~ route_length_km * weather, data = train)
summary(m5)
pred5 <- predict(m5, newdata = test)
SSE5 <- sum((test$delivery_time_min - pred5)^2)
R2_5 <- 1 - SSE5 / SST
R2_5

# m6: route_length_km * customer_zone
m6 <- lm(delivery_time_min ~ route_length_km * customer_zone, data = train)
summary(m6)
pred6 <- predict(m6, newdata = test)
SSE6 <- sum((test$delivery_time_min - pred6)^2)
R2_6 <- 1 - SSE6 / SST
R2_6

# m7: route_length_km * restaurant_zone
m7 <- lm(delivery_time_min ~ route_length_km * restaurant_zone, data = train)
summary(m7)
pred7 <- predict(m7, newdata = test)
SSE7 <- sum((test$delivery_time_min - pred7)^2)
R2_7 <- 1 - SSE7 / SST
R2_7

# m8: route_length_km + traffic_level * delivery_mode
m8 <- lm(delivery_time_min ~ route_length_km + traffic_level * delivery_mode, data = train)
summary(m8)
pred8 <- predict(m8, newdata = test)
SSE8 <- sum((test$delivery_time_min - pred8)^2)
R2_8 <- 1 - SSE8 / SST
R2_8

# m9: route_length_km + traffic_level * weather
m9 <- lm(delivery_time_min ~ route_length_km + traffic_level * weather, data = train)
summary(m9)
pred9 <- predict(m9, newdata = test)
SSE9 <- sum((test$delivery_time_min - pred9)^2)
R2_9 <- 1 - SSE9 / SST
R2_9

# m10: route_length_km + hour * delivery_mode  (kept as "hour" exactly)
m10 <- lm(delivery_time_min ~ route_length_km + hour * delivery_mode, data = train)
summary(m10)
pred10 <- predict(m10, newdata = test)
SSE10 <- sum((test$delivery_time_min - pred10)^2)
R2_10 <- 1 - SSE10 / SST
R2_10

# m11: route_length_km + hour * weather (kept as "hour" exactly)
m11 <- lm(delivery_time_min ~ route_length_km + hour * weather, data = train)
summary(m11)
pred11 <- predict(m11, newdata = test)
SSE11 <- sum((test$delivery_time_min - pred11)^2)
R2_11 <- 1 - SSE11 / SST
R2_11


############################################
# 8) Additional interaction models (m12, m13, m14)
############################################

# Model 12: distance_km * route_length_km
m12 <- lm(delivery_time_min ~ distance_km * route_length_km, data = train)
summary(m12)

pred12 <- predict(m12, newdata = test)
SSE_12 <- sum((test$delivery_time_min - pred12)^2)

# NOTE: kept your original line exactly (it references SSE_A)
R2_12 <- 1 - SSE_A / SST
R2_12

# Model 14: route_length_km + distance_km * delivery_mode
m14 <- lm(
  delivery_time_min ~ route_length_km + distance_km * delivery_mode,
  data = train
)
summary(m14)

# Model 13: route_length_km + distance_km * traffic_level
m13 <- lm(
  delivery_time_min ~ route_length_km + distance_km * traffic_level,
  data = train
)
summary(m13)

pred13 <- predict(m13, newdata = test)
SSE_13 <- sum((test$delivery_time_min - pred13)^2)
R2_13 <- 1 - SSE_13 / SST
R2_13

pred14 <- predict(m14, newdata = test)
SSE_14 <- sum((test$delivery_time_min - pred14)^2)
R2_14 <- 1 - SSE_14 / SST
R2_14


############################################
# 9) Mixed model (final interaction mix)
############################################
m_mix <- lm(
  delivery_time_min ~ route_length_km +
    distance_km * traffic_level +
    route_length_km * customer_zone,
  data = train
)
summary(m_mix)

pred_mix <- predict(m_mix, newdata = test)
SSE_mix <- sum((test$delivery_time_min - pred_mix)^2)
R2_mix <- 1 - SSE_mix / SST
R2_mix


############################################
# 10) Diagnostic plots
############################################
par(mfrow = c(2, 2))
plot(full_model)

par(mfrow = c(2, 2))
plot(m_mix)
