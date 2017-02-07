# http://rpubs.com/skydome20/R-Note12-DigitRecognizer-Kaggle
# https://www.kaggle.com/c/digit-recognizer

# 42000個觀測值，每一筆代表一張手寫數字的圖片
# 784個自變數： 28 x 28 pixels，以灰階值0~255表示。
# 應變數label，代表這張圖片象徵的數字，
# 也是在測試資料(test.csv)中要預測的值。
train <- read.csv('data/train.csv')
dim(train)

# 28000個觀測值，每一筆代表一張手寫數字的圖片
# 784個自變數： 28 x 28 pixels，以灰階值0~255表示。
# 無label，要預測的
test <- read.csv('data/test.csv')
dim(test)

# 資料轉換成 28x28 的矩陣
obs.matrix <- matrix(unlist(train[1, -1]), # ignore 'label'
                     nrow = 28,            
                     byrow=T)
str(obs.matrix)

# 用 image 畫圖，顏色指定為灰階值 0~255
image(obs.matrix, 
      col=grey.colors(255))

# 由於原本的圖是倒的，因此寫一個翻轉的函式：
# rotates the matrix
rotate <- function(matrix){
  t(apply(matrix, 2, rev)) 
} 

# 畫出第1筆~第8筆資料的圖
par(mfrow=c(2,4))
for(x in 1:8){
  obs.matrix <- matrix(unlist(train[x, -1]), # ignore 'label'
                       nrow = 28,            
                       byrow=T)
  
  image(rotate(obs.matrix),
        col=grey.colors(255),
        xlab=paste("Label", train[x, 1], sep=": "),
        cex.lab = 1.5
  )
}

# data preparation
train.x <- t(train[, -1]/255) # train: 28 x 28 pixels
train.y <- train[, 1]         # train: label
test.x <- t(test/255)         # test: 28 x 28 pixels


# download mxnet package from github 
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
require(mxnet)


# 輸入層
data <- mx.symbol.Variable("data")

# 第一隱藏層: 500節點，狀態是Full-Connected
fc1 <- mx.symbol.FullyConnected(data, name="1-fc", num_hidden=500)
# 第一隱藏層的激發函數: Relu
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
# 這裡引入dropout的概念
drop1 <- mx.symbol.Dropout(data=act1, p=0.5)

# 第二隱藏層: 400節點，狀態是Full-Connected
fc2 <- mx.symbol.FullyConnected(drop1, name="2-fc", num_hidden=400)
# 第二隱藏層的激發函數: Relu
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
# 這裡引入dropout的概念
drop2 <- mx.symbol.Dropout(data=act2, p=0.5)

# 輸出層：因為預測數字為0~9共十個，節點為10
output <- mx.symbol.FullyConnected(drop2, name="output", num_hidden=10)
# Loss Function: Softmax
dnn <- mx.symbol.SoftmaxOutput(output, name="dnn")



# 神經網路中的各個參數的資訊
arguments(dnn)

# 視覺化DNN結構
graph.viz(dnn$`as.json`(),
          graph.title= "DNN for Kaggle－Digit Recognizer"
)




mx.set.seed(0) 

# 訓練剛剛創造/設計的模型
dnn.model <- mx.model.FeedForward.create(
  dnn,       # 剛剛設計的DNN模型
  X=train.x,  # train.x
  y=train.y,  #  train.y
  ctx=mx.cpu(),  # 可以決定使用cpu或gpu
  num.round=10,  # iteration round
  array.batch.size=100, # batch size
  learning.rate=0.07,   # learn rate
  momentum=0.9,         # momentum  
  eval.metric=mx.metric.accuracy, # 評估預測結果的基準函式*
  initializer=mx.init.uniform(0.07), # 初始化參數
  epoch.end.callback=mx.callback.log.train.metric(100)
)

# # define my own evaluate function (R-squared)
# my.eval.metric <- mx.metric.custom(
#   name = "R-squared", 
#   function(real, pred) {
#     mean_of_obs <- mean(real)
#     
#     SS_tot <- sum((real - mean_of_obs)^2)
#     SS_reg <- sum((predict - mean_of_obs)^2)
#     SS_res <- sum((real - predict)^2)
#     
#     R_squared <- 1 - (SS_res/SS_tot)
#     R_squared
#   }
# )


# test prediction 
test.y <- predict(dnn.model, test.x)
test.y <- t(test.y)
test.y.label <- max.col(test.y) - 1

# Submission format for Kaggle
result <- data.frame(ImageId = 1:length(test.y.label),
                     label = test.y.label)





