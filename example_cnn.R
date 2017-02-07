# 輸入層
data <- mx.symbol.Variable('data')

# 第一卷積層，windows的視窗大小是 5x5, filter=20
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20, name="1-conv")
# 第一卷積層的激發函數：tanh
conv.act1 <- mx.symbol.Activation(data=conv1, act_type="tanh", name="1-conv.act")
# 第一卷積層後的池化層，max，大小縮為 2x2
pool1 <- mx.symbol.Pooling(data=conv.act1, pool_type="max", name="1-conv.pool",
                           kernel=c(2,2), stride=c(2,2))

# 第二卷積層，windows的視窗大小是 5x5, filter=50
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50, name="2-conv")
# 第二卷積層的激發函數：tanh
conv.act2 <- mx.symbol.Activation(data=conv2, act_type="tanh", name="2-conv.act")
# 第二卷積層後的池化層，max，大小縮為 2x2
pool2 <- mx.symbol.Pooling(data=conv.act2, pool_type="max", name="2-conv.pool",
                           kernel=c(2,2), stride=c(2,2))

# Flatten
flatten <- mx.symbol.Flatten(data=pool2)
# 建立一個Full-Connected的隱藏層，500節點
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500, name="1-fc")
# 隱藏層的激發函式：tanh
fc.act1 <- mx.symbol.Activation(data=fc1, act_type="tanh", name="1-fc.act")

# 輸出層：因為預測數字為0~9共十個，節點為10
output <- mx.symbol.FullyConnected(data=fc.act1, num_hidden=10, name="output")
# Loss Function: Softmax 
cnn <- mx.symbol.SoftmaxOutput(data=output, name="cnn")

# 神經網路中的各個參數的資訊
arguments(cnn)

# 視覺化CNN結構
graph.viz(cnn$as.json(),
          graph.title= "CNN for Kaggle－Digit Recognizer"
)




# Transform matrix format for LeNet 
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))

mx.set.seed(0)

# 訓練剛剛設計的CNN模型
cnn.model <- mx.model.FeedForward.create(
  cnn,       # 剛剛設計的CNN模型
  X=train.array,  # train.x
  y=train.y,  #  train.y
  ctx=mx.cpu(),  # 可以決定使用cpu或gpu
  num.round=10,  # iteration round
  array.batch.size=100, # batch size
  learning.rate=0.07,   # learn rate
  momentum=0.7,         # momentum  
  eval.metric=mx.metric.accuracy, # 評估預測結果的基準函式*
  initializer=mx.init.uniform(0.05), # 初始化參數
  epoch.end.callback=mx.callback.log.train.metric(100)
)


# test prediction 
test.y <- predict(cnn.model, test.array)
test.y <- t(test.y)
test.y.label <- max.col(test.y) - 1

# Submission format for Kaggle
result <- data.frame(ImageId = 1:length(test.y.label),
                     label = test.y.label)

