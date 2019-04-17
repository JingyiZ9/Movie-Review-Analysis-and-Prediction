
  if (!require("pacman")) 
    install.packages("pacman")
  
  pacman::p_load(
    "text2vec",
    "slam",
    "glmnet",
    "e1071",
    "xgboost"
  )  
  

set.seed(1)
################ read data ##################
#Clean html tags, and split data into training and test
movie_data=read.table("Project2_data.tsv", stringsAsFactors = F, header= T)
movie_data$review = gsub('<.*?>', ' ', movie_data$review)
splits = read.table("Project2_splits.csv", header = T)
train = movie_data[-which(movie_data$new_id%in%splits[,s]),]
test = movie_data[which(movie_data$new_id%in%splits[,s]),]

#use R package 'text2vec' to built vocabulary and construct DT matrix (maximum 4-grams). 
#library(text2vec)
prep_fun = tolower
tok_fun = word_tokenizer #totoken
it_train = itoken(train$review,
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun)
it_test = itoken(test$review,
                 preprocessor = prep_fun, 
                 tokenizer = tok_fun)
stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")
vocab = create_vocabulary(it_train,ngram = c(1L,4L), stopwords = stop_words)#9293186
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)#28628
bigram_vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train = create_dtm(it_train, bigram_vectorizer)#xtrain
dtm_test = create_dtm(it_test, bigram_vectorizer)#xtest

################# screening method#########################
#use commands from the R package 'slam' to efficiently compute the mean and var for each column of dtm_train. 
#library(slam)
v.size = dim(dtm_train)[2]
ytrain = train$sentiment

summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), mean)
summ[,2] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), var)
summ[,3] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), mean)
summ[,4] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), var)

n1=sum(ytrain); 
n=length(ytrain)
n0= n - n1

myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0) #t-statistics

#order the words by the magnitude of their t-statistics, pick the top 2000 words,
#which are then divide into two lists: positive words and negative words. 
words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:2000]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]

#submit file
write(words[id], file="myvocab.txt")
#
myvocab = scan(file = "myvocab.txt", what = character())

all = read.table("Project2_data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("Project2_splits.csv", header = T)

train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
vocab = create_vocabulary(it_train,ngram = c(1L,4L))
vocab = vocab[vocab$term %in% myvocab, ] ##
bigram_vectorizer = vocab_vectorizer(vocab)
dtm_train_new = create_dtm(it_train, bigram_vectorizer)
dtm_test_new = create_dtm(it_test, bigram_vectorizer)

#######################################################################################
## model one : lasso regression
#Then tried lasso regression on the selected 2000 words and both could achieve AUC above 0.96 on the test data. 
#library(glmnet)
ytrain = train$sentiment
ytest = test$sentiment
mycv = cv.glmnet(x=dtm_train, ytrain, 
                 family='binomial',type.measure = "auc", 
                 nfolds = 10, alpha=1)
myfit = glmnet(x=dtm_train, y=ytrain, 
               lambda = mycv$lambda.min, family='binomial', alpha=1)
logit_pred = predict(myfit, dtm_test, type = "response")
glmnet:::auc(ytest, logit_pred)
submission1=cbind(test$new_id,logit_pred)
write.table(submission1, 'mysubmission1.txt', row.names = FALSE, 
            col.names = c("new_id", "prob"), sep = ", ")


##########################################################################################
##model two: Naive Bayes
#library("e1071")
##whole data
ytrain = train$sentiment
ytest = test$sentiment
train_data=data.frame(as.matrix(dtm_train_new))
test_data=data.frame(as.matrix(dtm_test_new))
NB_model = naiveBayes(train_data,as.factor(ytrain ))
pred_NB = predict(NB_model,test_data)
pred_NB = as.numeric(pred_NB)-1
auc(ytest, pred_NB)
submission2=cbind(test$new_id,as.numeric(pred_NB))
write.table(submission2, 'mysubmission2.txt', row.names = FALSE, 
            col.names = c("new_id", "prob"), sep = ", ")

#########################################################################################
##Model three: Tree Model
#library(xgboost)
all = read.table("Project2_data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("Project2_splits.csv", header = T)

train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")
vocab = create_vocabulary(it_train,ngram = c(1L,4L), stopwords = stop_words)#9293186
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
bigram_vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train = create_dtm(it_train, bigram_vectorizer)#xtrain
dtm_test = create_dtm(it_test, bigram_vectorizer)#xtest

train.y = train$sentiment
test.y = test$sentiment
param = list(max_depth = 2, 
             subsample = 0.5, 
             objective='binary:logistic')
ntrees = 500
set.seed(500)
bst = xgb.train(params = param, 
                data = xgb.DMatrix(data = dtm_train, label = train.y),
                nrounds = ntrees, 
                nthread = 2)

dt = xgb.model.dt.tree(model = bst)
words = unique(dt$Feature[dt$Feature != "Leaf"])
length(words) 

new_feature_train = xgb.create.features(model = bst, dtm_train)
new_feature_train = new_feature_train[, - c(1:ncol(dtm_train))]
new_feature_test = xgb.create.features(model = bst, dtm_test)
new_feature_test = new_feature_test[, - c(1:ncol(dtm_test))]
c(ncol(new_feature_test), ncol(new_feature_train))
                          
#logistic regression on the lasso
ytrain = train$sentiment
ytest = test$sentiment
mycv3 = cv.glmnet(x=new_feature_train, y=ytrain, 
                  family='binomial',type.measure = "auc", 
                  nfolds = 10, alpha=1)
myfit3 = glmnet(x=new_feature_train, y=ytrain, 
                lambda = mycv3$lambda.min, family='binomial', alpha=1)
logit_pred3 = predict(myfit3, new_feature_test, type = "response")
glmnet:::auc(ytest, logit_pred3)

submission3=cbind(test$new_id,logit_pred3)
write.table(submission3, 'mysubmission3.txt', row.names = FALSE, 
            col.names = c("new_id", "prob"), sep = ", ")
