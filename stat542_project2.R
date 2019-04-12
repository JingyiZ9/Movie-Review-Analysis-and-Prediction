setwd("~/Desktop/STAT542_project2")#delet after complete

################ read data ##################
#Clean html tags, and split data into training and test
movie_data=read.table("Project2_data.tsv", stringsAsFactors = F, header= T)
movie_data$review = gsub('<.*?>', ' ', movie_data$review)
splits = read.table("Project2_splits.csv", header = T)
s = 3 # Here we use the 3rd training/test split. 
train = movie_data[-which(movie_data$new_id%in%splits[,s]),]
test = movie_data[which(movie_data$new_id%in%splits[,s]),]

#use R package 'text2vec' to built vocabulary and construct DT matrix (maximum 4-grams). 
library(text2vec)
prep_fun = tolower
tok_fun = word_tokenizer #字符序列totoken
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


################ model one ##################
#Then tried lasso regression on the selected 2000 words and both could achieve AUC above 0.96 on the test data. 
library(glmnet)
ytrain = train$sentiment
ytest = test$sentiment
mycv = cv.glmnet(x=dtm_train, ytrain, 
                 family='binomial',type.measure = "auc", 
                 nfolds = 10, alpha=1)
myfit = glmnet(x=dtm_train, y=ytrain, 
               lambda = mycv$lambda.min, family='binomial', alpha=1)
logit_pred = predict(myfit, dtm_test, type = "response")
glmnet:::auc(ytest, logit_pred)

#s=1: 0.9614826 
#s=2: 0.9609024  
#s=3: 0.9622723


#################Model 2: screening method #############
#use commands from the R package 'slam' to efficiently compute the mean and var for each column of dtm_train. 
library(slam)
v.size = dim(dtm_train)[2] #vocabulary size 28628
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

myp = (summ[,1] - summ[,3])/ sqrt(summ[,2]/n1 + summ[,4]/n0) #t-statistics

#order the words by the magnitude of their t-statistics, pick the top 2000 words,
#which are then divide into two lists: positive words and negative words. 
words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:4000]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]
summary(pos.list)#712
summary(neg.list)#1288

#submit file
write(words[id], file="myvocab.txt")
#
myvocab = scan(file = "myvocab.txt", what = character())

all = read.table("Project2_data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("Project2_splits.csv", header = T)

s = 2
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
vocab = create_vocabulary(it_train,ngram = c(1L,4L))
vocab = vocab[vocab$term %in% myvocab, ]
bigram_vectorizer = vocab_vectorizer(vocab)
dtm_train_new = create_dtm(it_train, bigram_vectorizer)
dtm_test_new = create_dtm(it_test, bigram_vectorizer)


###################ridge regression on the selected 2000 words
mycv2 = cv.glmnet(x=dtm_train_new, y=ytrain, 
                  family='binomial',type.measure = "auc", 
                  nfolds = 10, alpha=0)
myfit2 = glmnet(x=dtm_train_new, y=ytrain, 
                lambda = mycv2$lambda.min, family='binomial', alpha=0)
logit_pred2 = predict(myfit2, dtm_test_new, type = "response")
glmnet:::auc(ytest, logit_pred2)

#s=1: 0.5012127 
#s=2: 0.5028331 
#s=3: 0.9615834 


#############################################
#Model 3: Use the Tree Model
library(xgboost)
all = read.table("Project2_data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("Project2_splits.csv", header = T)

s = 3
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
                                doc_proportion_min = 0.001)#28628
bigram_vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train = create_dtm(it_train, bigram_vectorizer)#xtrain
dtm_test = create_dtm(it_test, bigram_vectorizer)#xtest

train.y = train$sentiment
test.y = test$sentiment
param = list(max_depth = 2, 
             subsample = 0.5, 
             objective='binary:logistic')
ntrees = 500
set.seed(100)
bst = xgb.train(params = param, 
                data = xgb.DMatrix(data = dtm_train, label = train.y),
                nrounds = ntrees, 
                nthread = 2)

dt = xgb.model.dt.tree(model = bst)
words = unique(dt$Feature[dt$Feature != "Leaf"])
length(words) # use 1084 words

new_feature_train = xgb.create.features(model = bst, dtm_train)
new_feature_train = new_feature_train[, - c(1:ncol(dtm_train))]
new_feature_test = xgb.create.features(model = bst, dtm_test)
new_feature_test = new_feature_test[, - c(1:ncol(dtm_test))]
c(ncol(new_feature_test), ncol(new_feature_train))
                          
###################logistic regression on the lasso
mycv3 = cv.glmnet(x=new_feature_train, y=ytrain, 
                  family='binomial',type.measure = "auc", 
                  nfolds = 10, alpha=1)
myfit3 = glmnet(x=new_feature_train, y=ytrain, 
                lambda = mycv3$lambda.min, family='binomial', alpha=1)
logit_pred3 = predict(myfit3, new_feature_test, type = "response")
glmnet:::auc(ytest, logit_pred3)

#s1: 0.5028621
#s2: 0.5
#s3: 0.950082


