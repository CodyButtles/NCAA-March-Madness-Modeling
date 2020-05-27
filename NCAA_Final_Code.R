rm(list=ls())
#load libraries
library(data.table)
library(mlr)
library(tidyverse)
library(xgboost)
library(mlrMBO)
library(DiceKriging)
library(mlr3)
library(caret)
library(readxl)

###### STEP 1 ###########
#Load in data
NCAADir<-"C:\\Users\\buttl\\OneDrive - UWSP\\Projects\\NCAAMarchMadness\\"
NCAAData<-read_csv(paste(NCAADir,"Model_Diff_Data.csv",sep = ""))

data_Keep<-NCAAData[-c(1,2,5,6,8)]


#load data
train<-sample(1:nrow(data_Keep),floor(2*nrow(data_Keep)/3))
trainData<-other_Round[train,]
testData<-other_Round[-train,]

#convert data frame to data table
setDT(trainData) 
setDT(testData)

#Storing target field
labels <- trainData$team_1_winner
ts_label <- testData$team_1_winner

#convert factor to numeric 
labels <- as.numeric(labels)
ts_label <- as.numeric(ts_label)

new_tr <- model.matrix(~.+0,data = trainData[,-c("team_1_winner"),with=F]) 
new_ts <- model.matrix(~.+0,data = testData[,-c("team_1_winner"),with=F])

#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)


#################################
#Grid and random search
#################################
#convert characters to factors
fact_col <- colnames(train)[sapply(trainData,is.character)]

for(i in fact_col) set(trainData,j=i,value = factor(trainData[[i]]))
for (i in fact_col) set(testData,j=i,value = factor(testData[[i]]))

#create tasks
trainData$team_1_winner<-factor(trainData$team_1_winner)
testData$team_1_winner<-factor(testData$team_1_winner)

trainData$team_1_winner<-as.character(trainData$team_1_winner)

trainData<-data.frame(trainData)
testData<-data.frame(testData)

traintask <- makeClassifTask(data = trainData,target = 'team_1_winner')
testtask <- makeClassifTask (data = testData,target = "team_1_winner")

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "prob",eval_metric='auc')

#set parameter space
params <- makeParamSet( #makeDiscreteParam("booster",values = c("gbtree","gblinear")),
  makeIntegerParam("max_depth",lower = 3L,upper = 10L),
  makeNumericParam("min_child_weight",lower = 1L,upper = 75L),
  makeNumericParam("subsample",lower = 0.3,upper = 1),
  makeIntegerParam("nrounds", lower = 75, upper = 2500),
  makeNumericParam("eta", lower = 0.001, upper = 0.3),
  makeNumericParam("colsample_bytree",lower = 0.25,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=3L)

############## MBO version
mbo.ctrl = makeMBOControl(save.on.disk.at = 0L)

mbo.ctrl = setMBOControlTermination(mbo.ctrl, iters = 75)
surrogate.lrn = makeLearner("regr.km", predict.type = "se")
ctrl<- makeTuneControlMBO(learner = surrogate.lrn, mbo.control = mbo.ctrl)
###################

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

for (i in 1:10) {
  
  
  
  #parameter tuning
  mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                       measures = auc, par.set = params, control = ctrl, show.info = T)
  mytune$y
  #acc.test.mean 
  #0.7379835
  dfparam<-data.frame(mytune$x)
  dfparam$train_Accuracy<-mytune$y
  #mytune$x
  #set hyperparameters
  lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)
  
  #train model
  x <- subset(traintask$env$data, select=-team_1_winner)
  y <- traintask$env$data$team_1_winner
  
  xgmodel <- train(learner = lrn_tune,task = traintask,x=x,y=y)
  
  #predict model
  xgpred <- predict(xgmodel,testtask$env$data)
  
  confMat<-confusionMatrix(xgpred,testtask$env$data$team_1_winner)
  testAcc<-confMat$overall['Accuracy']
  dfparam$test_Accuracy<-testAcc
  if(i==2){
    newdf<-rbind(dfparam,dfparamstore)
  }
  else if (i>2) {
    newdf<-rbind(newdf,dfparam)
  }
  dfparamstore<-dfparam
  #Accuracy : 0.7041
}


write_csv(newdf,'Your file path to save the ten results from above')

######### Step 2 ################

##################################
#Test on 2019 Data
##################################
train<-NCAAData[which(NCAAData$matchup_year!=2019),]
test_2019<-NCAAData[which(NCAAData$matchup_year==2019),]

train<-train[-c(1,2,5,6,8)]
test_2019<-test_2019[-c(1,2,5,6,8)]

#convert data frame to data table
setDT(train) 
setDT(test_2019)

#Storing target field
labels <- train$team_1_winner
ts_label <- test_2019$team_1_winner

#convert factor to numeric 
labels <- as.numeric(labels)
ts_label <- as.numeric(ts_label)

new_tr <- model.matrix(~.+0,data = train[,-c("team_1_winner"),with=F]) 
new_ts <- model.matrix(~.+0,data = test_2019[,-c("team_1_winner"),with=F])

#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.001053702, gamma=0,
               max_depth=3, min_child_weight=18.98075554, subsample=0.998440441, colsample_bytree=0.432143225)

#train on 500 rounds to find best nround
xgb1 <- xgb.train (params=params,data = dtrain, nrounds = 500,nfold=10, 
                   watchlist = list(val=dtest,train=dtrain), print_every_n = 10,
                   early_stop_round = 10, maximize = F , eval_metric = "error",set.seed(1234567))

##Look at results to find best nrounds
xgb1$evaluation_log$error_diff<-xgb1$evaluation_log$val_error-xgb1$evaluation_log$train_error
xgb1$evaluation_log[which(xgb1$evaluation_log$val_error==min(xgb1$evaluation_log$val_error)),]

#Best nround 22
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 22,nfold=10, 
                   watchlist = list(val=dtest,train=dtrain), print_every_n = 10,
                   early_stop_round = 10, maximize = F , eval_metric = "error",set.seed(1234567))

##best iteration = 72
xgb1$evaluation_log$error_diff<-xgb1$evaluation_log$val_error-xgb1$evaluation_log$train_error
xgb1$evaluation_log[which(xgb1$evaluation_log$val_error==min(xgb1$evaluation_log$val_error)),]

#xgb1$evaluation_log$auc_diff<-xgb1$evaluation_log$val_auc-xgb1$evaluation_log$train_auc
#xgb1$evaluation_log[which(xgb1$evaluation_log$auc_diff==min(xgb1$evaluation_log$auc_diff)),]


#Graph nrounds and 


#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#confusion matrix
confusionMatrix (factor(xgbpred), factor(ts_label))
#Accuracy - 74.6%
#nrounds 22
#seed 1234567
#nfold 10

############## Step 3 ##################

################################################
#Predict theoretical matchups after first round
################################################
NCAADir<-"C:\\Users\\Cody\\OneDrive - UWSP\\Projects\\NCAAMarchMadness\\"
NCAAData_2019<-read_excel(paste(NCAADir,"NCAABasketballStatistics.xlsx",sep = ""),sheet = "2019")

NCAAData_2019<-NCAAData_2019[-c(24,25)]

for (i in range(21,22,24,27,28,29,30)) {
  NCAAData_2019[,i] <- as.numeric(unlist(NCAAData_2019[,i]))
}
NCAAData_2019$Turnovers<-as.numeric(NCAAData_2019$Turnovers)
NCAAData_2019$`ESPN Strength of Schedule`<-as.numeric(NCAAData_2019$`ESPN Strength of Schedule`)
NCAAData_2019$`Total Points`<-as.numeric(gsub(",","",NCAAData_2019$`Total Points`))
NCAAData_2019$`Average PPG`<-as.numeric(NCAAData_2019$`Average PPG`)
NCAAData_2019$`Total Opp Points`<-as.numeric(gsub(",","",NCAAData_2019$`Total Opp Points`))



#################Create function to get predictions for hypothetical model created matchups#################
Model_Diff<- function(team_1_name,team_2_name){
  
  team1<-NCAAData_2019[which(NCAAData_2019$Team==team_1_name),]
  team2<-NCAAData_2019[which(NCAAData_2019$Team==team_2_name),]
  Team<-team1$Team
  team1_Diff<-data.frame(Team)
  Team<-team2$Team
  team2_Diff<-data.frame(Team)
  
  
  team1_Diff$team_1_conference_tourney_champs<-team1$`Conference Tournament Champion`
  team1_Diff$team_1_made_tourney_previous_year<-team1$`Made Tournament Previous Year`
  team1_Diff$team_1_winner<-ifelse(team1$`Number of Tournament Wins`>team2$`Number of Tournament Wins`,1,0)
  team1_Diff$pointers3_attempted_diff<-team1$`3-Pointers Attempted`-team2$`3-Pointers Attempted`
  team1_Diff$pointers3_made_diff<-team1$`3-Pointers Made`-team2$`3-Pointers Made`
  team1_Diff$assists_diff<-team1$Assists - team2$Assists
  team1_Diff$assists_to_turnovers_ratio_diff<-team1$`Assist to Turnover Ratio`- team2$`Assist to Turnover Ratio`
  team1_Diff$average_opp_ppg_diff<-team1$`Average Opp PPG`- team2$`Average Opp PPG`
  team1_Diff$average_ppg_diff<-team1$`Average PPG`- team2$`Average PPG`
  team1_Diff$espn_strength_of_schedule_diff<-team1$`ESPN Strength of Schedule`- team2$`ESPN Strength of Schedule`
  team1_Diff$free_throw_percentage_diff<-(team1$`Free Throw Percentage`- team2$`Free Throw Percentage`)*100
  team1_Diff$free_throws_attempted_diff<-team1$`Free Throws Attempted`- team2$`Free Throws Attempted`
  team1_Diff$free_throws_made_diff<-team1$`Free Throws Made`- team2$`Free Throws Made`
  team1_Diff$game_count_diff<-team1$`Game Count`- team2$`Game Count`
  team1_Diff$losses_diff<-team1$Losses- team2$Losses
  team1_Diff$losses_against_top_25_rpi_teams_diff<-team1$`Losses Against Top 25 RPI Teams`- team2$`Losses Against Top 25 RPI Teams`
  team1_Diff$offensive_rebounds_diff<-team1$`Offensive Rebounds`- team2$`Offensive Rebounds`
  team1_Diff$opponent_rebounds_diff<-team1$`Opponent's Rebounds`- team2$`Opponent's Rebounds`
  team1_Diff$rebound_differential_diff<-team1$`Rebound Differential`- team2$`Rebound Differential`
  team1_Diff$rebounds_diff<-team1$Rebounds- team2$Rebounds
  team1_Diff$scoring_differential_per_game_diff<-team1$`Scoring Differential Per Game`- team2$`Scoring Differential Per Game`
  team1_Diff$seed_diff<-team1$Seed- team2$Seed
  team1_Diff$three_point_percentage_diff<-(team1$`3-Point Percentage`- team2$`3-Point Percentage`)*100
  team1_Diff$total_opp_points_diff<-team1$`Total Opp Points`- team2$`Total Opp Points`
  team1_Diff$total_points_diff<-team1$`Total Points`- team2$`Total Points`
  team1_Diff$total_scoring_differential_diff<-team1$`Total Scoring Differential`- team2$`Total Scoring Differential`
  team1_Diff$turnovers_diff<-team1$Turnovers- team2$Turnovers
  team1_Diff$wins_diff<-team1$Wins- team2$Wins
  team1_Diff$wins_against_top_25_rpi_teams_diff<-team1$`Wins Against Top 25 RPI Teams`- team2$`Wins Against Top 25 RPI Teams`
  
  
  team2_Diff$team_1_conference_tourney_champs<-team2$`Conference Tournament Champion`
  team2_Diff$team_1_made_tourney_previous_year<-team2$`Made Tournament Previous Year`
  team2_Diff$team_1_winner<-ifelse(team2$`Number of Tournament Wins`>team1$`Number of Tournament Wins`,1,0)
  team2_Diff$pointers3_attempted_diff<-team2$`3-Pointers Attempted`-team1$`3-Pointers Attempted`
  team2_Diff$pointers3_made_diff<-team2$`3-Pointers Made`-team1$`3-Pointers Made`
  team2_Diff$assists_diff<-team2$Assists-team1$Assists
  team2_Diff$assists_to_turnovers_ratio_diff<-team2$`Assist to Turnover Ratio`- team1$`Assist to Turnover Ratio`
  team2_Diff$average_opp_ppg_diff<-team2$`Average Opp PPG`- team1$`Average Opp PPG`
  team2_Diff$average_ppg_diff<-team2$`Average PPG`- team1$`Average PPG`
  team2_Diff$espn_strength_of_schedule_diff<-team2$`ESPN Strength of Schedule`- team1$`ESPN Strength of Schedule`
  team2_Diff$free_throw_percentage_diff<-(team2$`Free Throw Percentage`- team1$`Free Throw Percentage`)*100
  team2_Diff$free_throws_attempted_diff<-team2$`Free Throws Attempted`- team1$`Free Throws Attempted`
  team2_Diff$free_throws_made_diff<-team2$`Free Throws Made`- team1$`Free Throws Made`
  team2_Diff$game_count_diff<-team2$`Game Count`- team1$`Game Count`
  team2_Diff$losses_diff<-team2$Losses- team1$Losses
  team2_Diff$losses_against_top_25_rpi_teams_diff<-team2$`Losses Against Top 25 RPI Teams`- team1$`Losses Against Top 25 RPI Teams`
  team2_Diff$offensive_rebounds_diff<-team2$`Offensive Rebounds`- team1$`Offensive Rebounds`
  team2_Diff$opponent_rebounds_diff<-team2$`Opponent's Rebounds`- team1$`Opponent's Rebounds`
  team2_Diff$rebound_differential_diff<-team2$`Rebound Differential`- team1$`Rebound Differential`
  team2_Diff$rebounds_diff<-team2$Rebounds- team1$Rebounds
  team2_Diff$scoring_differential_per_game_diff<-team2$`Scoring Differential Per Game`- team1$`Scoring Differential Per Game`
  team2_Diff$seed_diff<-team2$Seed- team1$Seed
  team2_Diff$three_point_percentage_diff<-(team2$`3-Point Percentage`- team1$`3-Point Percentage`)*100
  team2_Diff$total_opp_points_diff<-team2$`Total Opp Points`- team1$`Total Opp Points`
  team2_Diff$total_points_diff<-team2$`Total Points`- team1$`Total Points`
  team2_Diff$total_scoring_differential_diff<-team2$`Total Scoring Differential`- team1$`Total Scoring Differential`
  team2_Diff$turnovers_diff<-team2$Turnovers- team1$Turnovers
  team2_Diff$wins_diff<-team2$Wins- team1$Wins
  team2_Diff$wins_against_top_25_rpi_teams_diff<-team2$`Wins Against Top 25 RPI Teams`- team1$`Wins Against Top 25 RPI Teams`
  
  team_Diff<-rbind(team1_Diff, team2_Diff)
  
  
  testDF<-team_Diff[-c(1)]
  
  #convert data frame to data table
  setDT(testDF)
  
  #Storing target field
  testDF_label <- testDF$team_1_winner
  
  #convert factor to numeric 
  testDF_label <- as.numeric(testDF_label)
  
  new_testDF <- model.matrix(~.+0,data = testDF[,-c("team_1_winner"),with=F])
  
  #preparing matrix 
  dtestDF <- xgb.DMatrix(data = new_testDF,label=testDF_label)
  
  #model prediction
  xgbpred_test <- predict (xgb1,dtestDF)
  pred_results<-team_Diff[c(1,4)]
  pred_results$pred<-xgbpred_test
  
  return(pred_results)
}


#############Generate hypothetical matchups created from modeling#################
# Round 2: Belmont vs LSU, Kansas St vs Wisconsin, Utah St vs North Carolina, Iowa St vs Houston
bel_lsu<-Model_Diff("Belmont","LSU")
kan_wi<-Model_Diff("Kansas St.","Wisconsin")
ut_nc<-Model_Diff("Utah St.","North Carolina")
io_hou<-Model_Diff("Iowa St.","Houston")
round2_model<-rbind(bel_lsu,kan_wi,ut_nc,io_hou)

#Round 3: Wofford vs Houston, Wisconsin vs Virginia, Belmont vs Michigan St
woff_hou<- Model_Diff("Wofford","Houston")
wis_vir<-Model_Diff("Wisconsin","Virginia")
bel_mich<-Model_Diff("Belmont","Michigan St.")
round3_model<-rbind(woff_hou,wis_vir,bel_mich)

#Round 4: Florida St vs Michigan, North Carolina vs Houston
flo_mich<-Model_Diff("Florida St.","Michigan")
nc_hou<-Model_Diff("North Carolina","Houston")
round4_model<-rbind(flo_mich,nc_hou)

#Round 5: Michigan st vs Michigan, Virginia vs Houston
michst_mich<-Model_Diff("Michigan St.","Michigan")
vir_hou<-Model_Diff("Virginia","Houston")
round5_model<-rbind(michst_mich,vir_hou)

#Round 6:Virginia vs Michigan St
vir_mich<-Model_Diff("Virginia","Michigan St.")

round2_default<-results[which(results$tourney_round==4),]
round2_default<-round2_default[which(round2_default$seed_diff>0),]


vir_duke<-Model_Diff("Virginia","Duke")
mich_gon<-Model_Diff("Michigan St.","Gonzaga")
vir_gon<-Model_Diff("Virginia","Gonzaga")



#Graph errors and feature importance
xgb1 <- xgb.train (params=params,data = dtrain, nrounds = 100,nfold=10, 
                   watchlist = list(val=dtest,train=dtrain), print_every_n = 10,
                   early_stop_round = 10, maximize = F , eval_metric = "error",set.seed(1234567))

plot(xgb1$evaluation_log$iter,xgb1$evaluation_log$train_error,type="l",col="red",
     ylim=range( c(xgb1$evaluation_log$train_error,xgb1$evaluation_log$val_error) ),
     ylab="Error",xlab="Iterations")
lines(xgb1$evaluation_log$iter,xgb1$evaluation_log$val_error,col="green")
abline(h=min(xgb1$evaluation_log$val_error),col="black")
legend(75, 0.32, legend=c("Train_Error", "Test_Error"),
       col=c("red", "green"), lty=1)

plot(xgb1$evaluation_log$iter,xgb1$evaluation_log$error_diff,type="l",col="blue",xlab="Iterations",
     ylab="Error Difference from Test-Train")
abline(h=0,col="black")
legend(75, -0.01, legend=c("Error_Diff"),
       col=c("blue", "green"), lty=1)
#view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20])


#Create Datafram with actual wins and predicted
results<-NCAAData[which(NCAAData$matchup_year==2019),]

results<-results[c(1,7,8,27)]
results$predicted_win<-xgbpred
results

confusionMatrix (factor(results$team_1_winner), factor(results$predicted_win))

#Look at upsets
upset_df<-results[which(results$seed_diff>0),]
confusionMatrix (factor(upset_df$team_1_winner), factor(upset_df$predicted_win))


#Round 1 Accuracy
round_1<-results[which(results$tourney_round==1),]
confusionMatrix (factor(round_1$team_1_winner), factor(round_1$predicted_win))





#Looking at Average upsets from 2013-2018
avgUpset<-NCAAData[which(NCAAData$matchup_year!=2019),]
avgUpset<-avgUpset[which(avgUpset$tourney_round==6),]
avgUpset<-avgUpset[which(avgUpset$seed_diff>0),]
avgUpset<-avgUpset[which(avgUpset$team_1_winner==1),]
#51/6=8.5 upsets on avg for round 1
#27/6=4.5 upsets on avg for round 2
#14/6=2.33 upsets on avg for round 3
#13/6=2.16 upsets on avg for round 4
#2/6=0.3333 upsets on avg for round 5
#1/6=0.16666 upsets on avg for round 6
