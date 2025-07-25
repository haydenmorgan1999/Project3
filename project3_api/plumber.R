#load libraries
library("plumber")
library("dplyr")
library("tidyverse")
library("tidymodels")
library("caret")
library("yardstick")
library("ranger")

#read in data 
diabetes <- read.csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")

diabetes$response <- factor(diabetes$Diabetes_binary, levels = c(0, 1), labels = c("No Diabetes", "Has Prediabetes or Diabetes")) #response variable is Diabetes_binary

diabetes$HighBP <- factor(diabetes$HighBP, levels = c(0, 1), labels = c("No High BP", "High BP")) 

diabetes$HighChol <- factor(diabetes$HighChol, levels = c(0, 1), labels = c("No High Chol", "High Chol"))

diabetes$CholCheck <- factor(diabetes$CholCheck, levels = c(0, 1), labels = c("Does Not Check Cholesterol", "Does Check Cholesterol")) 

diabetes$Smoker <- factor(diabetes$Smoker, levels = c(0, 1), labels = c("Not Smoker", "Smoker"))

diabetes$Stroke <- factor(diabetes$Stroke, levels = c(0, 1), labels = c("No Stroke", "Has Had Stroke")) 

diabetes$HeartDiseaseorAttack <- factor(diabetes$HeartDiseaseorAttack, levels = c(0, 1), labels = c("No Heart Disease", "Has Had Heart Disease"))

diabetes$PhysActivity <- factor(diabetes$PhysActivity, levels = c(0, 1), labels = c("Does Not Exercise", "Exercises")) 

diabetes$Fruits <- factor(diabetes$Fruits, levels = c(0, 1), labels = c("Does Not Eat Fruit", "Eats Fruit")) 

diabetes$Veggies <- factor(diabetes$Veggies, levels = c(0, 1), labels = c("Does Not Eat Veggies", "Eats Veggies")) 

diabetes$HvyAlcoholConsump <- factor(diabetes$HvyAlcoholConsump, levels = c(0, 1), labels = c("Not A Heavy Drinker", "Heavy Drinker")) 

diabetes$AnyHealthcare <- factor(diabetes$AnyHealthcare, levels = c(0, 1), labels = c("No Healthcare", "Has Healthcare")) 

diabetes$NoDocbcCost <- factor(diabetes$NoDocbcCost, levels = c(0, 1), labels = c("No Cost Barrier In Seeing Doctor", "Cost Barrier In Seeing Doctor")) 

diabetes$GenHlth <- factor(diabetes$GenHlth, levels = c(1:5), labels = c("Excellent Health", "Very Good Health", "Good Health", "Fair Health", "Poor Health")) 

diabetes$DiffWalk <- factor(diabetes$DiffWalk, levels = c(0, 1), labels = c("No Difficulty Walking", "Has Difficulty Walking")) 

diabetes$Sex <- factor(diabetes$Sex, levels = c(0, 1), labels = c("Female", "Male"))

diabetes$Age <- factor(diabetes$Age, levels = c(1:13), labels = c("18 - 24", "25 - 29", "30 - 34", "35 - 39", "40 - 44", "45 - 49", "50 - 54", "55 - 59", "60 - 64", "65 - 69", "70 - 74", "75 - 79", "80+")) 

diabetes$Education <- factor(diabetes$Education, levels = c(1:6), labels = c("No School/K Only", "Elementary School", "Middle School", "High School", "College", "Graduate Education")) 

diabetes$Income <- factor(diabetes$Income, levels = c(1:8), labels = c("Less than $10,000", "$10,000 - less than $15,000", "$15,000 - less than $20,000", "$20,000 - less than $25,000", "$25,000 - less than $35,000", "$35,000 - less than $50,000", "$50,000 - less than $75,000", "More than $75,000"))

#fit the best model; code repeated from the Modeling file 
set.seed(100) #chose 100 because it is a similar value to our HW5 assignment 

split <- initial_split(diabetes, prop = 0.7) #project instructions specify 70/30 split 
train <- training(split)
test <- testing(split)

folds <- vfold_cv(train, 5)

rtree_phys <- recipe(response ~ PhysActivity + BMI, data = train) |>
  step_dummy(PhysActivity) |>
  step_normalize(BMI)

rtree_spec <- rand_forest(mtry = tune()) |>
  set_engine("ranger") |>
  set_mode("classification")

rtreephys_wfl <- workflow() |>
  add_recipe(rtree_phys) |>
  add_model(rtree_spec)

metrics <- metric_set(mn_log_loss)

rtreephys_fit <- rtreephys_wfl |>
  tune_grid(resamples = folds, metrics = metrics)

best_rphys <- select_best(rtreephys_fit, metric = "mn_log_loss")

rphys_final_wfl <- rtreephys_wfl |>
  finalize_workflow(best_rphys)

rphys_final_fit <- rphys_final_wfl |>
  last_fit(split, metrics = metrics)

#* @apiTitle Project 3 API

mean(diabetes$BMI) #for fxn below (default)
diabetes |>
  count(PhysActivity) #also for fxn below (default)

#Endpoint 1
#* Take in any predictors used in the 'best' model. Include default values for each that is the mean of the variable's values (numeric) or the most prevalent class (categorical).
#* @param PhysActivity 1st predictor
#* @param BMI 2nd predictor
#* @get /pred
function(PhysActivity = "Exercises", BMI = 28.4){
  pred_data <- data.frame(
    PhysActivity = factor(PhysActivity, levels = c("Does Not Exercise", "Exercises")),
    BMI = as.numeric(BMI)
  )
  
  fitted_wfl <- extract_workflow(rphys_final_fit)
  
  prediction <- predict(fitted_wfl, pred_data, type = "prob")
  
  return(prediction)

}

#Three example function calls to copy/paste
# 1 (default): http://127.0.0.1:17124/pred
# 2: http://127.0.0.1:17124/pred?PhysActivity=Exercises&BMI=20
# 3: http://127.0.0.1:17124/pred?PhysActivity=Does%20Not%20Exercise&BMI=35

#* No inputs. Output is a message.
#* @get /info
function(){
  list(
    msg = "Name: Hayden Morgan | GitHub URL: https://haydenmorgan1999.github.io/Project3/"
  ) 
}