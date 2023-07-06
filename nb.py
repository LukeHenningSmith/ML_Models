# Implementation of the Naive Bayes Algorithm for Binary Classification
# By Luke Henning-Smith

import math

def classify_nb(training_filename, testing_filename):
  results = []
  training_data = []
  testing_data = []
 
  #first, load in both files into nested arrays
  for line in open(training_filename):
    training_data.append(line.strip("\n").split(","))
  
  for line in open(testing_filename):
    testing_data.append(line.strip("\n").split(","))
  
  #Uses Bayes theorem to predict yes or no class:
  #make two arrays, with same index as attributes in training_data and test_data
  #[P(E1|yes), P(E2|yes), ...]
  #[P(E1|no), P(E2|no), ...]
  #correction: for each element in the above arrays, 
  #need to do a mean and standard deviation
  vals_yes = []
  vals_no = []  
  
  #first, find P(yes) and P(no)
  yes_count=0
  no_count=0
  index_of_class = len(training_data[0])-1
  for i in training_data:
    if(i[index_of_class] == "yes"):
      yes_count+=1
    else:
      no_count+=1
  
  p_yes = yes_count / (yes_count+no_count)
  p_no = no_count / (yes_count+no_count)
  

  #next, find out mean and standard deviation for each attribute
  for col_index in range(len(training_data[0])-1):
    #find mean and std dev for each column
    
    #Mean of class YES   
    count=0
    summation = 0
    for example in training_data:
      if(example[len(training_data[0])-1] == "yes"):
        summation += float(example[col_index])
        count+=1
    yes_mean = float(summation / count)
    
    #Standard Deviation of class YES 
    yes_sqr_dif_sum = 0
    yes_denom = count-1
    for example in training_data:
      if(example[len(training_data[0])-1] == "yes"):
        yes_sqr_dif_sum += float((float(example[col_index]) - float(yes_mean)) * (float(example[col_index]) - float(yes_mean)))
    
    yes_std_dev = float(math.sqrt(yes_sqr_dif_sum/yes_denom))
    
    #Mean of class NO 
    count=0
    summation = 0
    for example in training_data:
      if(example[len(training_data[0])-1] == "no"):
        summation += float(example[col_index])
        count+=1
    no_mean = float(summation / count)
    
    #Standard Deviation of class NO 
    no_sqr_dif_sum = 0
    no_denom = count-1
    for example in training_data:
      if(example[len(training_data[0])-1] == "no"):
        no_sqr_dif_sum += float((float(example[col_index]) - float(no_mean)) * (float(example[col_index]) - float(no_mean)))
    
    no_std_dev = float(math.sqrt(no_sqr_dif_sum/no_denom))
  
    #now we have the values, put them in the arrays
    vals_yes.append([yes_mean,yes_std_dev])
    vals_no.append([no_mean,no_std_dev])
  
  #now we have all the mean/std deviation we need, can calculate probability for each training example
  #using the Bayes theorem numerator:
  for e in testing_data:
    yes_numerator = p_yes
    no_numerator = p_no
    #now *= all of the attributes
    for index in range(len(e)):
      attr_mean = vals_yes[index][0]
      attr_std_dev = vals_yes[index][1]
      yes_numerator *= float((1/(attr_std_dev * math.sqrt(2*math.pi))) * math.pow(math.e, -( ((float(e[index])-attr_mean)*(float(e[index])-attr_mean)) / (2*attr_std_dev*attr_std_dev)) ))
      
      attr_mean = vals_no[index][0]
      attr_std_dev = vals_no[index][1]
      no_numerator *= float((1/(attr_std_dev * math.sqrt(2*math.pi))) * math.pow(math.e, -( ((float(e[index])-attr_mean)*(float(e[index])-attr_mean)) / (2*attr_std_dev*attr_std_dev)) ))

    #now compare them, and classify
    if(yes_numerator < no_numerator):
      results.append("no")
    else:
      results.append("yes")

  return results