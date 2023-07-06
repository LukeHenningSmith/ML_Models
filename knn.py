def classify_nn(training_filename, testing_filename, k):
  results = []
  training_data = []
  testing_data = []
  
  #first, load in both files into nested arrays
  for line in open(training_filename):
    training_data.append(line.strip("\n").split(","))
  
  for line in open(testing_filename):
    testing_data.append(line.strip("\n").split(","))
  
  #now we have to data, time to go through each test example and classify it
  #all normalised numeric attributes, so simply do Euclidean distance to all, 
  #then average of k nearest
  
  for e in testing_data:
    #find ordered distances to all test data, then pick 5 smallest
    distances = []    #distances are tuples [distance,outcome] where outcome is yes or no
    for t in training_data:
      #calculate euclidean distance between e and t
      #don't square root, since distance comparison is the same
      cur_dist = 0
      for index in range(len(e)):
        #remember, t is 1 longer because yes/no
        #don't square root, since square distance comparison is the same
        cur_dist += ((float(e[index])-float(t[index]))*(float(e[index])-float(t[index])))
      #now use that dist as the answer
      distances.append([cur_dist,t[len(e)]]) #[distance,outcome]
    #now order the distances
    sorted_distances = sorted(distances, key=lambda x: x[0])
    
    #now take the k smallest, and find most common (yes or no)
    yes_count=0
    no_count=0
    for x in range(k):
      if(sorted_distances[x][1] == "yes"):
        yes_count+=1
      else:
        no_count+=1
    
    #now that is the result
    if(yes_count<no_count):
      results.append("no")
    else:
      results.append("yes")
    
  return results