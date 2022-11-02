import sys, csv, random, math

#Function that casts input appropriately
def cast(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        try:
            return float(x)
        except (ValueError, TypeError):
            return x

#Calculates net and out for a given set of data and weights
#Inputs are:
#   data - an instance of normalized data from the set
#   weights - the list of weights for a given layer
#   e - the mathematical constant
def calculate_out(data, weights, e):
    net = weights[0]
    for j in range(1, len(data)):
        net += data[j] * weights[j]
    return 1/(1+(e ** -net))

#Read arguments
file_path = sys.argv[1]
hidden_layer_size = int(sys.argv[2])
learning_rate = float(sys.argv[3])
training_percentage = float(sys.argv[4])
seed = int(sys.argv[5])
prediction_threshold = float(sys.argv[6])

#Read input .csv into data
with open(file_path) as f:
    reader = csv.reader(f)
    data = list(reader)

#Remove labels and cast the attributes to correct type
data.pop(0)
for row in data:
    for i in range(len(row)):
        row[i] = cast(row[i])

#Standardize data:
standardized_data = list()
for row in data:
    standardized_data.append([row[0]])

for j in range(1, len(data[0])):
    #If type is string, attribute is nominal, so do One Hot Coding
    if type(data[0][j]) is str:

        #Make a list of each value in the attribute
        values = list()
        for i in range(len(data)):
            if data[i][j] not in values:
                values.append(data[i][j])
    
        #Iterate through data, and append the appropriate 0's and 1's to standardized data
        for i in range(len(data)):
            one_hot = [0] * (len(values) - 1)
            value_index = values.index(data[i][j])
            if value_index != len(one_hot):
                one_hot[value_index] = 1
            for x in one_hot:
                standardized_data[i].append(x)

    #If type is numerical, standardize data to between 0 and 1
    else:

        #Find min and max value
        temp_label_list = list()
        for i in range(len(data)):
            temp_label_list.append(data[i][j])

        maxi = max(temp_label_list)
        mini = min(temp_label_list)

        #To avoid dividing by 0, plus I'm lazy
        if(maxi == 0 and mini == 0):
            maxi = 1

        #Standardize data
        for i in range(len(data)):
            standardized_data[i].append((data[i][j] - mini)/(maxi - mini))

#Shuffle data
random.seed(seed)
random.shuffle(standardized_data)

#Split data into training set and test set according to input percentage
training_set = list()
validation_set = list()
test_set = list()
training_pivot = int(len(standardized_data) * training_percentage)
validation_pivot = int(len(standardized_data) * (training_percentage + (1 - training_percentage)/2))

for i in range(training_pivot):
    training_set.append(standardized_data[i])
for i in range(training_pivot, validation_pivot):
    validation_set.append(standardized_data[i])
for i in range(validation_pivot, len(standardized_data)):
    test_set.append(standardized_data[i])

                    #Training Phase

#Initialize weights as lists of random numbers between -0.1 and 0.1
#hidden_weights is a 2d array, with a list of weights for each hidden node
hidden_weights = [[random.random() * 0.2 - 0.1 for _ in range(len(training_set[0]))] for __ in range(hidden_layer_size)]
output_weights = [random.random() * 0.2 - 0.1 for _ in range(hidden_layer_size + 1)]
n = 0
accuracy = 0
e = math.e

#Do the algorithm until accuracy is greater than 99% or until 500 epochs
while n < 500 and accuracy < 0.99:
    for i in training_set:
        #Calculate out for each layer
        hidden_out = [1]
        for j in hidden_weights:
            hidden_out.append(calculate_out(i, j, e))
        output_out = calculate_out(hidden_out, output_weights, e)

        #Calculate error and feedback
        error = i[0] - output_out
        delta_o = output_out * (1 - output_out) * error

        #Calculate feedback for hidden layer and update weights
        for j in range(hidden_layer_size):
            activation = hidden_out[j+1] * (1 - hidden_out[j+1])
            delta_j = activation * output_weights[j+1] * delta_o

            #Update weights for hidden layer
            delta_weight = -1 * delta_j
            hidden_weights[j][0] -= learning_rate * delta_weight
            for k in range(1,len(i)):
                delta_weight = -i[k] * delta_j
                hidden_weights[j][k] -= learning_rate * delta_weight

        #Update weights for output layer
        for j in range(len(output_weights)):
            delta_weight = -hidden_out[j] * delta_o
            output_weights[j] -= learning_rate * delta_weight

    #Check weights on validation set
    total = 0
    correct = 0
    for i in validation_set:
        hidden_out = [1]
        for j in hidden_weights:
            hidden_out.append(calculate_out(i, j, e))
        output_out = calculate_out(hidden_out, output_weights, e)
        prediction = 1 if output_out > 0.5 else 0

        total += 1
        if prediction == i[0]:
            correct += 1

    accuracy = correct/total
    n += 1


#Set up confusion matrix
labels = list()
for i in data:
    if i[0] not in labels:
        labels.append(i[0])

confusion_matrix = [[]]
for i in labels:
    confusion_matrix[0].append(i)
    temp_list = list()
    for y in labels:
        temp_list.append(0)
    temp_list.append(i)
    confusion_matrix.append(temp_list)

#Make predictions on test set, and write confusion matrix
for i in test_set:
    #Calculate out for each layer
    hidden_out = [1]
    for j in hidden_weights:
        hidden_out.append(calculate_out(i, j, e))
    output_out = calculate_out(hidden_out, output_weights, e)

    prediction = 1 if output_out > prediction_threshold else 0

    confusion_matrix[labels.index(prediction) + 1][labels.index(i[0])] += 1

#Write confusion matrix to csv
file_name = "results_" +  file_path[:len(file_path) - 4] + '_' + str(hidden_layer_size) + "n_" + str(learning_rate) + "r_" + str(prediction_threshold) + "t_" + str(training_percentage) + "p_" + str(seed) + ".csv"
with open(file_name, 'w') as f:
    write = csv.writer(f)
    write.writerows(confusion_matrix)