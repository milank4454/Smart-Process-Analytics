import pandas as pd
import json
import matplotlib.pyplot as plt
#import time

# get json file from the replaceevents.json file as replace_json

def main():
    instance = input("Provide the instance:")
    data = get_train_data()
    #initial_time = time.time()
    #print(initial_time)
    calculate_prediction(instance, data)
    #final_time = time.time()
    #print(final_time-initial_time)
	#prediction = predict_single_value(instance,len(instance),data)
    #print(prediction)
    #print(data)

def calculate_prediction(instance, data):
    # need to create a function that will generate a matrix
    try:
        
        n = len(instance)
        while n != 0:
            instance = instance[len(instance)-n:len(instance)]
            prediction = predict_single_value(instance, n, data)
            print("n-gram value=",n)
            print(prediction.iloc[1: , :])
            visualise_probablities(prediction,n)
            prediction.head()
            n = n - 1
        # Create an array with the predictions received 
        # from the above function
        print("Prediction calculated:")
    except:
        print("There is no predicted data relates to mined data")
        

def get_train_data():
    with open("replaceevents.json","r") as file:
        replace_json = json.load(file) 
        df = pd.read_csv(r"spadata.csv")
        df =  df.sort_values(by="Complete Timestamp")
        df1 = df["Activity"].replace(replace_json)
        df["sequance"] = df1
        df = df[['Case ID', 'sequance']]
        df1 = df.groupby('Case ID')['sequance'].apply(''.join).reset_index()
        del df1['Case ID']
        stringdata = df1["sequance"].tolist()
    return stringdata

def predict_single_value(instance, m, data):
    #instantiate empty lists
    data_sorted = []
    cropped_list = []
    #Looping over the data with given instance to get the prediction matrix
    for i in range(len(data)):
        index = data[i].find(instance)
        if index != -1:
          data_sorted.append(data[i])
          if index + 2 <= len(data[i]) - 1:
          #print(data[i][index + len(instance)])
            cropped_list.append(data[i][index + len(instance)])
    df = pd.DataFrame(cropped_list, columns = ["letters"])
    df['letter_probability'] = df.groupby('letters')['letters'].transform(lambda x : x.count()/len(df))
    prediction = df.drop_duplicates() 
    prediction = prediction.T
    #prediction = df1.rename(columns=df1.iloc[0]).drop(df1.index[0])
    prediction.columns = prediction.iloc[0]
    return prediction

def visualise_probablities(prediction,n):
    trans_df = prediction.T
    trans_df.plot(x="letters", y="letter_probability",kind="bar", title= "n-gram value= %i" %n);
    plt.show()
main()

