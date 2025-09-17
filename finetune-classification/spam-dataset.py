import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data( url, zip_path, extracted_path, data_file_path):

    if data_file_path.exists():

        print(f"{data_file_path} already exists. Skipping download "
              "and extraction."
        )

        return

    # download the file
    with urllib.request.urlopen(url) as response:
      
        # unzip the file
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # add the .TSV extention
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

## play with the data a bit
## the .TSV extension file is saved as a tab (\t) separated text file

## load the file data into a pandas dataframe (df) object

import pandas as pd
df = pd.read_csv( data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df)

# now check the balance of the labels : how many hams (good) vs spams (bad)
# (or, we are checking the class label distribution)

print(df["Label"].value_counts())

# we see ham = 4825, spam = 747. This is imbalanced. Would like to have spam = ham.
# for simplicity, we undersample, and so basically take a set of 747 data points from each 
# of the 2 classes (see logs.txt), ham and spam, thusly:

def create_balanced_dataset(df):

    # get the number of spams
    num_spam = df[df["Label"] == "spam"].shape[0]

    # sample from the ham a number of "num_spam" messages (random)
    # and create a ham_subset that is unsampled but equal to the spam dataset
    # quantity-wise

    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )

    # create a new pandas dataframe object that is balanced w.r.t the 2 classes
    # basically to the ham_subset of 747, add an additional spam class data (which
    # actually would be the fill spam class dataset itself. Thus I have both ham
    # and spam datapoints now of the same quantity, inside a balanced_df dataframes obj.

    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])

    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

# next, we will convert the label names to "tokens" (0 and 1)
# (or integer class labels)

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
print(balanced_df)

# next, we create a random_split function to split the dataset (balanced_df) 
# into 3 parts: 70% for training, 10% for validation, and 20% for testing
# (this is a common ML ratio to train, adjust and evaluate models)

# the good thing about this func is that it will auto adjust the 3 subsets
# based on the frac values it recieves

def random_split(df, train_frac, validation_frac):

    # shuffle the entire data frame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True) 

    # calculate the split indices
    train_end      = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df      = df[:train_end]                    # grab the first chunk for training (dataframes)
    validation_df = df[train_end:validation_end] # grab the 2nd chunk for validation
    test_df       = df[validation_end:]

    return train_df, validation_df, test_df

# test the above :

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1) # meaning test is 0.2

# save the batched sets as their own .CSV files:

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)




    
