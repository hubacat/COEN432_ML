import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Converting categorical data into numerical format:
# Using One-hot encoding, the nucleotides 'A', 'G', 'C', and 'U' are converted into a binary vector.
# Example:
#     'A' -> [1, 0, 0, 0]
#     'C' -> [0, 1, 0, 0]
#     'G' -> [0, 0, 1, 0]
#     'U' -> [0, 0, 0, 1] 
# Each category gets its own column and only one of these columns is 'hot' (set to 1).
# This increase the dimension immensly, that's why we'll need to remove some nucelotides that dont affect the model's accuracy.
def cat_encoding(ds):
    # ****** USING ONEHOTENCODER: ******
    # # Encoding RNA sequences to binary vectors.
    # encoder = OneHotEncoder(sparse=False)
    # # Convert String sequence to list of nucleotides.
    # sequences = ds['sequence'].apply(list)  
    # # Reshaping sequences into a 2D array for OneHotEncoding
    # reshaped_sequences = sequences.apply(lambda x: [''.join(x)]).values.reshape(-1, 1)
    # # Fit-transform the data and obtain encoded values
    # encoded_sequences = encoder.fit_transform(reshaped_sequences)
    # # Create the dataset for encoded RNA sequences (anothe rone will be made for the BPPM).
    # encoded_sequence_ds = pd.DataFrame(encoded_sequences, columns=encoder.encoder.get_feature_names_out(input_features=['sequence']))

    # # Remove unecessary String characters and transforming String sequence to list array of floats
    # ds['BPPM'] = ds['BPPM'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).str.replace("'", "", regex=True).str.replace(" ", "", regex=True)
    # ds['BPPM'] = ds['BPPM'].apply(lambda x: np.array([float(val) for val in x.split(',')]))

    # # Create the dataset for encoded BPPMS features.
    # bppm_ds = pd.DataFrame(ds['BPPM'].tolist(), columns=[f'BPPM_{i}' for i in range(60 * 60)])

    # # Concatenate and return the encoded sequence DataFrame with the BPPM matrix
    # dataset = pd.concat([encoded_sequence_ds, bppm_ds], axis=1)
    # return dataset

    # ****** USING ONEHOTENCODER V2: ******
    # sequences = ds['sequence'].astype(str)
    # onehot_encoder = OneHotEncoder(sparse_output=False)
    # #integer encode the sequence
    # label_encoder = LabelEncoder()
    # encoded_sequences = []
    # iteration = 0
    # for sequence in sequences:
    #     iteration += 1
    #     print(iteration)
    #     sequence = np.array(list(sequence))
    #     integer_encoded_seq = label_encoder.fit_transform(sequence) 
    #     #reshape because that's what OneHotEncoder likes
    #     integer_encoded_seq = integer_encoded_seq.reshape(-1, 1)
    #     onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
    #     onehot_encoded_seq(np.array(onehot_encoded_seq).T)
        
    # # Create a DataFrame for the encoded sequences
    # encoded_sequences = np.array(encoded_sequences).T
    # encoded_sequences_ds = pd.DataFrame(encoded_sequences)
    # print(encoded_sequences_ds.shape)
    # # Concatenate the encoded sequences DataFrame with BPPM and target variable
    # new_ds = pd.concat([encoded_sequences_ds, ds[['BPPM', 'activity_level']]], axis=1)
    
    # return new_ds

    # # ****** MANUALLY IMPLEMENTING: ******
    # Convert String sequence to list of nucleotides.
    sequences = ds['sequence'].apply(list)  
    nucleotide_mapping = {'A': '1000', 'C': '0100', 'G': '0010', 'U': '0001'}
    
    # Iterate through each nucleotide in the sequence and create binary vectors
    binary_vectors = []
    for sequence in sequences:
        binary_seq = ''
        for nucleotide in sequence:
            binary_seq += nucleotide_mapping.get(nucleotide, '0000')  # Assign '0000' for unknown nucleotides; this should never happen.
        binary_vectors.append(list(binary_seq))
    
    # Create a DataFrame for the binary vectors
    encoded_ds = pd.DataFrame(binary_vectors)
    
    # Reset the index to match the ds DataFrame before concatenating
    encoded_ds.reset_index(drop=True, inplace=True)
    ds.reset_index(drop=True, inplace=True)

    # Formating BPPM feature that is currently in a linearized String matrix in seperated columns with each matrix element sequencially in its own column 
    # Split the linearized matrix string by comma and convert to separate columns
    ds['BPPM'] = ds['BPPM'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).str.replace("'", "", regex=True).str.replace(" ", "", regex=True)
    ds['BPPM'] = ds['BPPM'].apply(lambda x: np.array([float(val) for val in x.split(',')]))
    formated_bppm = pd.DataFrame(ds['BPPM'].tolist())

    # Concatenation of sequence encoding, formatted BPPM, and activity level (target variable).
    ds = pd.concat([encoded_ds, formated_bppm, ds['activity_level']], axis=1)

    # Naming the columns
    encoded_columns = [f'N_{i/4}' for i in range(len(encoded_ds.columns))]
    bppm_columns = [f'BPPM_{i}' for i in range(len(formated_bppm.columns))]
    ds.columns = encoded_columns + bppm_columns + ['activity_level']

    return ds

def main(n_neighbors):
    start_time = time.time()

    # ********************* IMPORTING DATA *******************
    # # Loading, tabulating, and creating a DataFrame with all the data in the program.
    # # TODO: Change filepath for easier implementation.
    with open(r'C:\Users\Windows\Documents\Education\Concordia\2023_Fall\COEN432_ML_&_EA\Assignmnet_2\0_data\A2_15000.txt', 'r') as file:
        lines = file.readlines()
    data = [line.strip().split(';') for line in lines]
    columns = ['id', 'sequence', 'BPPM', 'activity_level']
    ds = pd.DataFrame(data, columns=columns)
    # Removing the ID column since it doesn't contain relevant information for our problem.
    ds.drop(columns='id', inplace=True)
    # ********************* PREPROCESSING ********************
    print('Pre-processing ...')
    ds = cat_encoding(ds)
    # def dim_reduction():
    # def outliers_handling():
    # def gaussian_noise():
    # ********************* SPLITTING **********************
    print('Splitting data ...')
    # Split features and target.
    # Select all columns except the last one as features, since following cthe categorical encoding the column name of the all 
    # the features isn't only 'sequence'.
    X = ds.iloc[:, :-1]  
    y = ds.iloc[:, -1]  
    # Splitting the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # ********************* SCALLING ***********************
    print('Data scalling...')
    # Feature scaling based on the reliance of KNN on distance metrics.
    # Scalling takes place after splitting the data to make sure trainning data isnt leaked in testing data.
    # For this we remove the mean value from each instances respective feature.
    # Target not scalled, since it might affect its the interpretability, and nature.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # ********************* TRAINING ***********************
    print('Model training...')
    # Initializing the KNN model.
    # Test 1: Using default values for the KNeighborsClassifier Constructor (these are part of hyperparametization which will be used in EA).
    # n_neighbors   =5;
    # weight        ='uniform'
    # algorithm     ='auto'
    # p             =1          (Manhattan distance, p=2 is Euclidean distance)
    # n_jobs        =none          (# of parallel jobs to run for neighbors search; -1 means using all processors
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', algorithm='auto', p=1, n_jobs=None)
    knn.fit(X_train_scaled, y_train)
    # ********************* TESTING ***********************
    print('Model testing...')
    # Predicting on the independent test dataset.
    y_pred = knn.predict(X_test_scaled)
    # ********************* EVALUATING ***********************
    print('Model evaluating...')
    # Evaluating the predictions with the actual target test data.
    accuracy = accuracy_score(y_test, y_pred)
    # ********************* DISPLAYING ***********************
    print(f"*******\nAccuracy of KNN: {accuracy:.2f}\nn_neighbors = {n_neighbors}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds\n*******\n")

n_neighbors = 1
test_size = 0.01
while n_neighbors < 0.5:
    main(n_neighbors)
    n_neighbors+=0.02