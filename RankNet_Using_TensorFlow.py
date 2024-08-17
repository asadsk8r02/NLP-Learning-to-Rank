import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam


classRankNet:
    def__init__(self, input_shape, hidden_units=64, learning_rate=0.001):
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model, self.scoring_model = self.build_model()

    defbuild_model(self):
        input1 = Input(shape=(self.input_shape,))
        input2 = Input(shape=(self.input_shape,))

        hidden_layer = Dense(self.hidden_units, activation='relu')

        score1 = hidden_layer(input1)
        score2 = hidden_layer(input2)

        score_diff = Lambda(lambda x: x[0] - x[1])([score1, score2])
        probability = Dense(1, activation='sigmoid')(score_diff)

        model = Model(inputs=[input1, input2], outputs=probability)

        # Define the scoring model separately
        scoring_input = Input(shape=(self.input_shape,))
        scoring_output = hidden_layer(scoring_input)
        scoring_model = Model(inputs=scoring_input, outputs=scoring_output)

        return model, scoring_model

    defcompile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(self.learning_rate))

    deftrain(self, train_data_1, train_data_2, train_labels, batch_size=32, epochs=30):
        self.model.fit([train_data_1, train_data_2], train_labels, batch_size=batch_size, epochs=epochs)

    defevaluate_ndcg(self, test_grouped, k=10):
        ndcg_scores = []
        for query, group in test_grouped:
            group = group.drop('1', axis=1)  # Remove the query column
            true_rel = group['0'].values
            features = group.drop('0', axis=1).values
            pred_rel = self.scoring_model.predict(features).flatten()

            # Combine true_rel and pred_rel into a list of tuples and sort by pred_rel
            combined = list(zip(true_rel, pred_rel))
            combined.sort(key=lambda x: x[1], reverse=True)

            # Extract the sorted true_rel
            true_rel_sorted = [x[0] for x in combined]

            ndcg_scores.append(self.ndcg_at_k(true_rel_sorted, k))

        return np.mean(ndcg_scores)

    @staticmethoddefdcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    @staticmethoddefndcg_at_k(r, k):
        dcg_max = RankNet.dcg_at_k(sorted(r, reverse=True), k)
        ifnot dcg_max:
            return0.0return RankNet.dcg_at_k(r, k) / dcg_max


classDataLoader:
    @staticmethoddefload_data(file_path):
        data = pd.read_csv(file_path, sep=' ', header=None)
        data.drop(data.columns[-1], axis=1, inplace=True)  # Drop the last empty column
        data[0] = data[0].str.split(':').str.get(1).astype(int)  # Extract the relevance labelfor col inrange(1, data.shape[1]):
            data[col] = data[col].str.split(':').str.get(1).astype(float)  # Extract the feature valuesreturn data

    @staticmethoddefgenerate_pairwise_data(grouped_data):
        pairwise_data_1, pairwise_data_2, binary_preferences = [], [], []

        for _, group in grouped_data:
            group = group.drop('1', axis=1)  # Remove the query column
            docs = group.drop('0', axis=1).values
            labels = group['0'].values

            for i inrange(len(labels)):
                for j inrange(i + 1, len(labels)):
                    if labels[i] != labels[j]:
                        pairwise_data_1.append(docs[i])
                        pairwise_data_2.append(docs[j])
                        binary_preferences.append(1if labels[i] > labels[j] else0)

        return np.array(pairwise_data_1), np.array(pairwise_data_2), np.array(binary_preferences)


classRankNetRunner:
    def__init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

    defrun(self):
        # Load the MSLR-WEB10K dataset
        train_data = pd.read_csv(self.train_file)
        test_data = pd.read_csv(self.test_file)

        # Normalize the features
        scaler = MinMaxScaler()
        train_data.iloc[:, 2:] = scaler.fit_transform(train_data.iloc[:, 2:])
        test_data.iloc[:, 2:] = scaler.transform(test_data.iloc[:, 2:])

        # Group the data by query
        train_grouped = train_data.groupby('1')
        test_grouped = test_data.groupby('1')

        # Generate pairwise data
        train_pairwise_data_1, train_pairwise_data_2, train_binary_preferences = DataLoader.generate_pairwise_data(train_grouped)
        test_pairwise_data_1, test_pairwise_data_2, test_binary_preferences = DataLoader.generate_pairwise_data(test_grouped)

        # Instantiate and compile the RankNet model
        ranknet = RankNet(input_shape=136, hidden_units=64, learning_rate=0.001)
        ranknet.compile_model()

        # Train the model using the pairwise data
        ranknet.train(train_pairwise_data_1, train_pairwise_data_2, train_binary_preferences, batch_size=32, epochs=30)

        # Evaluate the model
        ndcg = ranknet.evaluate_ndcg(test_grouped)
        print("NDCG@10:", ndcg)


# Usageif __name__ == "__main__":
    runner = RankNetRunner("fold1_train_sample.csv", "fold1_test_sample.csv")
    runner.run()
