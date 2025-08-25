# __<center>Biomedical Text Analysis Using Deep Neural Networks</center>__

## __<center>Overview</center>__
[This project](https://github.com/VivianNg9/Biomedical-Text-Analysis-Using-Deep-Neural-Networks/blob/main/Biomedical%20Text%20Analysis%20Using%20Deep%20Neural%20Networks.ipynb) focuses on query-focused summarisation for medical questions. The goal is to determine whether a sentence extracted from relevant medical publications can be used as part of the answer to a given medical question.

## __<center>Dataset</center>__
[`data.zip`](https://github.com/VivianNg9/Biomedical-Text-Analysis-Using-Deep-Neural-Networks/blob/main/data.zip)
Using data that has been derived from the **BioASQ challenge** (http://www.bioasq.org/), after some data manipulation to make it easier to process for this assignment. 
The BioASQ challenge organises several "shared tasks", including a task on biomedical semantic question answering which we are using here. 
Utilising a labeled dataset (`bioasq10_labelled.csv`) with:
- **Questions:** Medical queries.
- **Sentences:** Extracted text from relevant publications.
- **Labels:** Binary labels indicating relevance (1 for relevant, 0 for not relevant).

## __<center>Project Workflow</center>__
### 1. Data Review 
![Data Review](https://github.com/VivianNg9/Biomedical-Text-Analysis-Using-Deep-Neural-Networks/blob/main/image/DataReview%20.png)

The columns of the CSV file are:
* `qid`: an ID for a question. Several rows may have the same question ID, as we can see above.
* `sentid`: an ID for a sentence.
* `question`: The text of the question. In the above example, the first rows all have the same question: "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
* `sentence text`: The text of the sentence.
* `label`: 1 if the sentence is a part of the answer, 0 if the sentence is not part of the answer.

**Due to limited computational resources, the dataset is divided into smaller subsets, with 50% allocated for `training`, `dev_test` and `test`.**

### 2. Simple Siamese NN 
<details>
  <summary>Click to view: Build and Train the Siamese Model :</summary>
  
```python
# Function to build and train the Siamese model
def train_siamese_model(input_shape, dense_layer_sizes, anchor_input, positive_input, negative_input, X_train, X_val):

    # Create the shared network
    shared_network = create_siamese_model(input_shape, dense_layer_sizes, dropout_rate=0)

    # Create the embeddings
    anchor_embedding = shared_network(anchor_input)
    positive_embedding = shared_network(positive_input)
    negative_embedding = shared_network(negative_input)

    # Compute the distances
    distances = DistanceLayer()([anchor_embedding, positive_embedding, negative_embedding])

    # Create the Siamese model
    siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)

    # Compile the model
    siamese_model.compile(optimizer='adam', loss=triplet_loss)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    siamese_model.fit([X_train['anchor'], X_train['positive'], X_train['negative']], np.ones(len(X_train['anchor'])), 
                       validation_data=([X_val['anchor'], X_val['positive'], X_val['negative']], np.ones(len(X_val['anchor']))),
                       epochs=3, 
                       batch_size=32, 
                       callbacks=[early_stopping])
    return siamese_model
```
</details>


**Dense Layer Configurations**</p>
I trained models with different configurations of the dense layers and evaluated their performance using the validation set. The configurations tested were:
- [16, 16, 16]
- [32, 32, 32]
- [64, 64, 64]
- [64, 32, 16]

**Evaluation and Results**</p>
For each configuration, I calculated the F1 score on the validation set. The results are as follows:

- **[16, 16, 16]**: F1 Score = 0.6951
- **[32, 32, 32]**: F1 Score = 0.7223
- **[64, 64, 64]**: F1 Score = 0.7211
- **[64, 32, 16]**: F1 Score = 0.7193

The best performing model was with the dense layer sizes [32, 32, 32], achieving an F1 score of 0.7223</p>
As testing of the test.csv file, this output indicates that for Question ID 13 and Question ID 45, the sentences with IDs 1 and 0 have the highest predicted scores.

### 3. Recurrent NN 

Max length of sentence text for training test 
![Sentence length](https://github.com/VivianNg9/Biomedical-Text-Analysis-Using-Deep-Neural-Networks/blob/main/image/sentence%20length.png)

The maximum sentence length is: 382

<details>
  <summary>Click to view: Train the Siamese LSTM Model:</summary>
  
```python
# Function to create the Siamese model with embedding and LSTM layers
def create_siamese_model_lstm(vocab_size, max_length, embedding_dim, lstm_units, dense_layer_sizes, dropout_rate=0.5):
    input = Input(shape=(max_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=(max_length,))(input)  # Embedding layer
    x = LSTM(lstm_units)(x)  # LSTM layer
    for size in dense_layer_sizes:
        x = Dense(size, activation='relu')(x)  # Dense layers with ReLU activation
        x = Dropout(dropout_rate)(x)  # Dropout layers for regularization
    return Model(inputs=input, outputs=x)  # Return the model

def train_lstm_model(vocab_size, max_length, embedding_dim, lstm_units, dense_layer_sizes, anchor_input, positive_input, negative_input, X_train, X_val):
    shared_network = create_siamese_lstm_model(vocab_size, max_length, embedding_dim, lstm_units, dense_layer_sizes, dropout_rate=0.5)
    
    anchor_embedding = shared_network(anchor_input)
    positive_embedding = shared_network(positive_input)
    negative_embedding = shared_network(negative_input)
    
    distances = DistanceLayer()([anchor_embedding, positive_embedding, negative_embedding])  # Calculate distances
    siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    siamese_model.compile(optimizer='adam', loss=triplet_loss)  # Compile the model with Adam optimizer and triplet loss
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Early stopping callback
    
    siamese_model.fit([X_train['anchor'], X_train['positive'], X_train['negative']], np.ones(len(X_train['anchor'])), 
                      validation_data=([X_val['anchor'], X_val['positive'], X_val['negative']], np.ones(len(X_val['anchor']))),
                      epochs=3, batch_size=32, callbacks=[early_stopping])  # Train the model with early stopping
    
    return siamese_model
```
</details>

For the Recurrent Neural Network, a dense layer configuration of [32, 32, 32] was selected, as it demonstrated the best F1 score in the Simple Siamese Neural Network.

**LSTM and Dense Layer Configurations**</p>
I chose dense layer [32, 32, 32] that showed the best F1 score in task 1</p>
I trained models with different configurations of the LSTM and dense layers and evaluated their performance using the validation set. The configurations tested were:
- **LSTM units**: 16, 32, 64
- **Dense layers**: [32, 32, 32] 

**Evaluation and Results**</p>
For each configuration, I calculated the F1 score on the validation set. The results are as follows:

- **LSTM units 64 and dense layer sizes [32, 32, 32]**: F1 Score = 0.5091
- **LSTM units 32 and dense layer sizes [32, 32, 32]**: F1 Score = 0.6403
- **LSTM units 16 and dense layer sizes [32, 32, 32]**: F1 Score = 0.5860

The best performing model was with the LSTM units 32 and dense layer sizes [32, 32, 32], achieving an F1 score of 0.6403 which is lower than Task 1. This performance drop is likely due to the increased complexity of the LSTM-based model, which may have led to overfitting and suboptimal parameter tuning.</p>
As testing of the test.csv file, this output indicates that for Question ID 13 and Question ID 45, the sentences with IDs 0 and 1 have the highest predicted scores.

### 4. Transformer

Implement a simple Transformer neural network that is composed of the following layers:

* Use BERT as feature extractor for each token.
* A few of transformer encoder layers, hidden dimension 768. You need to determine how many layers to use between 1~3.
* A few of transformer decoder layers, hidden dimension 768. You need to determine how many layers to use between 1~3.
* 1 hidden layer with size 512.
* The final output layer with one cell for binary classification to predict whether two inputs are related or not.

Note that each input for this model should be a concatenation of a positive pair (i.e. question + one answer) or a negative pair (i.e. question + not related sentence). The format is usually like [CLS]+ question + [SEP] + a positive/negative sentence.

Train the model with the training data, use the dev_test set to determine a good size of the transformer layers, and report the final results using the test set. Again, remember to use the test set only after you have determined the optimal parameters of the transformer layers.

<details>
  <summary>Click to view: Define the Transformer Model:</summary>
  
```python
# Define the Transformer model
class TransformerModel(tf.keras.Model):
    def __init__(self, hidden_size, num_encoder_layers=1):
        super(TransformerModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        
        # Set the pooler layer to be non-trainable
        self.bert.bert.pooler.trainable = False
        # Define transformer encoder layers
        self.encoder_layers = [tf.keras.layers.Dense(768, activation='relu', dtype='float32') for _ in range(num_encoder_layers)]
        # Define a hidden dense layer
        self.hidden_layer = tf.keras.layers.Dense(hidden_size, activation='relu', dtype='float32')
        # Define the output layer for binary classification
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        for encoder_layer in self.encoder_layers:
            sequence_output = encoder_layer(sequence_output)
        pooled_output = tf.reduce_mean(sequence_output, axis=1) # Pool the output by taking the mean across the sequence
        hidden_output = self.hidden_layer(pooled_output) # Apply the hidden layer
        logits = self.output_layer(hidden_output) # Apply the output layer
        return logits

# Initialize the model
model = TransformerModel(hidden_size=64)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) # Use Adam optimizer
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False) # Use binary cross-entropy loss
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # Compile the model

print("Model defined and compiled.")
```
</details>

**Transformer Layer Configurations**</p>
I trained models with different configurations of the transformer layers and evaluated their performance using the validation set. The configurations tested were:
- Encoder layers: 1, 2, 3

**Evaluation and Results**</p>
For each configuration, I calculated the F1 score on the validation set. The results are as follows:

- **1 Encoder Layer for dev_test set**: F1 Score = 0.05286783042394015 
- **2 Encoder Layers for test set**: F1 Score = 0.07497565725413827

The best performing model was with 2 encoder layers, achieving an F1 score of 0.07497565725413827.</p>
As testing `test.csv` file, this output indicates that for the given test inputs, the sentence with ID 3944 has the highest predicted score.
