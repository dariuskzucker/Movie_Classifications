# Movie_Review_Classifications

## Goal

Train an SVM on Amazon Prime Video's movie review dataset to classify each review's sentiment as positive, neutral, or negative.

## Implementation

### Preprocessing
#### Deleting stop-words

- Stop-words such as “the”, “and”, “it”, overpopulate the training data, without providing us with useful information about the sentiment of the text.
- Because the feature engineering process ends up being very memory intensive and computationally expensive, deleting these stop-words decreases the feature space without losing important sentiment classification.

#### Lemmatization

- Imagine a model that sees the word “good” hundreds of times in the training data, associating its presence with positive sentiment. Suppose the word “best” wasn’t in the training dataset, and the model isn’t able to apply any of its knowledge of the word “good” to it, despite the two words having the same lexical root.
- Lemmatization solves this problem, reducing all words to their root value before feature engineering, so that we can gain more value in our word comparisons.
- Additionally, it reduces the feature space greatly.


### Data Augmentation

#### EDA Synonym Replacement

- Used wordnet’s synonym feature to generate (n+1) times the number of training data points, by creating n new data points where each word gets replaced with a random synonym. Keeps the same label.
- Allows the model to learn many more words and sentences that ideally have the same meaning.

### Feature engineering

#### Custom Bag of N-Grams model

- Adapted the simple bag of words model to be much more sophisticated, capturing phrases as well as words.
- Bag of words model creats a feature matrix where fm[i][j] represents if the ith word in the dictionary appears in training datapoint j.
- Custom Bag of N-Gram model creates a feature matrix where fm[i][j] represents how many times the ith n-gram in the dictionary appears in training datapoint j.
- Given an input n, I create a dictionary of all possible n-grams, (n-1)-grams, (n-2)-grams, …, and unigrams.
- Implemented these calculations myself rather than using a library
- Many n-grams are unimportant, yet appear a great amount in our training data and dominate the feature space. 
- To account for this, I perform L2 normalization on the data to make our standardize the magnitude of our data points.

### Hyperparameter Selection

- Training data is balanced - no need to tune for class weight.
- Tuned all of the following parameters based on cross validation scores.
- Tuned linearSVC for C and r.
- Tuned kernel SVC for C, gamma, and different kernel types.
- Tuned the number of augmentations in the dataset
- Tuned the n in “n-grams”

### Algorithm selection

- Chose between linearSVC and SVC based on CV empirical testing of all of the kernel’s.
- Ended up with ‘rbf’ as the best

### Multiclass selection

- SVC already defaults to one-vs-one for multi-classification
- I preferred this functionality over one-vs-rest because
- I want each classifier to be given balanced training datasets since I’m not tuning class weights
- Data probably isn’t linearly separable

