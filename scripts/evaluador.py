from sklearn.metrics import accuracy_score

class Evaluator:

    def __init__(self, iterations=1):
        self.iterations = iterations
    
    def evaluate(self, model, train_data, test_data, target_column):
        model.fit(train_data, target_column)
        predictions = model.predict(test_data)
        accuracy = accuracy_score(test_data[target_column], predictions)
        return accuracy
