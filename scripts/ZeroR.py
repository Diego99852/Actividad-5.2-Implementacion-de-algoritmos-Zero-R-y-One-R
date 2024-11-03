class ZeroR:
    def __init__(self):
        self.prediction = None

    def fit(self, data, target_column):
        class_counts = data[target_column].value_counts()
        self.prediction = class_counts.idxmax()

    def predict(self, data):
        return [self.prediction] * len(data)
