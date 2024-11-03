class OneR:
    def __init__(self):
        self.best_feature = None
        self.prediction_map = {}

    def fit(self, data, target_column):
        min_error = float('inf')
        for feature in data.columns.drop(target_column):
            prediction_map = data.groupby(feature)[target_column].agg(lambda x: x.mode()[0]).to_dict()
            predictions = data[feature].map(prediction_map)
            error = (predictions != data[target_column]).sum()
            if error < min_error:
                min_error = error
                self.best_feature = feature
                self.prediction_map = prediction_map
    
    def predict(self, data):
        return data[self.best_feature].map(self.prediction_map).fillna(0)
