from sklearn.model_selection import train_test_split

class DatasetSplitter:
    def __init__(self, test_size=0.2):
        self.test_size = test_size

    def split(self, data): #Random state es una manera en la cual se dividen los datos para todo el dataset de manera equivalente
        train, test = train_test_split(data, test_size=self.test_size, random_state=42, stratify=data['Clase']) #el 'target' se cambia segun la columna objetivo del dataset 
        return train, test
