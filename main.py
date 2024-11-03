from scripts.Loader import DataLoader
from scripts.splitter import DatasetSplitter
from scripts.ZeroR import ZeroR
from scripts.OneR import OneR
from scripts.evaluador import Evaluator

def main():
    
    loader = DataLoader("D:/tareas de la uni/6xto semestre/mineria de datos/5.1 tarea/archivos/Prueba.csv")
    data = loader.load_data()
    #Spliter define el tamaño del data set de prueba
    splitter = DatasetSplitter(test_size=0.3)
    train_data, test_data = splitter.split(data)

    zero_R = ZeroR()
    evaluator = Evaluator(iterations=5)
    accuracy_zeror = evaluator.evaluate(zero_R, train_data, test_data, target_column="Clase")
    print(f"precision de zero rule: {accuracy_zeror}")


    one_r = OneR()
    accuracy_oner = evaluator.evaluate(one_r, train_data, test_data, target_column="Clase")
    print(f"precision de One Rule: {accuracy_oner}")

if __name__ == "__main__":
    main()