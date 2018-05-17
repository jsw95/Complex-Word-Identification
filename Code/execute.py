from utils.dataset import Dataset
from utils.system import System
from utils.scorer import report_score


def execute_system(language, modelName, featureSet):
    data = Dataset(language)


    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))
    print("Features: {}".format(featureSet))
    print("Model: {}".format(modelName))


    system = System(language, modelName, featureSet)

    print("Training...")
    system.train(data.trainset)

    print("Testing...")
    predictions = system.test(data.testset)

    gold_labels = [sent['gold_label'] for sent in data.testset]

    score = report_score(gold_labels, predictions, detailed=True)

    

    

if __name__ == '__main__':

    all_feats = ['baseline', 'cap_feat', 'freq_feat', 'uni_feat', 'bi_feat', 'tri_feat',
                'syl_feat', 'sense_feat', 'pos_feat']
   
    models = ['LogisticRegression', 'NeuralNetwork', 'RandomForest']
   
    languages = ['english', 'spanish']



    execute_system('english','RandomForest', all_feats)


