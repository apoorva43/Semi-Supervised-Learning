#!/bin/python
import matplotlib.pyplot as plt
import numpy as np

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))
    
    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    #sentiment.count_vect = CountVectorizer()
    sentiment.count_vect = TfidfVectorizer(analyzer = 'word', norm = 'l2', sublinear_tf = True)
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    #transformer = TfidfTransformer()
    #sentiment.trainX = transformer.fit_transform(sentiment.trainX)
    #sentiment.devX = transformer.transform(sentiment.devX)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    #print("Features: \n")
    #print(sentiment.count_vect.get_feature_names()[0])
    tar.close()
    return sentiment

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    import classify
    test_acc, dev_acc, max_dev_acc, best_c, best_p = [], [], 0.0, 0.0, 'l2'
    testacc, devacc = [], []
    for c in [0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0]:
        for p in ['l1', 'l2']:
            cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, c, p)
            print("\nEvaluating at C = ", c, " , Penalty = ", p)
            t_acc = classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
            d_acc = classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
            if p == 'l1':
                test_acc.append(t_acc)
                dev_acc.append(d_acc)
            else:
                testacc.append(t_acc)
                devacc.append(d_acc)
            if d_acc > max_dev_acc:
                best_c = c
                best_p = p
                max_dev_acc = d_acc
            
    print("\nBest c: ", best_c, ", Best penalty: ", best_p, " | Accuracy: ", max_dev_acc)
    C = [0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0]
    plt.plot(C, dev_acc)
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy for dev data')
    plt.show()
    plt.plot(C, test_acc)
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy for train data')
    plt.show()
    plt.plot(C, devacc)
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy for dev data')
    plt.show()
    plt.plot(C, testacc)
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy for train data')
    plt.show()
    
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 5.0, 'l2')
    print("\nEvaluating at C = ", best_c, " and penalty = ", best_p)
    print(classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train'))
    print(classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev'))
    
    print("Weights: ")
    words = ['best', 'worst', 'average', 'great', 'horrible']
    for i in range(len(sentiment.count_vect.get_feature_names())):
        for word in words:            
            if sentiment.count_vect.get_feature_names()[i] == word:
                print(cls.coef_[0][i], ": ", sentiment.count_vect.get_feature_names()[i])
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    print("Writing predictions to a file")
    write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
    #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
