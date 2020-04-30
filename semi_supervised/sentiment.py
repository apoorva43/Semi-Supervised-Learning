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
    sentiment.count_vect = CountVectorizer()
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
    

def read_files_new(tarfname, labels, unlabeled):
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
    print(len(sentiment.train_data)) # added
    '''len_10 = (int)(perc/100 * len(sentiment.train_data))
    print("*** LEN", len_10)
    sentiment.train_data = sentiment.train_data[0: len_10]
    sentiment.train_labels = sentiment.train_labels[0: len_10]
    print(len(sentiment.train_data))'''
    print("**&&**",len(labels), len(unlabeled.data))
    len_90 = (int)(0.9 * len(labels))
    for i in range(len_90):
        #print(unlabeled.data[i], labels[i])
        sentiment.train_data.append(unlabeled.data[i])
        sentiment.train_labels.append(labels[i])
    print(len(sentiment.train_data))
    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    len_10 = (int)(0.1 * len(labels))
    for i in range(len_10):
        #print(unlabeled.data[i], labels[i])
        sentiment.dev_data.append(unlabeled.data[i + len_90])
        sentiment.dev_labels.append(labels[i + len_90])
    
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    #sentiment.count_vect = CountVectorizer()
    sentiment.count_vect = TfidfVectorizer(analyzer = 'word', norm = 'l2', sublinear_tf = True, max_features = 10000)
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
        
    print("Len:", len(unlabeled.data)) 
    print(unlabeled.data[0])
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

def add_unlabeled(unlabeled, cls, cls_1, cls_2, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    class Data: pass
    unlabeled_new = Data()
    unlabeled_new.data = []
    
    yp_lr = cls.predict(unlabeled.X)
    yp_nb = cls_1.predict(unlabeled.X)
    yp_svm = cls_2.predict(unlabeled.X)

    labels_lr = sentiment.le.inverse_transform(yp_lr)
    labels_nb = sentiment.le.inverse_transform(yp_nb)
    labels_svm = sentiment.le.inverse_transform(yp_svm)
    
    un_lab = [] 
    cnt = 0
    for i in range(len(labels_lr)):
        if labels_lr[i] == labels_nb[i] == labels_svm[i]:
            cnt += 1
            un_lab.append(labels_lr[i])
            unlabeled_new.data.append(unlabeled.data[i])
            
    print("Count: ", cnt)
    print("Len:", len(unlabeled_new.data), len(un_lab)) 
    unlabeled_new.X = sentiment.count_vect.transform(unlabeled_new.data)
    return un_lab, unlabeled_new



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
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 5.0, 'l2')
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    
    cls_nb = classify.train_classifier_2(sentiment.trainX, sentiment.trainy)
    
    cls_svm = classify.train_classifier_3(sentiment.trainX, sentiment.trainy)
    
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    print("Unlabeled data***", len(unlabeled.data))
    lab, unlab = add_unlabeled(unlabeled, cls, cls_nb, cls_svm, sentiment)
    print("Len labeled data: ", len(lab))
    test_acc, dev_acc = [], []
    lens_ = []
    val_10 = 9152
    for i in range(10):
        lens_.append(val_10)
        val_10 += 9152
        
    print("Unlabeled data***", len(unlabeled.data))
    
    class Data: pass
    unlab_10 = Data()
    unlab_10.data = []
    print(lens_)
    perc = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in range(10):
        lab_10 = lab[0: lens_[i]]
        unlab_10.data = unlab.data[0: lens_[i]]
        unlab_10.X = unlab.X[0: lens_[i]]
        print("Now: ", len(lab_10))
        sentiment = read_files_new(tarfname, lab, unlab, perc[i])
        cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 10.5, 'l2')
        print("\nEvaluating ")
        t_acc = classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
        d_acc = classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
        test_acc.append(t_acc)
        dev_acc.append(d_acc)
    
    C = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.plot(C, dev_acc)
    plt.xlabel('Percentage of labeled data used')
    plt.ylabel('Accuracy for dev data')
    plt.show()
    plt.plot(C, test_acc)
    plt.xlabel('Percentage of labeled data used')
    plt.ylabel('Accuracy for train data')
    plt.show()
    
    sentiment = read_files_new(tarfname, lab, unlab)
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 10.5, 'l2')
    print("\nEvaluating ")
    t_acc = classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    d_acc = classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    
    print("Weights: ")
    words = ['best', 'worst', 'average', 'great', 'horrible']
    for i in range(len(sentiment.count_vect.get_feature_names())):
        for word in words:            
            if sentiment.count_vect.get_feature_names()[i] == word:
                print(cls.coef_[0][i], ": ", sentiment.count_vect.get_feature_names()[i])
    
    coefficients=cls.coef_[0]
    k = 6
    top_k =np.argsort(coefficients)[-k:]
    top_k_words = []

    print('-'*50)
    print('Top k=%d' %k)
    print('-'*50)

    for i in top_k:
        print(sentiment.count_vect.get_feature_names()[i])
        top_k_words.append(sentiment.count_vect.get_feature_names()[i])

    print('-'*50)
    print('Bottom k=%d' %k)
    print('-'*50)

    bottom_k =np.argsort(coefficients)[:k]
    bottom_k_words = []

    for i in bottom_k:
        print(sentiment.count_vect.get_feature_names()[i])
        bottom_k_words.append(sentiment.count_vect.get_feature_names()[i])
        
    top_k_reviews = []
    top_k_labels = []
    for j in range(len(sentiment.dev_data)):
        review = sentiment.dev_data[j]
        for word in top_k_words:
            if word in review:
                top_k_reviews.append(review)
                top_k_labels.append(sentiment.dev_labels[j])
                break

            #print(review)
    dev_X = sentiment.count_vect.transform(top_k_reviews)
    classify.evaluate(dev_X, sentiment.le.transform(top_k_labels), cls, 'these reviews')


    yp = cls.predict_proba(dev_X) 
    #print(yp.shape)
    plt.plot(yp[:,0],'b.') # ['0: NEGATIVE', '1:POSITIVE'] 
    plt.plot(yp[:,1],'ro') 
    plt.title('Classifier Predictions on reviews with top-k words (highly positive)')
    plt.ylabel('Predicted score')
    plt.xlabel('Example #')
    plt.show()
    
    bottom_k_reviews = []
    bottom_k_labels = []
    for j in range(len(sentiment.dev_data)):
        review = sentiment.dev_data[j]
        for word in bottom_k_words:
            if word in review:
                bottom_k_reviews.append(review)
                bottom_k_labels.append(sentiment.dev_labels[j])
                break

            #print(review)
    dev_X = sentiment.count_vect.transform(bottom_k_reviews)
    classify.evaluate(dev_X, sentiment.le.transform(bottom_k_labels), cls, 'these reviews')
    yp = cls.predict_proba(dev_X) 
    #print(yp.shape)
    plt.plot(yp[:,0],'b.') # ['0: NEGATIVE', '1:POSITIVE'] 
    plt.plot(yp[:,1],'ro') 
    plt.title('Classifier Predictions on reviews with botton-k words (highly negative)')
    plt.ylabel('Predicted score')
    plt.xlabel('Example #')
    plt.show()
    
    unlabeled = read_unlabeled(tarfname, sentiment)
    print("Writing predictions to a file")
    write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
    #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
