import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string 

from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec


date = '2023-06-21'
data_path = '../data/' 
current_date = str(np.datetime64('today'))

def load_profiles_nannies(datapath:str = "../data/input/", 
                         pattern:str = "profiles_ontario.csv"):
    pattern = '*' + pattern + '*'
    file_list = glob.glob(f"{datapath}/**/{pattern}", recursive=True)
    profiles_df = []
    for file in file_list:
        profiles_df.append(pd.read_csv(file, index_col=0))
    data = pd.concat(profiles_df).reset_index(drop=True)
    data = data.drop_duplicates(subset=["short_blurb", "reasons", "about_me", "link"], keep='last')
    return data.reset_index(drop=True)

def data_setup(datapath:str, pattern:str): 
    profiles_clean = load_profiles_nannies(datapath=datapath, pattern=pattern)
    profiles_clean = profiles_clean.fillna('')
    profiles_clean_text = profiles_clean[["short_blurb", "reasons", "about_me"]].apply(lambda x: x.str.replace('\r\n'," . "))
    text = profiles_clean_text.short_blurb + ". " + profiles_clean_text.reasons + ". " + profiles_clean_text.about_me
    text = text.str.replace(r'([.,;:!?])(?=[^\s.])', r'\1 ', regex=True)   
    data = pd.DataFrame({
        "name": profiles_clean["name"],
        "date": profiles_clean["date"],
        "link": profiles_clean["link"]
        #"text": self.text
    })
    return text, data


class TopicModelNannys:
    def __init__(
        self,
        model_embed: str = 'all-MiniLM-L6-v2',
        load_embeddings: bool = False,
        #cluster_algorithm: str = "kmeans",
        input_path = data_path + "input/", 
        output_path = data_path + "output/python_tests/",
        filename: str = "profiles_ontario.csv"
    ):
        
        self.model_embed = model_embed
        self.input_path = input_path
        self.output_path = output_path + model_embed + "/" + current_date
        self.filename = filename
    
        self.text, self.data = data_setup(self.input_path, self.filename)
        #self.get_model()
        self.model = SentenceTransformer(self.model_embed)
        
        self.sentences = self.sentence_tokenizer(self.text)
        self.data["sentences"] = self.sentences
        self.data = self.data.explode('sentences').reset_index(drop=True)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if load_embeddings:
            self.output_path = data_path + "python_tests/" + model_embed + "/" + date
            self.embeddings = pd.read_csv(self.output_path + '/embeddings_' + model_embed + ".csv", index_col = 0).to_numpy()
        else:
            self.embeddings = self.model.encode(self.data.sentences.values, show_progress_bar=True)
            pd.DataFrame(self.embeddings).to_csv(self.output_path + "/embeddings_" + model_embed + ".csv")
        
        # Running clustering
        embeddings_red = self.pca(n_dim=10)
        self.data["kmeans_topic"] = self.kmeans(embeddings_red, n_clusters=5)
        self.data.to_csv(self.output_path + "/profiles_sentences_labeled.csv")
    
    def sentence_tokenizer(self, text):
        text_tokenized_sentence = [sent_tokenize(x) for x in text]
        text_tokenized_sentence = [pd.Series(x).str.replace(r'^[.;]\s*','', regex = True).tolist() for x in text_tokenized_sentence]
        text_tokenized_sentence = [[sentence for sentence in sublist if sentence] for sublist in text_tokenized_sentence]
        
        return text_tokenized_sentence
        
    # Dimensionality reduction algorithms
    def pca(self, n_dim):
         return PCA(n_components=n_dim).fit_transform(self.embeddings)
    
    # Clustering models
    def kmeans(self, x, n_clusters):
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, n_init = 'auto').fit(x)
        return kmeans_model.labels_
    
    def dbscan(self, x, eps=0.5, min_samples=5):
        cluster_model = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        return cluster_model.labels_        

    def gaussian_mixture(self, x, n_clusters):
        gm = GaussianMixture(n_components=n_clusters, random_state=0).fit_predict(x)
        return gm
    
    # Plotting function
    def plot_embeddings(self, labels):
        pca_data = self.pca(n_dim = 2)
        result = pd.DataFrame(pca_data, columns=['x', 'y'])
        result['labels'] = labels
        fig, ax = plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=1)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=4, cmap='hsv_r')
        plt.colorbar()
        plt.show()

    # TF_IDF per class function
    def class_based_tf_idf(self, labels, n_gram_range = (1,1)):
        self.data["topic"] = labels
        docs_per_topic = self.data.groupby('topic', as_index=False).agg({'sentences':' '.join})
        vectorizer = TfidfVectorizer(ngram_range=n_gram_range, stop_words="english").fit(docs_per_topic["sentences"])
        self.tf_idf = vectorizer.transform(docs_per_topic.sentences).toarray()

        return vectorizer, docs_per_topic
    
    def get_top_n_words(self, labels, n):
        vectorizer, docs_per_topic = self.class_based_tf_idf(labels)
        words = vectorizer.get_feature_names_out()
        indices = self.tf_idf.argsort()[:,-n:]
        labels = list(docs_per_topic['topic'])

        top_n_words = {}
        for i, label in enumerate(labels):
            top_n_words[label] = [(words[j], self.tf_idf[i][j]) for j in indices[i]][::-1]
        
        return top_n_words


class WordModelNannys:
    def __init__(
            self,
            model_name: str = 'word2vec', 
            input_path: str = data_path + "input/",
            output_path: str = data_path + "output/python_tests/",
            filename: str = "profiles_ontario.csv"
            ):
        
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path + model_name + "/" + current_date
        self.filename = filename

        self.text, self.data = data_setup(self.input_path, self.filename)
        sw = stopwords.words('english')
        self.data["words"] = [word_tokenize(x) for x in self.text]
        self.data.words = self.data.words.apply(lambda x: list(filter(lambda y: y not in sw, x)))
        self.model = Word2Vec(self.data.words, min_count=1)
       
        words_dict = self.model.wv.key_to_index
        self.embeddings = self.model.wv[words_dict.keys()]
        
        words_labels = pd.DataFrame()
        words_labels['words'] = words_dict.keys()
        embeddings_red = self.pca(n_dim = 10)
        words_labels['labels'] = self.kmeans(embeddings_red, n_clusters = 10)
        
        self.data = self.data.explode('words').reset_index(drop = True).merge(words_labels, on = 'words', how = 'inner')
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        self.data.to_csv(self.output_path + "/profiles_words_labeled.csv")
        pd.DataFrame(self.embeddings).to_csv(self.output_path + "/embeddings_" + model_name + ".csv")

    # Dimensionality reduction algorithms
    def pca(self, n_dim):
         return PCA(n_components=n_dim).fit_transform(self.embeddings)
    
    # Clustering models
    def kmeans(self, x, n_clusters):
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, n_init = 'auto').fit(x)
        return kmeans_model.labels_
    
    def dbscan(self, x, eps=0.5, min_samples=5):
        cluster_model = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        return cluster_model.labels_        

    def gaussian_mixture(self, x, n_clusters):
        gm = GaussianMixture(n_components=n_clusters, random_state=0).fit_predict(x)
        return gm
    
    # Plotting function
    def plot_embeddings(self, labels):
        pca_data = self.pca(n_dim = 2)
        result = pd.DataFrame(pca_data, columns=['x', 'y'])
        result['labels'] = labels
        fig, ax = plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=1)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=4, cmap='hsv_r')
        plt.colorbar()
        plt.show()
    
    

if __name__ == "__main__":
    #model = TopicModelNannys()
    model = WordModelNannys()
    embeddings_red = model.pca(n_dim=5)
    #model.data["gaussian_labels"] = model.gaussian_mixture(embeddings_red, n_clusters=10)
    labels = model.kmeans(embeddings_red, n_clusters=10)
    #labels = model.dbscan(embeddings_red, eps = 0.05, min_samples = 10)
    model.plot_embeddings(labels)
    
    #model.data.to_csv(model.output_path + "/profiles_sentences_labeled.csv")
    #pd.DataFrame(model.get_top_n_words(model.data["gaussian_labels"], 20)).to_csv(model.output_path + "/top_20_words.csv")
