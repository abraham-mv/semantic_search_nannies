import os
import time
from typing import Union, Iterable
from datetime import datetime
#from tqdm.autonotebook import tqdm

import pandas as pd
import numpy as np
import torch
from torch import NoneType
import pickle
import glob

from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction import _stop_words
import string
import re
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
from dataset import NanniesDataframe

from topic_models import load_profiles_nannies

data_path = '../data/'
today_date = datetime.today().strftime('%Y-%m-%d')

def path_builder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def word_tokenizer(text):
    text = str(text)
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0: #and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

def remove_string(original_str: str, str_to_remove: str):
    start_index = original_str.find(str_to_remove)
    if start_index != -1:
        new_str = original_str[:start_index] + original_str[start_index + len(str_to_remove):]
    else:
        new_str = original_str
    return new_str
    
def save_to_csv(data: pd.DataFrame, path:str):
    pass

class InformationRetrieval:
    def __init__(
            self,
            model: str,
            #query: str,
            load_embeddings: bool = False,
            save_embeddings: bool = False,
            save_results: Union[bool, str] = True,
            input_path: str = data_path + "input/", 
            output_path: str = data_path + "output/python_tests/",
            filename: str = "nannies_profiles_canada.csv", 
            path_to_load_embeddings: str = None):
        
        # Initializing
        self.model_name = model
        #self.query = query
        self.load_embeddings = load_embeddings
        self.save_embeddings = save_embeddings
        self.input_path = input_path
        self.output_path = output_path + self.model_name + "/"
        self.filename = filename
        self.path_to_load_embeddings = path_to_load_embeddings

        # Checking if a GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Getting data
        print("Getting the data...")
        self.data = NanniesDataframe(input_path + filename).setup()
        self.sentences = self.data.sentences
        self.date = str(pd.to_datetime(self.data.date).max())[:-9]


        # Building directories
        path_builder(self.output_path)   
        

        # Model
        print("Setting up the model...")
        self.set_model()
        self.define_search()

        

        #self.results = self.search(self.query, self.data.sentences, top_k=1000)

        #if type(save_results) == bool and save_results:
        #    results_filepath = self.output_path + "ir_results_" + self.date + ".csv"
        #    self.results.to_csv(results_filepath, index=False)
        #elif type(save_results) == str:
        #    self.results.to_csv(save_results, index=False)
        #else:
        #    pass
            

    # A useful function, I guess...
    def re_assign_null_variables(self, **kwargs):
        """
        Input parameters: 
            self (object)
            **kwards: should be attributes of the self object
        
        Changes the parameter values to the attributes in the self object if the parameters are NULL.
        """
        new_args = []
        for name, value in kwargs.items():
            new_ar = self.__getattribute__(name) if value is None else value
            new_args.append(new_ar)
        return new_args

    # Define model
    def set_model(self):
        """
        Initialize the model attribute between a bi-encoder from sentence-transformers or the BM25 scoring function.
        """
        if self.model_name.lower() == "bm25":
            self.model = self.bm25_set_model(self.data.sentences)
        else:
            self.model = SentenceTransformer(self.model_name, device=self.device)
    
    # Define search
    def define_search(self):
        """
        Define the search function for the object based on the type of model selected.
        """
        if self.model_name.lower() == "bm25":
            self.search = self.bm25_lexical_search
        else:
            self.search = self.semantic_search_bi_encoder

    

    # Set BM25 model
    def bm25_set_model(self, sentences):
        """
        Initialize the BM25 model by tokenizing the sentences by word and feeding these into the model.
        """
        tokenized_corpus = []
        for passage in sentences:
            tokenized_corpus.append(word_tokenizer(passage))
        return BM25Okapi(tokenized_corpus)
    
    # More data stuff
    def build_results_dataframe(self, query:str, sentences:Iterable[str], ids: Iterable[int], scores: list):
        """
        Input Parameters:
            query (str): the query used to computed the similarity scores.
            ids (list): and index list relating to the main object's dataframe
            scores(list): list of similarity scores
        
        Ouput parameters:
            results (pd.DataFrame): ordered dataframe of query-sentence pairs ordered by similarity score.
        """
        hits = []
        for i in range(len(scores)):
            hits.append({"score": scores[i],
                        "query": query,
                        "link": self.data.link[ids[i]],
                        "name": self.data.name[ids[i]],
                        "sentences": self.data.sentences[ids[i]]})
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        results = pd.DataFrame(hits).drop_duplicates(subset="link", keep="first").reset_index(drop=True)
        return results
    

    # Embeddings loader
    def embeddings_loader(self):
        """
        Load embeddings from path_to_load_embeddings using pickle and sends the embeddings to device. 
        WARNING: The function overrides the current object's data in place of the data stored in the pickle object.
        """

        if not self.path_to_load_embeddings and self.load_embeddings:
            file_list = glob.glob(self.output_path + '*' + self.date + '.pkl')
            if len(file_list) > 0:
                self.path_to_load_embeddings = file_list[0]
            else: 
                raise TypeError("Embeddings haven't been computed for all sentences. Set load_embeddings = False to do so.")
        else:
            pass

        path = self.path_to_load_embeddings
        self.data = pd.DataFrame()
        if self.load_embeddings and os.path.exists(path):
            
            with open(path, "rb") as input:
                stored_data = pickle.load(input)
                self.embeddings = stored_data["embeddings"].to(self.device)
                self.model_name = stored_data["model"]
                self.data['name'] = stored_data['name']
                self.data['link'] = stored_data['link']
                self.data['sentences'] = stored_data['sentences']
                self.sentences = stored_data['sentences']
                self.date = stored_data['date']
        else:
            raise TypeError("Path doesn't exist")
    
    # Save embeddings
    def embeddings_saver(self):
        """
        Save embeddings and data to path_to_load_embeddings using a pickle object. 

        WARNING: the embeddings are send to the cpu right before dumping them,
                 because it makes it easier if we want read them again later 
                 and cuda isn't available.
        """
        if os.path.exists(self.path_to_load_embeddings):
            os.remove(self.path_to_load_embeddings)
        pathname = self.output_path + "embeddings_" + self.date + ".pkl"
        with open(pathname, "wb") as output:
                    pickle.dump({'name':self.data.name,
                                 'link':self.data.link,
                                 'sentences': self.data.sentences,
                                 'embeddings': self.embeddings.cpu(), 
                                 'date': self.date,
                                 'model': self.model_name}, output, protocol=pickle.HIGHEST_PROTOCOL)

    # Encode or load embeddings
    def sentence_bi_encoder(self, sentences: pd.Series = None):
        """
        Input parameters
            sentences (pd.Series): a pandas Series containing the sentences to compute embeddings.
        
        Computes sentence embeddings using a bi-encoder from the sentence-transformers package and sends them to device.
        It also saves the embeddings to a cache folder if option is True or loads them instead.
        """
        sentences = self.re_assign_null_variables(sentences)
        
        if not self.load_embeddings:
            self.embeddings = self.model.encode(sentences, 
                                                show_progress_bar=False, 
                                                convert_to_tensor=True).to(self.device)
            if self.save_embeddings: 
                self.embeddings_saver()
        else:
            self.embeddings_loader()
    

    # Search with BM25
    def bm25_lexical_search(self, query: str,  sentences: Union[pd.Series, NoneType] = None, top_k: int = 5, threshold: float = None):
        """
        Computes scores for each sentence using the BM25 function.
        """    

        sentences = self.re_assign_null_variables(sentences=sentences)[0]

        top_k = min(len(sentences), top_k)
        
        bm25 = self.bm25_set_model(sentences=sentences)
        bm25_scores = bm25.get_scores(word_tokenizer(query))
        top_n = np.argpartition(bm25_scores, -top_k)[-top_k:]
        ids = [sentences.index[idx] for idx in top_n]
        scores = [bm25_scores[idx] for idx in top_n]
        return self.build_results_dataframe(query, sentences, ids, scores)
    
    
    ################
    ## BI-ENCODER ##
    ################

    # Prepare embeddings for semantic search
    def prepare_embeddings(self, 
                           query: str, 
                           sentences: Union[pd.Series, NoneType] = None):
        
        pass
    
    # Semantic search with bi_encoder
    def semantic_search_bi_encoder(self, 
                                   query: str, 
                                   model: Union[SentenceTransformer, None] = None,
                                   sentences: pd.Series = None,  
                                   score_function = util.cos_sim, 
                                   top_k: int = 1000, 
                                   threshold: float = None):
        """
        Computes scores for each sentence using the bi-encoder.
        """
        model, sentences = self.re_assign_null_variables(model=model, sentences=sentences)
        
        if not isinstance(model, SentenceTransformer):
            model = SentenceTransformer('all-MiniLM-L6-v2')

        top_k = min(len(sentences), top_k)

        if not hasattr(self, "embeddings"):
            self.sentence_bi_encoder(sentences=sentences)
        
        corpus_embeddings = self.embeddings[sentences.index]
        query_embeddings = model.encode(query, convert_to_tensor=True).to(self.device)

        if top_k:
            hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=score_function)[0]
        elif threshold:
            hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=len(corpus_embeddings), score_function=score_function)[0]
            hits = [hit for hit in hits if hit['score'] > threshold]
        else:
            hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=len(corpus_embeddings), score_function=score_function)[0]
            raise ResourceWarning("A score threshold or number of top results wasn't selected, getting the score of all sentences provided.")
            

        ids = sentences.index[[hit["corpus_id"] for hit in hits]]
        scores = [hit["score"] for hit in hits]
        
        return self.build_results_dataframe(query, sentences, ids, scores)
    
    # Rank with cross-encoder
    def rank_sentences_cross_encoder(self, model_name: str, query: str, sentences: Union[pd.Series, NoneType] = None):
        """
        Computes scores for each sentence using a cross-encoder.
        """
        sentences = self.re_assign_null_variables(sentences=sentences)[0]

        model = CrossEncoder(model_name)
        cross_inp = [[query, str(sentence)] for sentence in sentences]
        cross_scores = model.predict(cross_inp)

        hits = []
        for i, score in enumerate(cross_scores):
            hits.append({'corpus_id': i,
                         'sentence': sentences[i], 
                         'score': score})

        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        return hits
    

    # Functions for specific use cases

    def retrieve_ages(self, sentences: Union[pd.Series, NoneType] = None, query: str = "I am years old", 
                     top_k: int = 1000, cache_folder: Union[str, NoneType] = None):
        """
        Age retrieval function
        
        Input parameters:
            sentences (Series): a pandas series with the sentences from which we will extract the ages. 
            query (str): ideally customized to search for authors age in the texts. Pre defined at "I am years old".
            top_k (int): the k sentences to retrieve using the object's search function. Ideally all will contain age information.
            cache_folder (str): path to store the results if not None.

        It performs semantic/lexical search on the sentences using the query. 
        For each of the top k sentences in the results dataframe we extract all numeric values to a list,
        and remove those less than 14 and greater than 100, finally we get the maximum out of those. 
        """

        results = self.search(query=query, sentences=sentences, top_k=top_k)
        results['age'] = results['sentences'].apply(lambda x: np.array([float(match) for match in re.findall(r'\d+\.\d+|\d+', x)])). \
                                              apply(lambda x: x[(x>14) & (x < 100)]). \
                                              apply(lambda x: np.nan if len(x) == 0 else np.max(x))
        
        if cache_folder:
            results.to_csv(cache_folder, index=False)

        return results
    
    def retrieve_countries(self, 
                           sentences: Union[pd.Series, NoneType] = None, 
                           query: str = "I am from", 
                           top_k: int = 1000, 
                           threshold: float = None,
                           cache_folder: str = None):
        """
        Country retrieval function
        
        Input parameters:
            sentences (Series): a pandas series with the sentences from which we will extract the country of origin. 
            query (str): ideally customized to search for authors age in the texts. Pre defined at "I am from".
            top_k (int): the k sentences to retrieve using the object's search function. Ideally all will contain country information.
            cache_folder (str): path to store the results if not None.

        1. It performs semantic/lexical search on the sentences using the query. 
        2. We read a list of adjectival and demonymic forms for countries retrieved from wikipedia into a text file.
        3. We iterate through every possible word and check matches in the top k sentences from the search. 
        4. Next we iterate through the matching sentences and skip if it's British Columbia 
           (since this is a Canadian province not a country and may be confused with British as demonym).
           and only replace the value for country if the the country name/demonym is longer
           (this was done to avoid having countries like Ukraine being identified as UK).
        5. Next we map every country name or demonymic/adjectival form into country names using the wikipedia table.

        List of adjectival and demonymic forms for countries and nations: 
        https://en.wikipedia.org/wiki/List_of_adjectival_and_demonymic_forms_for_countries_and_nations
        """
        results = self.search(query=query, sentences=sentences, top_k=top_k, threshold=threshold)
        
        countries = []
        with open("../data/input/countries.txt", "r") as file:
            for row in file:
                countries.append(row.strip())

        canadian_provinces = ["ontario", "nova scotia", "new brunswick", "alberta", "british columbia", 
                              "quebec", "saskatchewan", "manitoba", "newfoundland", "prince edward island", 
                              "montreal", "toronto"]
        
        countries.extend(canadian_provinces)
        
        results['country'] = ''
        #for country in countries:
        #    pattern = country.rstrip("s")
        #    pattern_regex =  rf'\b{pattern}s*\b|^{pattern}s*\b|\b{pattern}s*$|\b{pattern}s*\.'
        #    filtered_rows = results[results['sentences'].str.contains(pattern_regex, case=False, regex=True)]
        #    if len(filtered_rows.index) > 0:
        #        for row in filtered_rows.index:
        #            if "British Columbia".lower() in results.loc[row, "sentences"].lower():
        #                continue
        #            if len(country) > len(results.loc[row, "country"]):
        #                results.loc[row, "country"] = country
        #
        #results = results[results.country != ""]
#
        countries_df = pd.read_csv("../data/input/countries.csv")

        #
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        
        for i in results.index:
            sentence = results.loc[i, "sentences"]
            countries_match = []
            for country in countries:
                pattern = country.lower().rstrip("s")
                pattern_regex = rf'\b{pattern}s*\b'
                match = re.search(pattern_regex, sentence, flags=re.IGNORECASE)
                if match:
                    countries_match.append(country)

            if len(countries_match) == 1:
                contains_country = countries_df.applymap(lambda x: str(x).find(countries_match[0]) != -1)
                matches = countries_df[contains_country.any(axis=1)]
                
                if len(matches) > 0:
                    country_match = matches.iloc[0,0]
                else:
                    country_match = countries_match[0]
                
                if country_match in canadian_provinces: country_match = "Canada"
            
            elif len(countries_match) > 0:
                
                countries_input = []
                
                for possible_country in countries_match:
                    
                    if possible_country in canadian_provinces:
                        countries_input.append("I am from " + possible_country)
                    else:
                        contains_country = countries_df.applymap(lambda x: str(x).find(possible_country) != -1)
                        matches = countries_df[contains_country.any(axis=1)]
                    
                        if len(matches) > 0:
                            countries_input.append("I am from " + matches.iloc[0,0])
                        else: countries_input.append("")
                
                hits = self.rank_sentences_cross_encoder(model, query=sentence, sentences=countries_input)[0]

                country_match = remove_string(hits['sentence'], "I am from ")
            
                if country_match in canadian_provinces: country_match = "Canada"
                
                #query_embeddings = model.encode(sentence, convert_to_tensor=True)
                #embeddings = model.encode(countries_match, convert_to_tensor=True)
                #hits = util.semantic_search(query_embeddings, embeddings)[0]
                #country_match = remove_string(countries_match[hits[0]['corpus_id']], "I am from ")
            else:
                country_match = "Canada"

            results.loc[i, "country"] = country_match


        if cache_folder:
            results.to_csv(cache_folder, index=False)

        return results[results.country != "Canada"]

if __name__ == "__main__":
    nannies_ir = InformationRetrieval('multi-qa-mpnet-base-cos-v1', load_embeddings=True, save_results=False)
    print(nannies_ir.results)
    #print(nannies_ir.retrieve_countries(nannies_ir.data.sentences, save_to_cache=False))
