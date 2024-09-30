from opensearch_data_model import Topic, os_client
from opensearchpy import helpers
from opensearchpy import Float, OpenSearch, Field, Integer, Document, Keyword, Text, Boolean, DenseVector, Nested, Date, Object, connections, InnerDoc, helpers


import re, os
import unicodedata
from functools import wraps
import numpy as np
import pandas as pd
#from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from opensearch_data_model import TopicKeyword
from collections import defaultdict 
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm





class Cleaning_text:
    '''
    Limpiar elementos no deseados del texto 
    '''

    def __init__(self):
        # Definir los caracteres Unicode no deseados
        self.unicode_pattern    = ['\u200e', '\u200f', '\u202a', '\u202b', '\u202c', '\u202d', '\u202e', '\u202f']
        self.urls_pattern       = re.compile(r'http\S+')
        self.simbols_chars      = r"""#&’'"`´“”″()[]*+,-.;:/<=>¿?!¡@\^_{|}~©√≠"""                 # Lista de símbolos a eliminar
        self.simbols_pattern    = re.compile(f"[{re.escape(self.simbols_chars)}]")    
        self.escape_pattern     = ['\n', '\t', '\r']
        
    def _clean_decorator(clean_func):
        @wraps(clean_func)
        def wrapper(self, input_data):
            def clean_string(text):
                return clean_func(self, text)

            if isinstance(input_data, str):
                return clean_string(input_data)
            elif isinstance(input_data, list):
                return [clean_string(item) for item in input_data]
            else:
                raise TypeError("El argumento debe ser una cadena o una lista de cadenas.")
        return wrapper

    @_clean_decorator
    def unicode(self, text):
        for pattern in self.unicode_pattern:
            text = text.replace(pattern, ' ')
        return text

    @_clean_decorator
    def urls(self, text):
        return self.urls_pattern.sub(' ', text)
    
    @_clean_decorator
    def simbols(self, text):
        return self.simbols_pattern.sub(' ', text)

    @_clean_decorator
    def accents_emojis(self, text):
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
    @_clean_decorator
    def escape_sequence(self, text):
        for pattern in self.escape_pattern:
            text = text.replace(pattern, ' ').strip()
        return text
    
    @_clean_decorator
    def str_lower(self, text):
        return text.lower()
    
#---------------------------------------------------------------------------------------------------------
def clean_all(entities, accents=True, lower=True) -> list:
    """
    Función que toma una lista de entidades, realiza una operación de limpieza 
    y devuelve una lista de entidades limpias.
    """
    cleaner = Cleaning_text()

    entities_clean = []
    for ent in entities:
        clean_txt = cleaner.unicode(ent)
        clean_txt = cleaner.urls(clean_txt)
        clean_txt = cleaner.simbols(clean_txt)
        
        if accents:
            clean_txt = cleaner.accents_emojis(clean_txt)

        clean_txt = cleaner.escape_sequence(clean_txt)

        if lower:
            clean_txt = cleaner.str_lower(clean_txt)
        
        entities_clean.append(" ".join(clean_txt.split()))
            
    return entities_clean

#-----------------------------------------------------------------------------------------------------------

# Funcion para levantar los dataset. Además me quedo solo con los de fecha date_choice
def load_dataset_function(date_choice):
    path_file = f"jganzabalseenka/news_{date_choice}_24hs"
    dataset = load_dataset(path_file)
    df = pd.DataFrame(dataset['train'])
    df.sort_values("start_time_local", ascending=True, inplace=True)
    choice = "".join(date_choice.split('-'))
    df_date = df[df['start_time_local'].dt.date == pd.to_datetime(choice).date()]
    print(f"Registros para la fecha {date_choice} -> {len(df_date)} de un total de {len(df)}")
    return df_date



#Funcion para limpiar las entidades.

#SPANISH_STOPWORDS =  SSW
#SPANISH_STOPWORDS_PARTICULAR = SSWP



#-------------------------------------------------------------------------------------------------------------
def limpieza_entidades(df, SSW, SSWP):
    enti_df = df["entities"]
    enti_df_set = list(set([ ent.lower() for sublista in enti_df for ent in sublista ]))
    enti_df_clean = clean_all(enti_df_set, accents=False)
    enti_df_train = [ word for word in enti_df_clean if word not in SSW + SSWP]
    return enti_df_train


def limpieza_keywords(df, SSW, SSWP):
    key_df = df["keywords"]
    key_df_set = list(set([ keyw.lower() for sublista in key_df for keyw in sublista ]))
    key_df_clean = clean_all(key_df_set, accents=False)
    key_df_train = [ word for word in key_df_clean if word not in SSW+ SSWP]
    return key_df_train

#-------------------------------------------------------------------------------------------------------------
clean_data = Cleaning_text()

def limpieza_texto(df, SSW, SSWP):
    data_text = list(df["text"])
    proc_data_text= []
    for data_in in tqdm(data_text):
        aux = clean_data.unicode(data_in)
        aux = clean_data.urls(aux)
        aux = clean_data.simbols(aux)
        aux = clean_data.escape_sequence(aux)
        aux = " ".join([ word for word in aux.split() if word.lower() not in SSW+SSWP])
        proc_data_text.append(aux)
    return proc_data_text

#--------------------------------------------------------------------------------------------------------

def train_function(modelo, data):
    return modelo.fit_transform(data)

#--------------------------------------------------------------------------------------------------------
def tabla_frecuencia(modelo, date):
    num_topics = len(modelo.get_topic_freq())
    print(f"La cantidad de topicos, incluyendo al topico -1, para la fecha {date} es {num_topics}")
    print(modelo.get_topic_freq())

#-------------------------------------------------------------------------------------------------------

def  threshold_function(df):
    threshold = []
    num_max_topics = df["topicos"].max()
    for topico in range(num_max_topics+1):
        threshold.append(np.mean(df[df["topicos"] == topico]["probabilidad de pertenencia"]))
    return threshold


