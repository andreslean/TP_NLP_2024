from opensearchpy import Float, OpenSearch, Field, Integer, Document, Keyword, Text, DenseVector, Nested, Date, Object, connections, InnerDoc
import os

# local test (base de datos se corre de manera local en este caso)
# docker pull opensearchproject/opensearch:latest
# docker run -it -p 9200:9200 -p 9600:9600 -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=PassWord#1234! -e "discovery.type=single-node"  --name opensearch-node opensearchproject/opensearch:latest
# docker stop opensearch-node 
# docker start opensearch-node



OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', "localhost")
auth = ('admin', 'PassWord#1234!')
port = 9200
os_client = connections.create_connection(
    hosts = [{'host': OPENSEARCH_HOST, 'port': port}],
    http_auth = auth,
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = True,
    verify_certs = False,
    alias='default'
    # ssl_assert_hostname = False,
    # ssl_show_warn = False
)

TOPIC_DIMENSIONS = 384
TOPIC_INDEX_NAME = 'topic'
TOPIC_INDEX_PARAMS = {
    'number_of_shards': 1,
    'knn': True
}

knn_params = {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "nmslib"
}

class TopicKeyword(InnerDoc):
    name = Keyword()
    score = Float()

class SimilarTopics(Document):
    topic_id = Keyword()
    similar_to = Keyword()
    similarity = Float()
    common_keywwords = Keyword()
    keywords_not_in_similar = Keyword()
    keywords_not_in_topic = Keyword()

class KNNVector(Field):
    name = "knn_vector"
    def __init__(self, dimension, method, **kwargs):
        super(KNNVector, self).__init__(dimension=dimension, method=method, **kwargs)

class Topic(Document):
    # 
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params)
    # Umbral de cercania
    similarity_threshold = Float()
    # Cuando se creo el topico
    created_at = Date()
    # Cuando fue entrenado
    to_date = Date()
    from_date = Date()
    # Es el indice de topico. Empieza en -1.
    index = Integer()
    # 
    keywords = Object(TopicKeyword)
    #Nombre del topico
    name = Text()
    #Mejor documento. 
    best_doc = Text()
    
    class Index:
        name = TOPIC_INDEX_NAME
        if not os_client.indices.exists(index=TOPIC_INDEX_NAME):
            settings = {
                'index': TOPIC_INDEX_PARAMS
            }

    def save(self, ** kwargs):
        self.meta.id = f'{self.index}' + self.name.replace(', ', '-').replace(' ', '_')
        return super(Topic, self).save(** kwargs)