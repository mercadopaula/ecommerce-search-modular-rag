"""
This module provides functionality to query and retrieve product information
from a Chroma vector store with custom embeddings using Ollama. The module includes
a custom embeddings class, product search query representation, and product retrieval 
functions based on product IDs or query metadata such as price and category.

Dependencies:
    - langchain_chroma (for Chroma integration)
    - langchain_ollama (for Ollama Embeddings)
    - chromadb (for persistent client and collection handling)
    - config (for configuration settings)
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb
import config

# Custom class to fix the signature mismatch for the embeddings function
class CustomOllamaEmbeddings(OllamaEmbeddings):
    """
    A custom class to extend and modify the OllamaEmbeddings class, ensuring
    the correct signature for embedding functions.

    Methods:
        __init__(self, model, *args, **kwargs): Initializes the embedding function
            with the given model and additional arguments.
        _embed_documents(self, texts): Embeds a list of documents using Ollama.
        __call__(self, input): Callable method to retrieve embeddings for input.
    """
    
    def __init__(self, model, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        
    def _embed_documents(self, texts):
        return super().embed_documents(texts)  # <--- use OllamaEmbeddings's embedding function

    def __call__(self, input):
        return self._embed_documents(input)    # <--- get the embeddings

# Embeddings function
embeddings = CustomOllamaEmbeddings(model="mxbai-embed-large")

# Vector store connection
collection_name = "soeur-products"

persistent_client = chromadb.PersistentClient(path="./chroma_products_souer")
collection = persistent_client.get_collection(name=collection_name, embedding_function=embeddings)

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name=collection_name,
    embedding_function=embeddings
)

retriever = vector_store_from_client.as_retriever()

from typing import Dict

class ProductQuery:
    """Represents a product search query with optional metadata.

    Attributes:
        query (str): The search query string. Defaults to an empty string.
        metadata (dict): Additional metadata for the query. Defaults to an empty dictionary.
    """

    def __init__(self, query: str = "", metadata: Dict[str, str] | None = None):
        """Initializes a ProductQuery instance.

        Args:
            query (str, optional): The search query string. Defaults to an empty string.
            metadata (dict, optional): Additional metadata for the query. Defaults to an empty dictionary.
        """
        self.query = query
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self) -> str:
        """Returns a string representation of the ProductQuery instance."""
        return f"ProductQuery(query={self.query!r}, metadata={self.metadata!r})"

def retrieve_product_by_ids(product_ids: list) -> list:
    """
    Retrieves products from the vector store by their IDs.

    Args:
        product_ids (list): A list of product IDs to search for.

    Returns:
        list: A list of documents matching the provided product IDs.
    """
    if len(product_ids) == 0:
        return []
    result = collection.query(
        query_texts="",
        where={"key": { "$in": product_ids }}
    )
    return result['documents'][0]

def retrieve_products(product_query: ProductQuery, max_results=4) -> list:
    """
    Retrieves products from the vector store based on a given product query.

    Args:
        product_query (ProductQuery): A ProductQuery instance that defines the search parameters.
        max_results (int, optional): The maximum number of results to return. Defaults to 4.

    Returns:
        list: A list of product documents or content matching the query and metadata filters.
    """
    where_clause = None
    if 'price_amount' in product_query.metadata and 'comparison_operator' in product_query.metadata:
        price_amount = product_query.metadata['price_amount']
        comparison_operator = product_query.metadata['comparison_operator']
        where_clause = {"$and":[{"gender": "women"},{"price_regular": {comparison_operator:float(price_amount)}}]}
    if 'category' in product_query.metadata:
        category = product_query.metadata['category']
        if where_clause is None:
             where_clause = {"$and":[{"gender": "women"},{"category": category}]}
        else:
            where_clause['$and'].append({"category": category})
            
    if where_clause is not None:
        retriever_output = collection.query(
            query_texts=product_query.query,
            n_results=max_results,
            where=where_clause 
        )
        return retriever_output['documents'][0]
    else:
        retriever_output = retriever.invoke(product_query.query)[:max_results]
        return [doc.page_content for doc in retriever_output]