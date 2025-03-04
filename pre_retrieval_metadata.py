""" 
Metadata Extraction Module for Fashion Product Queries

This module provides functions to extract structured metadata from fashion product 
search queries using Llama3.2. It extracts the following attributes:

- **Price Amount**: Identifies and extracts a numeric price from the query.
- **Comparison Operator**: Maps natural language comparisons to structured operators.
- **Product Category**: Classifies the query into a predefined product category.

Dependencies:
    - product_category.py (get_product_category)
    - product_retriever.py (ProductQuery)
    - Llama3.2 model via LangChain-Ollama
    - Pydantic for structured outputs

Example Usage:
    ```python
    query = ProductQuery(query="Show me dresses under 300 euros")
    metadata = extract_metadata(query)
    print(metadata)
    ```
"""

from product_category import get_product_category 
from decimal import Decimal
from pydantic import BaseModel
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from product_retriever import ProductQuery
import re

# Define the response model for structured output
class PriceResponse(BaseModel):
    price: Decimal | None  # The extracted price as a Decimal or None if not found

# Llama3.2 model 
llm = ChatOllama(model="llama3.2", temperature=0, num_ctx = 24576)

def extract_price_amount(query: str) -> Decimal | None:
    """Extracts the price amount from a query string.
    
    Args:
        query (str): The input query string containing a potential price.
    
    Returns:
        Decimal | None: The extracted price amount as a Decimal, or None if no price is found.
    """
    values = re.findall(r"\d+\.?\d*", query) # Supports integers and decimals
    return Decimal(max(values)) if len(values) != 0 else None


# Define the response model with a restricted set of valid operators
class ComparisonOperatorResponse(BaseModel):
    operator: Literal["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"] | None

# Create an instance of the structured output parser
llm_operator_extractor = llm.with_structured_output(ComparisonOperatorResponse)

def extract_comparison_operator(query: str) -> str | None:
    """Extracts a number comparison operator from a query string using Llama3.2.
    
    Args:
        query (str): The input query string containing a potential comparison operator.
    
    Returns:
        str | None: The mapped comparison operator ('$eq', '$ne', '$gt', '$gte', '$lt', '$lte') or None if not found.
    """
    system_instruction = (
        "You are an AI assistant that extracts number comparison operators from text. "
        "Identify the operator in the given query and map it to one of the following: "
        "'$eq' (equal to), '$ne' (not equal to), '$gt' (greater than), "
        "'$gte' (greater than or equal to), '$lt' (less than), '$lte' (less than or equal to). "
        "Respond with only the operator. "
        "If no comparison is found, return null."
    )

    try:
        response = llm_operator_extractor.invoke([
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Query: {query}"}
        ])
        
        return response.operator if response is not None else None

    except Exception as e:
        print(f"Error extracting comparison operator: {e}")
        return None

def extract_price_comparison(product_query: ProductQuery) -> ProductQuery:
    """Extracts price and comparison operator from a product search query.

    This function updates the `metadata` of a `ProductQuery` instance with:
    - `price_amount`: Extracted price as a string.
    - `comparison_operator`: Structured comparison operator.

    Args:
        product_query (ProductQuery): The product query instance.
    
    Returns:
        ProductQuery: The updated product query with extracted metadata.
    """
    price_amount = extract_price_amount(product_query.query)
    comparison_operator = extract_comparison_operator(product_query.query)
    if price_amount is not None and comparison_operator is not None:
        product_query.metadata['price_amount'] = str(price_amount)
        product_query.metadata['comparison_operator'] = comparison_operator
        return product_query
    else:
        return product_query # Return the original query

def extract_product_category(product_query: ProductQuery) -> ProductQuery:
    """Extracts the product category from a query.

    This function classifies a product query into a predefined fashion category 
    and updates the `metadata` of the `ProductQuery` instance.

    Args:
        product_query (ProductQuery): The product query instance.
    
    Returns:
        ProductQuery: The updated product query with extracted category metadata.
    """
    product_category = get_product_category(product_query.query)
    if product_category:
        product_query.metadata['category'] = product_category
    return product_query

def extract_metadata(product_query: ProductQuery) -> ProductQuery:
    """Extracts structured metadata from a product search query.

    This function combines multiple metadata extraction functions, including:
    - **Price extraction** (numeric amount and comparison operator)
    - **Product category classification**

    Args:
        product_query (ProductQuery): The product query instance.
    
    Returns:
        ProductQuery: The updated product query containing extracted metadata.
    """
    product_query = extract_price_comparison(product_query)
    product_query = extract_product_category(product_query)
    return product_query