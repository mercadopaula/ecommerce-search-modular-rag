"""Module for tagging fashion products with predefined categories.

This module provides functionality to classify fashion products from the brand Soeur Paris into a set of predefined categories using an LLM (Llama 3.2) integrated with structured output parsing.

Dependencies:
    - pydantic
    - langchain
    - typing
    - ollama (LLM integration)

Attributes:
    product_categories (list): List of predefined product categories.

Functions:
    get_product_category(product: str) -> str:
        Classifies a given product into one of the predefined categories.
"""

from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from typing import Literal
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0, num_ctx = 24576)

# List of predefined product categories from the brand Soeur Paris
product_categories = [
    "Coats & Jackets", 
    "Pullovers & Cardigans",
    "Shirts & Tops",
    "Dresses",
    "T-shirts & Sweatshirts",
    "Trousers",
    "Denim",
    "Skirts & Shorts",
    "Frère",
    "Shoes",
    "Bags",
    "Small leather goods",
    "Hats",
    "Scarves",
    "Gloves",
    "Socks",
    "Belts",
    "Jewellery",
    "Objects"
]

product_categories_as_string = ", ".join(product_categories)

# Create Pydantic Model to enforce valid categories
class CategoryResponse(BaseModel):
    category: Literal[    
        "Coats & Jackets", 
        "Pullovers & Cardigans",
        "Shirts & Tops",
        "Dresses",
        "T-shirts & Sweatshirts",
        "Trousers",
        "Denim",
        "Skirts & Shorts",
        "Frère",
        "Shoes",
        "Bags",
        "Small leather goods",
        "Hats",
        "Scarves",
        "Gloves",
        "Socks",
        "Belts",
        "Jewellery",
        "Objects"
    ]

# Initialize LLM model with structured output
llm_product_category = llm.with_structured_output(CategoryResponse)

def get_product_category(product: str) -> str:
    """Classifies a product query into one of the predefined product categories.
    
    This function takes a product description as input and utilizes a language model 
    to classify the product into one of the predefined categories. The model follows 
    strict structured output enforcement to ensure valid responses.
    
    Args:
        product (str): The product description to classify.

    Returns:
        str: The classified category name if valid, otherwise an error message.
    
    Raises:
        Exception: If the LLM model fails to process the request or returns an invalid response.
    """
    
    system_prompt = f"You are a fashion product classifier. Classify the following product into exactly one of the predefined product categories. Product categories: {product_categories_as_string}. Respond with only the category name, and nothing else."

    try:
        response = llm_product_category.invoke(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": f"This is the product: {product}."},
            ])
        return response.category  # Return the validated category
    except Exception as e:
        return f"Error: {str(e)}"
