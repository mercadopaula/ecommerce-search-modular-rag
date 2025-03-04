from langchain_ollama import ChatOllama
from product_retriever import ProductQuery

# Create an instance of the llm model
llm = ChatOllama(model="llama3.2", temperature=0, num_ctx = 24576)

def remove_price_from_query(product_query: ProductQuery) -> ProductQuery:
    """Removes price-related information from a product search query using Llama3.2.

    Args:
        query (ProductQuery): The product search query containing potential price information.

    Returns:
        ProductQuery: The query with price-related information removed.
    """
    system_instruction = (
        "You are an AI that processes product search queries. "
        "Your task is to remove any price-related information from the given query while keeping all other details intact. "
        "Do not add new words or modify the meaning. "
        "Example: 'Find jackets under 200 euros' → 'Find jackets'. "
        "Only return the cleaned query."
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Query: {product_query.query}"}
        ])
        product_query.query = response.content
        return product_query

    except Exception as e:
        print(f"Error processing query: {e}")
        return product_query  # Return the original query in case of an error