import config
from product_retriever import ProductQuery
from langchain_ollama import ChatOllama

# Initialize the Ollama LLM with the Llama 3 model
llm = ChatOllama(model="llama3.2", temperature=0, num_ctx = 24576)

def generate_response(product_query: ProductQuery, search_results: str, customer_preferences: str) -> str:
    
    # System prompt
    system_prompt = f"You are a personal fashion stylist. Your task is to explain how each of the recommended products match the user query. Include the name, color, material of each product. Do not forget to mention the price for each product. Recommended products: {search_results}."
    generate_user_message_query = f"User query: {product_query.query}."

    # User messages
    user_messages = [{"role": "user", "content": generate_user_message_query}]

    # If enabled, personalise the results using customer preferences from purchase history 
    if config.enable_purchase_history and customer_preferences is not None:
        generate_user_message_customer_preferences = f"Here are the customer preferences: {customer_preferences}. "
        user_messages.append({"role": "user", "content": generate_user_message_customer_preferences})
        system_prompt = system_prompt + "\n\nPersonalise the answer by using customer preferences. Do not mention the categories."

    # Generate response
    response = llm.invoke(
        [{"role": "system", "content": system_prompt}] 
        + user_messages 
        + [{"role": "assistant", "content": "Your answer:"}])

    return response.content