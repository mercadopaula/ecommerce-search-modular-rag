"""
This module provides functionality to analyze customer purchase history and extract fashion preferences
using the LangChain and Ollama model API. The module retrieves product IDs associated with a customer, 
formats the purchase history, and then uses an AI model to generate insights into a user's fashion preferences.

Functions:
    get_product_ids_by_customer(customer_id: int) -> list:
        Retrieves all product IDs for a given customer based on their purchase history.
        
    extract_fashion_preferences(customer_id: int) -> str:
        Analyzes a customer's purchase history and extracts their fashion preferences 
        using a language model to identify patterns across different fashion categories.
"""

from langchain_ollama import ChatOllama
from product_retriever import retrieve_product_by_ids

llm = ChatOllama(model="llama3.2", temperature=0, num_ctx = 24576)

def get_product_ids_by_customer(customer_id: int) -> int:
    """Retrieve all product IDs for a given customer ID.
    
    Args:
        customer_id (int): The ID of the customer whose product history is to be retrieved.
    
    Returns:
        list: A list of product IDs associated with the given customer. Returns an empty list if the customer ID is not found.
    """
    for customer in customer_purchase_orders:
        if customer["customer_id"] == customer_id:
            return [product["product_id"] for order in customer["purchase_orders"] for product in order["products"]]
    return []  # Return an empty list if customer ID is not found

def extract_fashion_preferences(customer_id: int) -> str:
    """Extracts user fashion preferences from the customer purchase history.

    Args:
        customer_id (int): the customer id.

    Returns:
        str: A comma-separated list of user fashion preferences, sorted by category.
    """
    system_instruction = (
        "You are an expert fashion analyst. Your task is to analyze a list of purchased fashion products "
        "and extract the user's fashion preferences. Identify patterns based on the following categories: "
        "Styles, Colors, Fabrics, Fit, Occasion-Based Preferences, Accessory Preferences, "
        "Cultural & Subcultural Influences, Sustainability & Ethical Preferences, Budget, "
        "Functional & Practical Choices, and Trend Adoption. "
        "Return a comma-separated list of preferences, sorted by category."
    )

    product_ids = get_product_ids_by_customer(customer_id)
    if len(product_ids) > 0:
        purchase_history = retrieve_product_by_ids(product_ids)
        purchase_history_formatted = "\n\n".join(purchase_history)
        print("\n---Purchase history---\n")
        print(purchase_history_formatted)
        try:
            response = llm.invoke([
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Purchase History: {purchase_history_formatted}"}
            ])
            return response.content
    
        except Exception as e:
            print(f"Error extracting preferences: {e}")
            return None  # Return None if there's an error
    else:
        return None # Return None if there's no purchase history

# Synthetic generated data using the notebook synthetic-purchase-data-generator.ipynb
customer_purchase_orders = [
    {
        "customer_id": 1,
        "purchase_orders": []
    },
    {
        "customer_id": 2,
        "purchase_orders": [
            {
                "purchase_date": "2023-11-06",
                "products": [
                    {
                        "product_id": 194,
                        "price": 623.13,
                        "quantity": 1
                    },
                    {
                        "product_id": 347,
                        "price": 770.0,
                        "quantity": 1
                    },
                    {
                        "product_id": 212,
                        "price": 241.41,
                        "quantity": 1
                    },
                    {
                        "product_id": 631,
                        "price": 738.05,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-05-24",
                "products": [
                    {
                        "product_id": 423,
                        "price": 625.09,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-06-15",
                "products": [
                    {
                        "product_id": 88,
                        "price": 761.02,
                        "quantity": 1
                    },
                    {
                        "product_id": 532,
                        "price": 715.57,
                        "quantity": 1
                    },
                    {
                        "product_id": 182,
                        "price": 140.89,
                        "quantity": 1
                    },
                    {
                        "product_id": 249,
                        "price": 124.04,
                        "quantity": 1
                    },
                    {
                        "product_id": 538,
                        "price": 90.98,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-06-01",
                "products": [
                    {
                        "product_id": 544,
                        "price": 749.45,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-09-07",
                "products": [
                    {
                        "product_id": 327,
                        "price": 584.81,
                        "quantity": 1
                    },
                    {
                        "product_id": 75,
                        "price": 597.49,
                        "quantity": 1
                    },
                    {
                        "product_id": 27,
                        "price": 158.2,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-06-26",
                "products": [
                    {
                        "product_id": 636,
                        "price": 147.15,
                        "quantity": 1
                    },
                    {
                        "product_id": 15,
                        "price": 465.7,
                        "quantity": 1
                    },
                    {
                        "product_id": 624,
                        "price": 489.32,
                        "quantity": 1
                    }
                ]
            }
        ]
    },
    {
        "customer_id": 3,
        "purchase_orders": [
            {
                "purchase_date": "2023-04-04",
                "products": [
                    {
                        "product_id": 328,
                        "price": 249.63,
                        "quantity": 1
                    },
                    {
                        "product_id": 570,
                        "price": 587.19,
                        "quantity": 1
                    },
                    {
                        "product_id": 478,
                        "price": 410.65,
                        "quantity": 1
                    },
                    {
                        "product_id": 411,
                        "price": 391.39,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-01-20",
                "products": [
                    {
                        "product_id": 433,
                        "price": 802.14,
                        "quantity": 1
                    },
                    {
                        "product_id": 54,
                        "price": 966.05,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-06-08",
                "products": [
                    {
                        "product_id": 278,
                        "price": 388.23,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-03-06",
                "products": [
                    {
                        "product_id": 481,
                        "price": 877.61,
                        "quantity": 1
                    },
                    {
                        "product_id": 562,
                        "price": 999.11,
                        "quantity": 1
                    },
                    {
                        "product_id": 432,
                        "price": 828.75,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-05-03",
                "products": [
                    {
                        "product_id": 683,
                        "price": 265.79,
                        "quantity": 1
                    }
                ]
            }
        ]
    },
    {
        "customer_id": 4,
        "purchase_orders": [
            {
                "purchase_date": "2023-10-02",
                "products": [
                    {
                        "product_id": 130,
                        "price": 81.53,
                        "quantity": 1
                    },
                    {
                        "product_id": 65,
                        "price": 637.36,
                        "quantity": 1
                    },
                    {
                        "product_id": 200,
                        "price": 598.58,
                        "quantity": 1
                    }
                ]
            }
        ]
    },
    {
        "customer_id": 5,
        "purchase_orders": [
            {
                "purchase_date": "2024-03-19",
                "products": [
                    {
                        "product_id": 668,
                        "price": 972.53,
                        "quantity": 1
                    },
                    {
                        "product_id": 579,
                        "price": 500.17,
                        "quantity": 1
                    },
                    {
                        "product_id": 484,
                        "price": 980.05,
                        "quantity": 1
                    },
                    {
                        "product_id": 646,
                        "price": 55.68,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-04-09",
                "products": [
                    {
                        "product_id": 586,
                        "price": 388.61,
                        "quantity": 1
                    },
                    {
                        "product_id": 661,
                        "price": 262.99,
                        "quantity": 1
                    },
                    {
                        "product_id": 215,
                        "price": 684.68,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-10-28",
                "products": [
                    {
                        "product_id": 24,
                        "price": 144.4,
                        "quantity": 1
                    },
                    {
                        "product_id": 652,
                        "price": 539.42,
                        "quantity": 1
                    },
                    {
                        "product_id": 513,
                        "price": 846.18,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-02-25",
                "products": [
                    {
                        "product_id": 242,
                        "price": 395.7,
                        "quantity": 1
                    },
                    {
                        "product_id": 417,
                        "price": 677.56,
                        "quantity": 1
                    },
                    {
                        "product_id": 183,
                        "price": 616.57,
                        "quantity": 1
                    },
                    {
                        "product_id": 104,
                        "price": 333.9,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-04-12",
                "products": [
                    {
                        "product_id": 332,
                        "price": 576.52,
                        "quantity": 1
                    },
                    {
                        "product_id": 498,
                        "price": 671.99,
                        "quantity": 1
                    }
                ]
            }
        ]
    },
    {
        "customer_id": 6,
        "purchase_orders": [
            {
                "purchase_date": "2024-03-15",
                "products": [
                    {
                        "product_id": 448,
                        "price": 595.04,
                        "quantity": 1
                    },
                    {
                        "product_id": 575,
                        "price": 147.28,
                        "quantity": 1
                    },
                    {
                        "product_id": 320,
                        "price": 994.98,
                        "quantity": 1
                    },
                    {
                        "product_id": 686,
                        "price": 721.1,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-04-28",
                "products": [
                    {
                        "product_id": 385,
                        "price": 368.28,
                        "quantity": 1
                    },
                    {
                        "product_id": 320,
                        "price": 181.71,
                        "quantity": 1
                    },
                    {
                        "product_id": 516,
                        "price": 466.06,
                        "quantity": 1
                    },
                    {
                        "product_id": 169,
                        "price": 570.83,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-08-08",
                "products": [
                    {
                        "product_id": 557,
                        "price": 300.04,
                        "quantity": 1
                    },
                    {
                        "product_id": 141,
                        "price": 909.23,
                        "quantity": 1
                    },
                    {
                        "product_id": 34,
                        "price": 211.32,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-05-05",
                "products": [
                    {
                        "product_id": 305,
                        "price": 712.25,
                        "quantity": 1
                    }
                ]
            }
        ]
    },
    {
        "customer_id": 7,
        "purchase_orders": []
    },
    {
        "customer_id": 8,
        "purchase_orders": [
            {
                "purchase_date": "2023-03-23",
                "products": [
                    {
                        "product_id": 39,
                        "price": 367.23,
                        "quantity": 1
                    },
                    {
                        "product_id": 81,
                        "price": 992.12,
                        "quantity": 1
                    },
                    {
                        "product_id": 128,
                        "price": 856.77,
                        "quantity": 1
                    },
                    {
                        "product_id": 186,
                        "price": 751.98,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-08-13",
                "products": [
                    {
                        "product_id": 617,
                        "price": 245.54,
                        "quantity": 1
                    },
                    {
                        "product_id": 112,
                        "price": 516.79,
                        "quantity": 1
                    },
                    {
                        "product_id": 171,
                        "price": 469.37,
                        "quantity": 1
                    },
                    {
                        "product_id": 579,
                        "price": 870.92,
                        "quantity": 1
                    },
                    {
                        "product_id": 155,
                        "price": 99.59,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-01-02",
                "products": [
                    {
                        "product_id": 258,
                        "price": 956.55,
                        "quantity": 1
                    },
                    {
                        "product_id": 218,
                        "price": 106.69,
                        "quantity": 1
                    },
                    {
                        "product_id": 627,
                        "price": 144.17,
                        "quantity": 1
                    },
                    {
                        "product_id": 47,
                        "price": 815.5,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-04-19",
                "products": [
                    {
                        "product_id": 561,
                        "price": 547.56,
                        "quantity": 1
                    },
                    {
                        "product_id": 381,
                        "price": 533.29,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-01-06",
                "products": [
                    {
                        "product_id": 347,
                        "price": 814.07,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-01-17",
                "products": [
                    {
                        "product_id": 239,
                        "price": 860.83,
                        "quantity": 1
                    }
                ]
            }
        ]
    },
    {
        "customer_id": 9,
        "purchase_orders": [
            {
                "purchase_date": "2024-06-11",
                "products": [
                    {
                        "product_id": 493,
                        "price": 206.15,
                        "quantity": 1
                    },
                    {
                        "product_id": 199,
                        "price": 474.66,
                        "quantity": 1
                    },
                    {
                        "product_id": 458,
                        "price": 714.42,
                        "quantity": 1
                    }
                ]
            }
        ]
    },
    {
        "customer_id": 10,
        "purchase_orders": [
            {
                "purchase_date": "2023-11-11",
                "products": [
                    {
                        "product_id": 198,
                        "price": 433.51,
                        "quantity": 1
                    },
                    {
                        "product_id": 562,
                        "price": 864.3,
                        "quantity": 1
                    },
                    {
                        "product_id": 459,
                        "price": 856.82,
                        "quantity": 1
                    },
                    {
                        "product_id": 79,
                        "price": 684.04,
                        "quantity": 1
                    },
                    {
                        "product_id": 51,
                        "price": 425.17,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-12-10",
                "products": [
                    {
                        "product_id": 159,
                        "price": 947.54,
                        "quantity": 1
                    },
                    {
                        "product_id": 435,
                        "price": 283.37,
                        "quantity": 1
                    },
                    {
                        "product_id": 320,
                        "price": 350.54,
                        "quantity": 1
                    },
                    {
                        "product_id": 509,
                        "price": 427.31,
                        "quantity": 1
                    },
                    {
                        "product_id": 25,
                        "price": 928.2,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-03-15",
                "products": [
                    {
                        "product_id": 272,
                        "price": 309.71,
                        "quantity": 1
                    },
                    {
                        "product_id": 344,
                        "price": 166.27,
                        "quantity": 1
                    },
                    {
                        "product_id": 640,
                        "price": 493.3,
                        "quantity": 1
                    },
                    {
                        "product_id": 417,
                        "price": 665.02,
                        "quantity": 1
                    },
                    {
                        "product_id": 386,
                        "price": 160.48,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-02-01",
                "products": [
                    {
                        "product_id": 177,
                        "price": 185.08,
                        "quantity": 1
                    },
                    {
                        "product_id": 268,
                        "price": 204.48,
                        "quantity": 1
                    },
                    {
                        "product_id": 492,
                        "price": 147.17,
                        "quantity": 1
                    },
                    {
                        "product_id": 218,
                        "price": 607.38,
                        "quantity": 1
                    },
                    {
                        "product_id": 498,
                        "price": 610.28,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2024-08-18",
                "products": [
                    {
                        "product_id": 559,
                        "price": 722.56,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-08-14",
                "products": [
                    {
                        "product_id": 678,
                        "price": 501.3,
                        "quantity": 1
                    },
                    {
                        "product_id": 46,
                        "price": 174.31,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-01-12",
                "products": [
                    {
                        "product_id": 67,
                        "price": 767.56,
                        "quantity": 1
                    },
                    {
                        "product_id": 595,
                        "price": 991.87,
                        "quantity": 1
                    },
                    {
                        "product_id": 257,
                        "price": 402.85,
                        "quantity": 1
                    }
                ]
            },
            {
                "purchase_date": "2023-01-08",
                "products": [
                    {
                        "product_id": 15,
                        "price": 896.79,
                        "quantity": 1
                    },
                    {
                        "product_id": 308,
                        "price": 331.21,
                        "quantity": 1
                    },
                    {
                        "product_id": 220,
                        "price": 288.61,
                        "quantity": 1
                    },
                    {
                        "product_id": 55,
                        "price": 579.02,
                        "quantity": 1
                    },
                    {
                        "product_id": 502,
                        "price": 766.14,
                        "quantity": 1
                    }
                ]
            }
        ]
    }
]