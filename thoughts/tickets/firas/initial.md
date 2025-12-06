## In short
Let's just say you're a thesis jockey who's an AI expert and skilled in natural language processing and customer chatbots. You don't have to agree with what I say and plan; you have to think critically and provide the best possible advice.

So, I'm currently writing a thesis about using an AI-based chatbot to make it easier for customers to ask questions related to the things I want to focus on below:
- Order status = asks about the status of their order
- Payment = asks about available payment methods
- Product = asks about the item description, how much stock is left, or the item price
- Quantity: asks specifically about the stock level
- Description: asks about the product description
- Price: asks about the product
- Fallback = If the chatbot can't correctly recognize the intent, it will give you the option to contact the admin on WhatsApp at wa.me/6281327424953

I've built the ecommerce website and the database using the Bagisto library. My only thought was to use a text-to-SQL model to translate Natural Language, then query the database, and then display the results back to the user using Natural Language (not just SQL returns). But the problem is:
1. I don't have any training data.
2. For this thesis, there's no way I can just use the model without fine-tuning or anything else that can be written in the thesis that would make it look good.

My plan:
1. Create as many synthetic bikini models as possible in Indonesian.
2. Then fine-tune the SQL-to-text model with synthetic data from the training results, then compare the results before and after. Or maybe if you have other ideas that could improve the thesis and make it look thoughtful, or include calculations or cool comparison graphs, that would also be possible.

So, what do you think?

## Problem
I've tried creating the entire pipeline, from the chatbot model, backend, web format for the chatbot, to the UI for the admin from Bagisto. For reference. Here's an explanation of the folders and files:

- models/intent_classifier/best_model/ -> the folder where I placed the last model I trained, which I thought was okay, although the results were also quite poor.
- src/ -> the backend chatbot folder:
- chatbot.py -> the entry point for the chatbot
- query_handler -> processes queries and user input connected to the MySQL database
- respon_generator -> processes text that will be returned to the user according to the specified intent
- web/ -> the chatbot template folder
- bagisto_database_report.json -> the schema and contents of the Bagisto database as context
- bagisto_database_report.txt -> all tables from the Bagisto database

The problem is that the results I've trained using my own synthesized data are very poor and too good to be true. Here's a report from the model I trained.

Classification Report:
f1-score precision recall support

order_status 1.00 1.00 1.00 60
payment_info 0.98 1.00 0.99 60
product_info 1.00 0.98 0.99 60

accuracy 0.99 180
macro avg 0.99 0.99 0.99 180
weighted avg 0.99 0.99 0.99 180

Confusion Matrix:
[[60 0 0]
[ 0 60 0]
[ 0 1 59]]

📁 Model: ./models/intent_classifier/best_model
🎯 Value Accuracy: 100.00%
🎯 Accuracy Test: 99.44%
📊 Results: ./models/intent_classifier/test_results.csv

Imagine a 100% accuracy result and a 99.4% accuracy test result doesn't make sense. It's too good to be true. Even when I tested it with a query that clearly should have been included in the intent, for example:

User: "What about my order number 1?"

With a rich user intent, the chatbot couldn't recognize it.

On the other hand, the problem also lies in its unclear responses. For example, if a user asks, "What hat products are available?", it returns all products.

Furthermore, if a user asks for a price, like "How much is a Nike hat?", it returns not the price, but all products.

Essentially, this still doesn't solve the problem of chatbots commonly used as e-commerce assistants.

So, for now, let's focus on the notebook/skripsi_chatbot.ipynb folder. make a plan of what can be improved or even rewritten