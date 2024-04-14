TEXT-IMAGE-TEXT

Algorithms for information Retrieval Project

About:


Tech Used:
- Vision Transformer model
- Semantic Embeddings (TensorFlow)
- BLEU scores
- Mini LM L6 V2 model
- Streamlit(Frontend)

Dataset Link:
https://www.kaggle.com/datasets/adityajn105/flickr8k
(Make sure to adjust the paths accordingly while running)

Disclaimer:Streamlit sometimes needs python virtual environment to run properly


Download EncoderModel:

import tensorflow_hub as hub
import ssl
import certifi
import requests
import tarfile
import os

# Create a custom SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Disable SSL certificate verification (optional)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Create a custom HTTP session with the SSL context
session = requests.Session()
session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=3))

# Download the Universal Sentence Encoder model
response = session.get("https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed")
if response.status_code == 200:
    # Save the model to a temporary file
    with open("universal_sentence_encoder_4.tar.gz", "wb") as f:
        f.write(response.content)
    
    # Extract the contents of the compressed file
    with tarfile.open("universal_sentence_encoder_4.tar.gz", "r:gz") as tar:
        tar.extractall()
    
    # Load the model from the extracted directory
    model = hub.load(os.path.join("universal_sentence_encoder_4"))
else:
    print("Failed to download the model:", response.status_code)
