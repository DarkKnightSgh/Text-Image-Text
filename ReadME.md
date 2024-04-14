TEXT-IMAGE-TEXT

Algorithms for information Retrieval Project

#About:

Image-Text
Takes an image as input,this image is passed to ViT model,a caption is generated.Based on the caption generated,using semantic embedding search with bleu score comparision,the top five captions similar to the generated caption along with corresponding images are retrieved.


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


### Downloading and Loading Universal Sentence Encoder (Version 4) using TensorFlow Hub

```python
import tensorflow_hub as hub
import ssl
import certifi
import requests
import tarfile
import os

ssl_context = ssl.create_default_context(cafile=certifi.where())

ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

session = requests.Session()
session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=3))

response = session.get("https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed")
if response.status_code == 200:
    # Save the model to a temporary file
    with open("universal_sentence_encoder_4.tar.gz", "wb") as f:
        f.write(response.content)
    
    # Extract the model
    with tarfile.open("universal_sentence_encoder_4.tar.gz", "r:gz") as tar:
        tar.extractall()
    
    # Load the model
    model = hub.load(os.path.join("universal_sentence_encoder_4"))
else:
    print("Failed to download the model:", response.status_code)

#After Downloading:
- Extract the file,you would get :
  saved_model.pb
  variables/variables.data-00000-of-00001
  variables/variables.index
- add all of them to a directory and provide path for model in app.py


