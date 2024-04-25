### TEXT-IMAGE-TEXT

**Algorithms for information Retrieval Project**

## **About:**

**Image-Text**

The image-to-text retrieval system  takes an image as input and passes it through the BLiP (Bidirectional Language-Image Pretraining) model, to generate a descriptive caption for the input image. Using semantic embedding techniques, the generated caption is then compared against a dataset of existing captions and their corresponding images. Through this comparison, the system retrieves the top five captions that are semantically most similar to the generated caption. Along with these similar captions, the system also retrieves the corresponding images associated with each caption. Additionally, a similarity score is calculated for each retrieved caption, indicating the degree of semantic similarity between the generated caption and the retrieved captions. 

**Text-Image**

The text-to-image retrieval system operates by receiving a description of the desired image from the user. Leveraging the pre-trained model 'all-MiniLM-L6-v2', the system processes the predicted description, alongside the preprocessed textual captions of images within the dataset, encoding their semantic meaning. Utilizing cosine similarity, the system calculates the resemblance between the descriptions and the captions, applying a threshold of 0.5. Subsequently, the system ranks the similarities, presenting them in descending order, and exhibits the top five images most closely aligned with the input description.



## Tech Used:
- Vision Transformer model
- BlipProcessor, BlipForConditionalGeneration
- Semantic Embeddings (TensorFlow)
- BLEU ,Similarity and Relevance scores
- Mini LM L6 V2 model
- Streamlit(Frontend)

## Dataset Link:
https://www.kaggle.com/datasets/adityajn105/flickr8k
(Make sure to adjust the paths accordingly while running)


## HuggingFaceModels:
1) https://huggingface.co/Salesforce/blip-image-captioning-base   - BLIP model
2) https://huggingface.co/nlpconnect/vit-gpt2-image-captioning    - ViT model
3) https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  - all-MiniLM-L6-V2


Disclaimer:
- Streamlit sometimes needs python/conda virtual environment to run properly
- Make sure to run model2.py once before running main.py


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


