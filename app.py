import os
import re
import streamlit as st
from collections import Counter
from PIL import Image
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import spacy
import requests
from model2 import generate_caption
from sentence_transformers import SentenceTransformer

# Function to extract the main actor (noun) and verb from tokenized captions
def extract_main_actor_and_verb(captions):
    main_actors = []
    verbs = []
    nlp = spacy.load("en_core_web_sm")
    for caption in captions:
        doc = nlp(caption)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                main_actors.append(token.text)
            elif token.pos_ == 'VERB':
                verbs.append(token.text)
    return main_actors, verbs

# Function for semantic search using spaCy similarity
def semantic_search_spacy(generated_caption, captions):
    nlp = spacy.load("en_core_web_md")
    similarity_scores = [nlp(generated_caption).similarity(nlp(caption)) for caption in captions]
    return similarity_scores

# Function for semantic search
def semantic_search(generated_caption, main_actors, verbs, caption_file_path, uploaded_image_name):
    # Load captions and corresponding image paths from the file
    captions = []
    image_paths = []
    with open(caption_file_path, 'r') as f:
        for line in f:
            image_path, caption = line.strip().split(',', 1)
            image_paths.append(image_path)
            captions.append(caption)

    # Combine main actors and verbs into a single query string
    query = ' '.join(main_actors + verbs)

    # Calculate sentence embeddings for the generated caption and all captions in the file
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    caption_embeddings = model.encode(captions)
    query_embedding = model.encode([query])[0]

    # Calculate cosine similarity between query and captions
    similarity_scores = np.dot(caption_embeddings, query_embedding) / (np.linalg.norm(caption_embeddings, axis=1) * np.linalg.norm(query_embedding))

    # Sort captions based on similarity scores
    sorted_indices = np.argsort(similarity_scores)[::-1]
    top_relevant_captions = []
    top_similarity_scores = []
    top_image_paths = []
    displayed_images = set()  # Set to store filenames of displayed images
    for i in sorted_indices:
        if image_paths[i] != uploaded_image_name and image_paths[i] not in displayed_images:
            top_relevant_captions.append(captions[i])
            top_similarity_scores.append(similarity_scores[i])
            top_image_paths.append(image_paths[i])
            displayed_images.add(image_paths[i])  # Add filename to set of displayed images
        if len(top_relevant_captions) == 5:  # Get top 5 relevant captions
            break

    return top_relevant_captions, top_similarity_scores, top_image_paths

# List of fallback nouns and pronouns
fallback_nouns = ['man', 'woman', 'person', 'people']
fallback_pronouns = ['he', 'she', 'they']

# Streamlit app
st.title("IMAGE-TEXT")

# Upload an image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(Image.open(os.path.join("/tmp", uploaded_file.name)), caption="Uploaded Image", use_column_width=True)

    # Generate caption for the uploaded image
    caption = generate_caption("/tmp/" + uploaded_file.name)
    st.write("Generated caption:", caption)

    # Extract main actor (noun) and verb from the generated caption
    tokenized_captions = caption.split('\n')
    main_actors, verbs = extract_main_actor_and_verb(tokenized_captions)
    
    # If no main actors or verbs are found, use fallback terms
    if not main_actors:
        main_actors = [fallback_nouns[0]]  # Use the first fallback noun
    
    if not verbs:
        verbs = [fallback_pronouns[0]]  # Use the first fallback pronoun
    
    generated_main_actor = main_actors[0]  # Select the first main actor
    generated_verb = verbs[0]  # Select the first verb
    st.write("Generated main actor:", generated_main_actor)
    st.write("Generated verb:", generated_verb)

    # Perform semantic search based on the generated main actor and verb
    caption_file_path = "/Users/sirigowrih/Desktop/Text-Image-Text/Flickr8/captions.txt"
    top_relevant_captions, top_similarity_scores, top_image_paths = semantic_search(caption, main_actors, verbs, caption_file_path, uploaded_file.name)

    st.write("Top 5 Relevant Captions and Images:")
    for i, (caption, similarity_score, image_path) in enumerate(zip(top_relevant_captions, top_similarity_scores, top_image_paths), 1):
        full_image_path = os.path.join("/Users/sirigowrih/Desktop/Text-Image-Text/Flickr8/Images/", image_path)
        st.write(f"{i}. Caption: {caption}")
        st.write(f"   Similarity Score: {similarity_score}")
        st.image(Image.open(full_image_path), caption="Matched Image", use_column_width=True)
