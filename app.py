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

# Function to extract nouns from tokenized captions
def extract_nouns(captions):
    nouns = []
    for caption in captions:
        # Tokenize caption into words
        words = re.findall(r'\b\w+\b', caption.lower())
        # Extract nouns using POS tagging (assuming you have a POS tagging library)
        # Here, I'm simply assuming that nouns are words longer than 3 characters
        nouns.extend([word for word in words if len(word) > 3])
    return nouns

# Function for semantic search using spaCy similarity
def semantic_search_spacy(generated_caption, captions):
    nlp = spacy.load("en_core_web_md")
    similarity_scores = [nlp(generated_caption).similarity(nlp(caption)) for caption in captions]
    return similarity_scores

# Function for semantic search
def semantic_search(generated_caption, caption_file_path):
    # Load captions and corresponding image paths from the file
    captions = []
    image_paths = []
    with open(caption_file_path, 'r') as f:
        for line in f:
            image_path, caption = line.strip().split(',', 1)
            image_paths.append(image_path)
            captions.append(caption)

    # Calculate BLEU scores for each caption
    bleu_scores = [sentence_bleu([generated_caption.split()], caption.split()) for caption in captions]

    # Calculate spaCy similarity scores for each caption
    similarity_scores = semantic_search_spacy(generated_caption, captions)

    # Combine BLEU scores and spaCy similarity scores to get relevance scores
    relevance_scores = np.array(bleu_scores) + np.array(similarity_scores)

    # Sort captions based on relevance scores
    sorted_indices = np.argsort(relevance_scores)[::-1][:5]  # Get indices of top 5 captions
    top_captions = [captions[i] for i in sorted_indices]
    top_bleu_scores = [bleu_scores[i] for i in sorted_indices]
    top_similarity_scores = [similarity_scores[i] for i in sorted_indices]
    top_image_paths = [image_paths[i] for i in sorted_indices]

    return top_captions, top_bleu_scores, top_similarity_scores, top_image_paths

# Streamlit appIMAGE-TEXT")
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

    # Extract nouns from the generated caption
    tokenized_captions = caption.split('\n')
    nouns = extract_nouns(tokenized_captions)
    most_common_noun = Counter(nouns).most_common(1)[0][0]
    st.write("Most common noun:", most_common_noun)

    # Perform semantic search and display the top matching images and captions
    caption_file_path = "/Users/sirigowrih/Desktop/Text-Image-Text/Flickr8/captions.txt"
    top_captions, top_bleu_scores, top_similarity_scores, top_image_paths = semantic_search(caption, caption_file_path)

    st.write("Top Matching Captions and Images:")
    for i, (caption, bleu_score, similarity_score, image_path) in enumerate(zip(top_captions, top_bleu_scores, top_similarity_scores, top_image_paths)):
        full_image_path = os.path.join("/Users/sirigowrih/Desktop/Text-Image-Text/Flickr8/Images/", image_path)
        st.write(f"{i+1}. Caption: {caption}")
        st.write(f"   BLEU Score: {bleu_score}")
        st.write(f"   Similarity Score: {similarity_score}")
        st.image(Image.open(full_image_path), caption="Matched Image", use_column_width=True)
