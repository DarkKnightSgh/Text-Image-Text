import os
import re
import streamlit as st
from collections import Counter
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from model import upload_image_and_generate_caption
from nltk.translate.bleu_score import sentence_bleu

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

# Function for semantic search
def semantic_search(generated_caption, model_dir, caption_file_path):
    # Load the SavedModel
    model = tf.saved_model.load(model_dir)

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

    # Sort captions based on BLEU scores
    sorted_indices = np.argsort(bleu_scores)[::-1][:5]
    top_captions = [captions[i] for i in sorted_indices]
    top_bleu_scores = [bleu_scores[i] for i in sorted_indices]
    top_image_paths = [image_paths[i] for i in sorted_indices]

    return top_captions, top_bleu_scores, top_image_paths

# Streamlit app
st.set_page_config(page_title="TEXT-IMAGE-TEXT - IMAGE-TEXT")
st.title("TEXT-IMAGE-TEXT - IMAGE-TEXT")

# Upload an image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(Image.open(os.path.join("/tmp", uploaded_file.name)), caption="Uploaded Image", use_column_width=True)

    # Generate caption for the uploaded image
    caption = upload_image_and_generate_caption(os.path.join("/tmp", uploaded_file.name))
    st.write("Generated caption:", caption)

    # Extract nouns from the generated caption
    tokenized_captions = caption.split('\n')
    nouns = extract_nouns(tokenized_captions)
    most_common_noun = Counter(nouns).most_common(1)[0][0]
    st.write("Most common noun:", most_common_noun)

    # Perform semantic search and display the top matching images and captions
    model_dir = "/Users/sirigowrih/Desktop/Text-Image-Text/encoder_model"  # Replace with the actual path to the extracted model directory
    caption_file_path = "/Users/sirigowrih/Desktop/Text-Image-Text/Flickr8/captions.txt"
    top_captions, top_bleu_scores, top_image_paths = semantic_search(caption, model_dir, caption_file_path)

    st.write("Top Matching Captions and Images:")
    for i, (caption, bleu_score, image_path) in enumerate(zip(top_captions, top_bleu_scores, top_image_paths)):
        full_image_path = os.path.join("/Users/sirigowrih/Desktop/Text-Image-Text/Flickr8/Images/", image_path)
        st.write(f"{i+1}. Caption: {caption}")
        st.write(f"   BLEU Score: {bleu_score}")
        st.image(Image.open(full_image_path), caption="Matched Image", use_column_width=True)

# generate the most accurate captions after searching in the corpus (flickr8) ,use better similarity score and return the top5 captions in the order from top to bottom