import streamlit as st
import string
import Loader
import Text_Preprocess
from shutil import copyfile
import os
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    st.title("Text-Image Retrieval")

    # Create a session state to store the predicted description
    if "predicted_description" not in st.session_state:
        st.session_state.predicted_description = ""

    # Get the predicted description from the user or the session state
    predicted_description = st.text_area("Enter the predicted description", key="description_input", value=st.session_state.predicted_description)

    if st.button("Search"):
        # Preprocess the predicted description
        table = str.maketrans('', '', string.punctuation)
        desc = predicted_description.split()
        desc = [word.lower() for word in desc]
        desc = [word.translate(table) for word in desc]
        desc = [word for word in desc if len(word) > 1]
        desc = [word for word in desc if word.isalpha()]
        predicted_description = ' '.join(desc)

        # Update the session state with the preprocessed predicted description
        st.session_state.predicted_description = predicted_description

        if predicted_description:
            dataset_root_dir = 'D:/SUSHMITHA/6TH SEM NOTES/AIWR/Text-Image-Retrieval/archive'
            testFile = dataset_root_dir + '/Flickr_8k/Flickr_8k.testImages.txt'
            testImagesLabel = Loader.load_set(testFile)
            temp = dataset_root_dir + '/descriptions.txt'
            test_descriptions = Loader.load_clean_descriptions(temp, testImagesLabel)

            # Load the sentence transformer model
            # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2-minimal')
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # Encode the predicted description
            predicted_embedding = model.encode([predicted_description])[0]

            # Pre-compute and cache image embeddings
            desc_text = Text_Preprocess.load_text(dataset_root_dir + '/Flickr_8k/Flickr8k.token.txt')
            descriptions = Text_Preprocess.load_description(desc_text)
            image_embeddings = {}
            for img in testImagesLabel:
                captions = descriptions[img]
                image_embeddings[img] = model.encode(captions)

            matchedFiles = []
            for img in testImagesLabel:
                caption_embeddings = image_embeddings[img]
                similarities = np.dot(caption_embeddings, predicted_embedding) / (np.linalg.norm(caption_embeddings, axis=1) * np.linalg.norm(predicted_embedding))
                max_similarity = np.max(similarities)
                if max_similarity > 0.5:
                    matchedFiles.append((img, descriptions[img][np.argmax(similarities)]))

            if matchedFiles:
                st.write("Matched Images:")
                path = 'D:/SUSHMITHA/6TH SEM NOTES/AIWR/Text-Image-Retrieval/archive/Images'
                for img, caption in matchedFiles:
                    img_path = os.path.join(path, img + '.jpg')
                    try:
                        image = Image.open(img_path)
                        st.image(image, caption=caption, use_column_width=True)
                    except Exception as e:
                        st.write(f"Error opening image {img}: {e}")
            else:
                st.write("No matching images found.")
        else:
            st.warning("Please enter a description to search for matching images.")

if __name__ == "__main__":
    main()