import streamlit as st
import string
import Loader
import Text_Preprocess
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import os

def main():
    st.title("Text-Image Retrieval")

    if "predicted_description" not in st.session_state:
        st.session_state.predicted_description = ""

    predicted_description = st.text_area("Enter the predicted description", key="description_input", value=st.session_state.predicted_description)

    if st.button("Search"):
        table = str.maketrans('', '', string.punctuation)   #where all punctuation characters are mapped to an empty string.
        desc = predicted_description.split()
        desc = [word.lower() for word in desc]
        desc = [word.translate(table) for word in desc]
        desc = [word for word in desc if len(word) > 1]
        desc = [word for word in desc if word.isalpha()]
        predicted_description = ' '.join(desc)
        st.session_state.predicted_description = predicted_description

        if predicted_description:
            dataset_root_dir = '/Users/sirigowrih/Desktop/Text-Image-Text/Flickr_8k'
            testFile = dataset_root_dir + '/Flickr_8k.testImages.txt'
            testImagesLabel = Loader.load_set(testFile)
            

            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            predicted_embedding = model.encode([predicted_description])[0]
 
            desc_text =  Text_Preprocess.load_text(dataset_root_dir + '/Flickr8k.token.txt')
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
                    matchedFiles.append((img, max_similarity, descriptions[img][np.argmax(similarities)]))

            # Sort matched files based on similarity scores
            matchedFiles.sort(key=lambda x: x[1], reverse=True)

            # Limit results to top 5
            top_matched_files = matchedFiles[:5]

            if top_matched_files:
                st.write("Matched Images:")
                path = '/Users/sirigowrih/Desktop/Text-Image-Text/Flickr8/Images'
                for img, similarity, caption in top_matched_files:
                    img_path = os.path.join(path, img + '.jpg')
                    try:
                        image = Image.open(img_path)
                        st.image(image, caption=f"{caption} (Similarity: {similarity:.2f})", use_column_width=True)
                    except Exception as e:
                        st.write(f"Error opening image {img}: {e}")
            else:
                st.write("No matching images found.")
        else:
            st.warning("Please enter a description to search for matching images.")

if __name__ == "__main__":
    main()
