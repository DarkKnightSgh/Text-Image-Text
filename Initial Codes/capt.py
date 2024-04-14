import os
import re
from collections import Counter
import ipywidgets as widgets
from IPython.display import display, Image as IPImage
from model import upload_image_and_generate_caption


cap=upload_image_and_generate_caption()

# Tokenize the captions in the string
tokenized_captions = cap.split('\n')

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

# Extract nouns from the tokenized captions
nouns = extract_nouns(tokenized_captions)

# Count occurrences of nouns
noun_counts = Counter(nouns)

# Function to search caption.txt for captions containing the same noun and return corresponding images
def search_captions_for_noun(noun, caption_file):
    matching_data = []
    with open(caption_file, 'r') as f:
        for line in f:
            image, caption = line.strip().split(',', 1)
            if noun in caption:
                matching_data.append((image, caption))
                if len(matching_data) >= 5:
                    break
    return matching_data

# Search caption.txt for captions containing the most common noun
most_common_noun = noun_counts.most_common(1)[0][0]
matching_data = search_captions_for_noun(most_common_noun, '/Users/sirigowrih/Desktop/Flickr8/captions.txt')

# Display images and captions using ipywidgets
output = widgets.Output()
display(output)

with output:
    for image, caption in matching_data:
        # Load and display image
        image_path = os.path.join("/Users/sirigowrih/Desktop/Flickr8/Images/", image)
        img_widget = IPImage(filename=image_path, format='jpg', width=300, height=300)
        display(img_widget)
        
        # Display caption
        caption_widget = widgets.Label(value=caption)
        display(caption_widget)

