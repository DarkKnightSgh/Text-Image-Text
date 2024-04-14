import os
import re
import tkinter as tk
from collections import Counter
from PIL import Image, ImageTk
from PIL import Image, ImageFilter
from model import upload_image_and_generate_caption


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

# Function to display images and captions in Tkinter window
def display_images_and_captions(matching_data):
    # Create the Tkinter window
    root = tk.Tk()
    root.title("Images and Captions")

    # Create a frame to contain the images and captions
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas to hold the frame
    canvas = tk.Canvas(frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add a scrollbar to the canvas
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)

    # Function to resize the canvas when the window size changes
    def resize_canvas(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    canvas.bind("<Configure>", resize_canvas)

    # Create another frame to contain the images and captions inside the canvas
    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)

    # Display images and captions
    for i, (image, caption) in enumerate(matching_data):
        image_path = os.path.join("/Users/sirigowrih/Desktop/AiWR Project/Flickr8/Images/", image)
        img = Image.open(image_path)
        img = img.resize((300, 300), resample=Image.LANCZOS)  # Use LANCZOS resampling filter
        photo = ImageTk.PhotoImage(img)

        # Create label for image
        image_label = tk.Label(inner_frame, image=photo)
        image_label.image = photo
        image_label.grid(row=i, column=0, padx=10, pady=10, sticky=tk.W)

        # Create label for caption
        caption_label = tk.Label(inner_frame, text=caption)
        caption_label.grid(row=i, column=1, padx=10, pady=10, sticky=tk.W)

    # Update the canvas scroll region
    canvas.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

    # Run the Tkinter event loop
    root.mainloop()

# Generate captions and get matching data
cap = upload_image_and_generate_caption()
tokenized_captions = cap.split('\n')
nouns = extract_nouns(tokenized_captions)
most_common_noun = Counter(nouns).most_common(1)[0][0]
matching_data = search_captions_for_noun(most_common_noun, '/Users/sirigowrih/Desktop/AiWR Project/Flickr8/captions.txt')

# Display images and captions in Tkinter window
display_images_and_captions(matching_data)
