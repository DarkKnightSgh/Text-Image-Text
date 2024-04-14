import subprocess
import streamlit as st

def run_streamlit_app(app_file, port):
    subprocess.Popen(["streamlit", "run", app_file, "--server.port", str(port)])

def main():
    st.title("TEXT-IMAGE-TEXT")
    st.markdown("---")

    # Add two buttons for TEXT-IMAGE and IMAGE-TEXT
    if st.button("TEXT-IMAGE", key="text_image_button"):
        run_streamlit_app("transformers.py", 8050)
    
    if st.button("IMAGE-TEXT", key="image_text_button"):
        run_streamlit_app("app.py", 8051)

if __name__ == "__main__":
    main()
