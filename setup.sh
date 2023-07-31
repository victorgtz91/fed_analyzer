mkdir -p ~/.nltk_data/corpora
echo "Downloading NLTK corpora..."
python -m nltk.downloader stopwords -d ~/.nltk_data/corpora
echo "Done."
