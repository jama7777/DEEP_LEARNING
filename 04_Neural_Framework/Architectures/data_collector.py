import urllib.request
import re

def download_and_clean(url, label):
    print(f"📥 Downloading {label}...")
    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
            
            # Basic cleanup: Remove Project Gutenberg headers/footers
            # Usually starts after '*** START OF THE PROJECT GUTENBERG EBOOK ... ***'
            # And ends before '*** END OF THE PROJECT GUTENBERG EBOOK ... ***'
            start_marker = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
            end_marker = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
            
            if start_marker and end_marker:
                text = text[start_marker.end():end_marker.start()]
            
            # Remove excessive newlines and whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except Exception as e:
        print(f"❌ Error downloading {label}: {e}")
        return ""

# URLs for Nature-themed classics
urls = {
    "Walden (Thoreau)": "https://www.gutenberg.org/ebooks/205.txt.utf-8",
    "Origin of Species (Darwin)": "https://www.gutenberg.org/ebooks/2009.txt.utf-8"
}

full_corpus = ""
for label, url in urls.items():
    content = download_and_clean(url, label)
    if content:
        full_corpus += content + " "

if full_corpus:
    output_path = "/Users/indra/Desktop/DEEP_LEARNING/04_Neural_Framework/Architectures/nature_corpus.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_corpus)
    print(f"\n✅ Success! Nature Corpus saved to {output_path}")
    print(f"📊 Total word count: {len(full_corpus.split())}")
else:
    print("❌ Failed to collect data.")
