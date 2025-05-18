import os
from icrawler.builtin import GoogleImageCrawler

# List of search terms
SEARCH_TERMS = [ "modern sofa", "armchair", "lounge chair", "chaise lounge",
                 "residential chair", "scandinavian sofa", "industrial couch" ]

OUTPUT_DIR = "data/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_images(query, num_images=20):
    """
    Download images from icrawler based on the search query.
    
    Args:
        query (str): Search term.
        num_images (int): Number of images to download.
    """
    temp_dir = os.path.join(OUTPUT_DIR, query.replace(" ", "_"))
    os.makedirs(temp_dir, exist_ok=True)

    crawler = GoogleImageCrawler(storage={"root_dir": temp_dir})
    crawler.crawl(keyword=query, max_num=num_images)

    for idx, fname in enumerate(os.listdir(temp_dir)):
        src = os.path.join(temp_dir, fname)
        dst = os.path.join(OUTPUT_DIR, f"{query.replace(' ', '_')}_{idx}.jpg")
        os.rename(src, dst)

    # Remove the temporary directory
    os.rmdir(temp_dir)
    
if __name__ == "__main__":
    for term in SEARCH_TERMS:
        print(f"Downloading images for: {term}")
        download_images(term)