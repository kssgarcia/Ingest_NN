# %%
import os
import time
import requests
import base64
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd

# Load CSV file
data = pd.read_csv('./dataset/full_dataset.csv')
title_column = data['title']
title_column = title_column.drop_duplicates()
food_names = title_column.sample(n=10000, random_state=42)

# %%

def compress_image(image, folder_name, image_name, max_size_kb=5):
    """Compress the image to ensure it is under the specified size (in KB)."""
    quality = 85  # Initial quality setting
    image_format = 'JPEG'
    while True:
        buffer = BytesIO()
        image.save(buffer, format=image_format, quality=quality)
        size_kb = buffer.tell() / 1024
        if size_kb <= max_size_kb or quality <= 10:
            with open(os.path.join(folder_name, image_name), 'wb') as f:
                f.write(buffer.getvalue())
            break
        quality -= 5

def download_image(image_url, folder_name="downloaded_images", image_name="image.jpg"):
    """Download an image from a given URL and save it to the specified folder."""
    os.makedirs(folder_name, exist_ok=True)
    if "https://" in image_url:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    elif any(ext in image_url for ext in ['jpeg', 'jpg', 'png']):
        base64_str = image_url.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
    
    image = image.convert('RGB')
    compress_image(image, folder_name, image_name)

def download_images_with_selenium(driver, keyword, folder_name="Recipes10T"):
    """Use Selenium to download images from Google Images."""
    driver.get(f'https://www.google.com/search?hl=en&q={keyword + " food"}&tbm=isch')
    
    images = []
    previous_len_images = 0
    scroll_attempts = 0

    while scroll_attempts < 5:
        images = [img for img in driver.find_elements(By.CSS_SELECTOR, '.H8Rx8c img.YQ4gaf') if int(img.get_attribute('height')) > 100]
        current_len_images = len(images)

        if current_len_images > previous_len_images:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            previous_len_images = current_len_images
            time.sleep(5)
            scroll_attempts += 1
        else:
            break
    
    print(f"{keyword}: {len(images)} images found.")
    downloaded_images = 0

    for image in images:
        url = image.get_attribute('src')
        if url != None :
            if 'gif' in url: continue
            download_image(url, folder_name=f"{folder_name}/train/{keyword}", image_name=f"{keyword}_{downloaded_images + 1}.jpg")
            downloaded_images += 1
            if downloaded_images > 50: break

    if len(images) == 0:
        print(f"No images found for {keyword}")
        return keyword
    else:
        return "Good"

def download_images_for_food(food):
    """Download images for a given food item using a headless Chrome browser."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    try:
        result = download_images_with_selenium(driver, food)
        return result
    finally:
        driver.quit()

def parallel_download_images(foods):
    """Download images in parallel for a list of food items."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_images_for_food, food) for food in foods]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(results)

    # Retry for foods that failed to download images
    failed_downloads = [food for food, result in zip(foods, results) if result != "Good"]
    if failed_downloads:
        print('Retrying failed downloads...')
        return parallel_download_images(failed_downloads)
    return results

def read_txt(file_path):
    """Read a text file and return a list of lines."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Main execution

def move_images(source_directory="./Recipes10T/train"):
    subdirectories = [name for name in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, name))]

    poor_classes = []
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(source_directory, subdirectory)
        images = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]
        if len(images) < 10:
            poor_classes.append(subdirectory)

    return poor_classes

poor_class = move_images()
print(len(poor_class))

results = parallel_download_images(poor_class)
