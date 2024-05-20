# %%
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import requests
import os
import base64
from PIL import Image
from io import BytesIO

def download_image(image_url, folder_name="downloaded_images", image_name="image.jpg"):
    os.makedirs(folder_name, exist_ok=True)
    if "https://" in image_url:
        response = requests.get(image_url)
        response.raise_for_status()  
        with open(os.path.join(folder_name, image_name), 'wb') as f:
            f.write(response.content)
    elif any(ext in image_url for ext in ['jpeg', 'jpg', 'png']):
        base64_str = image_url.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')
        image.save(os.path.join(folder_name, image_name)) 

def download_images_with_selenium(driver, keyword, folder_name="downloaded_images"):
    driver.get(f'https://www.google.com/search?hl=en&q={keyword + " food"}&tbm=isch')

    scroll_attempts = 0
    scroll_attempts = 0
    images = []
    previous_len_images = 0

    while scroll_attempts < 5:
        images = [img for img in driver.find_elements(By.CSS_SELECTOR, 'img.YQ4gaf') if int(img.get_attribute('height')) > 100]
        current_len_images = len(images)

        if current_len_images > previous_len_images:
            # Scroll to load more images
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            previous_len_images = current_len_images
            # Espera explícita para verificar que se carguen nuevas imágenes
            time.sleep(5)  # Ajusta este tiempo según sea necesario
            scroll_attempts += 1
        else:
            # Rompe el ciclo si no se encuentran nuevas imágenes después de un scroll
            break
    
    print(keyword, len(images))
    downloaded_images = 0
    count_50 = 0
    for image in images:
        url = image.get_attribute('src')
        if url != None :
            if 'gif' in url: continue
            if count_50 < 50:
                count_50 += 1
                download_image(url, folder_name=f"{folder_name}/train/"+keyword, image_name=f"{keyword}_{downloaded_images + 1}.jpg")
                downloaded_images += 1
            else:
                break

    if len(images) == 0:
        print(f"No images found for {keyword}")
        return keyword
    else: 
        return "Good"

def download_images_for_food(food):
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    driver = webdriver.Chrome(options=chrome_options)
    try:
        result = download_images_with_selenium(driver, food)
        return result
    finally:
        driver.quit()

def parallel_download_images(foods):
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(download_images_for_food, food) for food in foods]

        # Wait for all tasks to complete and get the results
        results = []
        for future in as_completed(futures):
            result = future.result()  # This will throw exceptions if any task failed
            results.append(result)
            print(results)
            if len(results) == len(foods) and not all(r == 'Good' for r in results):
                print('Next round!')
                last_clean = [i for i in results if i != 'Good']
                results = parallel_download_images(last_clean)
        return results

def read_txt(file_path):
    with open(file_path, 'r') as f:
        content = [line.strip() for line in f.readlines()]
    return content


results = parallel_download_images([read_txt('./classes/adjusted_food_names.txt')[0]])
# %% Download left images for food 176

def get_folder_names(directory_path="./DatasetFood176/train"):
    return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

food_names = read_txt('adjusted_food_names.txt')
folder_names = get_folder_names()

# Find the difference between the two lists
diff_elements = list(set(food_names) - set(folder_names))
print(len(diff_elements))

# results = parallel_download_images(diff_elements)
# %% Create Validation Set
import shutil
import os

def move_images(source_directory="./DatasetIngredients/train", target_directory="./DatasetIngredients/val", num_images=20):
    os.makedirs(target_directory, exist_ok=True)
    subdirectories = [name for name in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, name))]

    poor_classes = []
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(source_directory, subdirectory)
        dest_directory = os.path.join(target_directory, subdirectory)
        os.makedirs(dest_directory, exist_ok=True)
        images = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]
        if len(images) < 50:
            poor_classes.append(subdirectory)
        for image in images[:num_images]:
            shutil.move(os.path.join(subdirectory_path, image), dest_directory)
    return poor_classes

poor_class = move_images()
print(poor_class)
# %% Download images for ingredients
import re

with open('ingredients.txt', 'r') as f:
    lines = f.readlines()
lines = [re.sub(r'\(.*?\)', '', line.rstrip('\n'))  for line in lines]
ines = [line if line[-1] != ' ' else line[:-1] for line in lines]
ingredients = [ingredient.rstrip() for ingredient in list(set(lines))]

def get_folder_names(directory_path="./DatasetIngredients/train"):
    return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

folder_names = get_folder_names()
diff_elements = list(set(ingredients) - set(folder_names))

# %% 
results = parallel_download_images([read_txt('./classes/adjusted_food_names.txt')[0]])
