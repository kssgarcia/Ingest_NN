# %%
import os

# Define the folder path
folder_path = 'valid/labels'

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # Check if the file is a text file
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify the contents of each line
        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = '0'  # Change the first value to '0'
            modified_line = ' '.join(parts) + '\n'
            modified_lines.append(modified_line)

        # Write the modified contents back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)