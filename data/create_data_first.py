import os
import json

def generate_image_paths_json(input_folder, output_json_path):
    image_paths = []
    print(f"Scanning base directory: {input_folder}")
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist!")
        return
    
    first_level_folders = os.listdir(input_folder)
    print(f"Found {len(first_level_folders)} first-level folders: {first_level_folders}")
    
    for first_level in first_level_folders:
        first_level_path = os.path.join(input_folder, first_level)
        if os.path.isdir(first_level_path):
            print(f"Processing first level folder: {first_level}")
            
            second_level_folders = os.listdir(first_level_path)
            print(f"Found {len(second_level_folders)} second-level folders in {first_level}")
            
            for second_level in second_level_folders:
                second_level_path = os.path.join(first_level_path, second_level)
                if os.path.isdir(second_level_path):
                    image_file_path = os.path.join(second_level_path, 'plain_ct.nii.gz')
                    if os.path.exists(image_file_path):
                        image_paths.append({"image": image_file_path})
                        print(f"Found image.nii.gz in: {second_level}")
                    else:
                        print(f"No image.nii.gz found in: {second_level}")

    print(f"Scan completed!")
    print(f"Total images found: {len(image_paths)}")
    
    if len(image_paths) == 0:
        print("Warning: No image.nii.gz files were found!")
        return

    json_data = {"training": image_paths}

    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        with open(output_json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
        print(f"JSON file successfully saved to: {output_json_path}")
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")

input_folder = os.getenv('INPUT_FOLDER', './dataset/new_data_pan')
output_json_path = os.getenv('OUTPUT_JSON_PATH', './data/new_data_pancreas_image.json')

print("Starting the process...")
print(f"Input folder: {input_folder}")
print(f"Output JSON path: {output_json_path}")

generate_image_paths_json(input_folder, output_json_path)