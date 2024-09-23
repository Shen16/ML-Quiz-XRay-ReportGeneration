import os
import json
import uuid
from PIL import Image
from typing import Any, Dict
import argparse

# Define split name
parser = argparse.ArgumentParser()
parser.add_argument('-s', "--split_data", type=str, required=True, help='Specify the split name in argument')
args = parser.parse_args()

# Assign the split name
split_name = args.split_data # train, test, val

# Define paths to the annotation file and the images folder
annotations_path = './data/annotation.json'
images_folder = './data/images'
output_folder = f'./dataset_{split_name}' # dataset for train

# Make sure the output folder exists and create it if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Define the function to convert the JSON object into a token sequence string
def json2token( obj: Any, sort_json_key: bool = True):
    """
    Convert the JSON object into a token sequence string.

    Args:
        obj (Any): The JSON object to convert, which can be a dictionary, list, or other types.
        sort_json_key (bool): Whether to sort the keys of a dictionary. Default is True.

    Returns:
        str: A string representing the token sequence extracted from the JSON object.
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                output += (
                    fr"<s_{k}>"
                    + json2token(obj[k], sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, sort_json_key) for item in obj]
        )
    else:
        obj = str(obj)
        return obj
    


# Load the annotations file from data_path
with open(annotations_path) as f: # annotation.json
    annotations = json.load(f)



# Need to convert the token back to JSON later using "llava-hf/llava-v1.6-mistral-7b-hf" processor
# Need this to process outputs laters
#from transformers import AutoProcessor
#MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
#processor = AutoProcessor.from_pretrained(MODEL_ID)


# Convert token sequence string to JSON object
import re
def token2json(tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}
        


# Generate dataset.json file and images folder from the annotations.json
def process_and_save(data_annotations, images_folder, output_folder, split= split_name):
    # Define a new output subfolder for the processed images
    new_image_folder = os.path.join(output_folder, 'images')
    if not os.path.exists(new_image_folder):
        os.makedirs(new_image_folder)

    # Initialize list to hold all JSON data
    json_data_list = []

    # Iterate through the training set
    for item in data_annotations[split]: # train, test, test
        patient_id = item['id']
        # Define path for the first image (0.png)
        image_path = os.path.join(images_folder, patient_id, '0.png')

        # Check if the image exists
        if not os.path.exists(image_path):
            continue  # Skip if the expected image is not found

        # Load the image
        image = Image.open(image_path)

        # Create a unique ID for each image
        unique_id = str(uuid.uuid4())

        # Define the new image path for saving
        new_image_path = os.path.join(new_image_folder, f"{unique_id}.png")

        # Save the image
        image.save(new_image_path)

        report_dict= item['report']
        report_json= json2token(report_dict, sort_json_key=False)

        #print(f"[INST] <image>\nGenerate Report [\INST] {target_sequence}")

        # Structure the JSON data in the LLaVA format
        json_data = {
            "id": unique_id,
            "image": f"{unique_id}.png",
            "conversations": [
                {
                    "from": "human",
                    "value": "Please describe the findings in the X-ray."
                },
                {
                    "from": "gpt",
                    "value": report_json  # Using the report as the GPT's response
                }
            ]
        }

        # Append to the list
        json_data_list.append(json_data)

    # Save the JSON data list to a file

    # create dir if not exist
    if not os.path.exists(os.path.join(output_folder, split)):
        os.makedirs(os.path.join(output_folder, split))
        
    json_output_path = os.path.join(output_folder, f'{split}/{split}_dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)



# Load the annotations
with open(annotations_path, 'r') as file:
    data_annotations = json.load(file)

# Process and save the dataset
process_and_save(data_annotations, images_folder, output_folder, split_name) # run once