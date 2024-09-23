# ----------- Task 1: Prompt engineering: reorganize the X-Ray report findings into predefined anatomical regions

# import packages
import openai
from openai import OpenAI
import json
import os
from tqdm import tqdm
import argparse


# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input_path", type=str, help='Path to the input JSON file containing reports')
parser.add_argument('-o', "--output_path", type=str, help='Path to the output JSON file to save categorized reports')
parser.add_argument('-k', "--openAI_API_key", type=str, required=True, help='Your OpenAI API key')
args = parser.parse_args()


# Set the input and output file paths 
if args.input_path:
   input_file_path = args.input_path
else:
   input_file_path = './data/annotation_quiz_all.json'

if args.output_path:
   output_file_path = args.output_path
else:
   output_file_path = './data/annotation.json' # Update existing file with categorized reports in val set


# Set your API key for OpenAI
client = OpenAI(api_key= args.openAI_API_key) # Specify your OpenAI API key here


# Function to prompt gpt-4o-mini to categorize the findings 
def categorize_findings(report):
    # Create a chat completion request using a structured prompt
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Categorize the findings of a chest X-ray report into predefined anatomical regions: bone, heart, lung, and mediastinal. 
                    If a finding does not clearly belong to these categories, classify it under 'others'. Read each sentence carefully. Determine the main anatomical focus of each sentence:
                    - If a sentence discusses any findings related to bones, categorize it under 'bone'.
                    - If it pertains to the heart, categorize it under 'heart'. 
                    - If a sentence discusses any findings related to the lungs or associated structures, categorize it under 'lung'.
                    - If it mentions any findings related to the mediastinal area, categorize it under 'mediastinal'.
                    - If a sentence does not fit any of the above categories or is ambiguous, place it under 'others'.
                    Provide the output as a JSON object with each category listing relevant sentences from the report in **plain text** without extra double quotes around the sentences. 
                    The format should be: {"bone": "", "heart": "", "lung": "", "mediastinal": "", "others": ""}.
                    """
            },
            {
                "role": "user",
                "content": report
            }
        ],
        response_format= {"type": "json_object"}
    )
    # Extract and return the model's response
    return chat_completion.choices[0].message.content


# Test code with 1 report
sample_report = "The cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size. The lungs are mildly hypoinflated but grossly clear of focal airspace disease, pneumothorax, or pleural effusion. There are mild degenerative endplate changes in the thoracic spine. There are no acute bony findings."
categorized_report = categorize_findings(sample_report)
print(categorized_report)

result = json.loads(categorized_report)
#print(result)
#print(result['lung'])


# -----------  For all reports

# Get all reports

# Read the JSON file
with open(input_file_path, 'r') as file:
    data = json.load(file)
    val_reports = data.get('val', []) # retrieve the value associated with the key 'val' from the dictionary

print("Num of Reports in Val set: ",len(val_reports))



# Categorize findings for all reports with batching

# Batch size
batch_size = 10  # set batch size
categorized_results = []

# Process the reports in batches
for i in tqdm(range(0, len(val_reports), batch_size), desc="Processing reports"):
    batch = val_reports[i:i + batch_size]  # Get the current batch of reports
    
    for report in batch:
        try:
            result = categorize_findings(report['original_report'])  # Get output from the model
            result = json.loads(result)  # Load string from JSON object
        except Exception as e:
            print(f"Error processing report {report['id']}: {e}")
            continue

        dict_results = {'id': report['id'], 'report': result, 'split': report['split']}
        categorized_results.append(dict_results)

    # Also replace the original file with the updated results
    with open(input_file_path, 'r') as file: # Read the JSON file
        data = json.load(file)
    data['val'] = categorized_results # Update the 'val' key with the categorized results

    with open(output_file_path, 'w') as file: # Write the updated JSON back to a new file
        json.dump(data, file, indent=4)
    
    print("File updated successfully.")