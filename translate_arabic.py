import json
import os
from groq import Groq

def translate_to_arabic(client, text):
    """
    Translate text to Arabic using Groq's Llama model
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the given text to Arabic while maintaining the original structure and meaning. Only provide the translation, no additional text."
                },
                {
                    "role": "user",
                    "content": f"Translate this to Arabic: {text}"
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,  # Low temperature for consistent translations
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error translating text: {e}")
        return None

def process_json_file(input_file, output_file, api_key):
    """
    Process JSON file and translate content to Arabic
    """
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Read input JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {input_file}")
        return
    
    # Process the data
    translated_data = []
    
    # If it's a single object, convert to list for uniform processing
    if isinstance(data, dict):
        data = [data]
    
    # Loop through each item in the JSON
    for i, item in enumerate(data):
        print(f"Processing item {i+1}/{len(data)}...")
        
        translated_item = {}
        
        # Process each key-value pair in the item
        for key, value in item.items():
            if isinstance(value, str):
                # Translate the string value
                arabic_translation = translate_to_arabic(client, value)
                if arabic_translation:
                    translated_item[f"{key}_arabic"] = arabic_translation
                    translated_item[key] = value  # Keep original
                else:
                    print(f"Failed to translate: {value}")
                    translated_item[key] = value  # Keep original if translation fails
            else:
                # Keep non-string values as is
                translated_item[key] = value
        
        translated_data.append(translated_item)
    
    # Save translated data to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        print(f"Translation completed! Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    # Configuration
    INPUT_FILE = "data\external_data.json"  # Change this to your input file name
    OUTPUT_FILE = "translated_output.json"
    
    # Get API key from environment variable or set it directly
    api_key = "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"
    
    if not api_key:
        print("Please set your GROQ_API_KEY environment variable or modify the script to include your API key")
        print("You can get your API key from: https://console.groq.com/keys")
        return
    
    # Process the file
    process_json_file(INPUT_FILE, OUTPUT_FILE, api_key)

if __name__ == "__main__":
    main()