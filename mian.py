    # Load input JSON
import json
import os
import logging
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(filename='logs/translation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize M2M100 model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

def load_json(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logging.info(f"Successfully loaded JSON file: {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON file: {file_path} - {e}")
        raise

def translate_text(text, target_language):
    """Translate a single string of text to the target language using M2M100."""
    try:
        tokenizer.src_lang = "en"
        encoded_text = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(target_language))
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        logging.info(f"Successfully translated text: {text} to {target_language}")
        return translated_text
    except Exception as e:
        logging.error(f"Error translating text: {text} to {target_language} - {e}")
        raise

def translate_json(data, target_language):
    """Translate all values in a JSON object to the target language."""
    translated_data = {}
    for key, value in data.items():
        if isinstance(value, str):  # Ensure the value is a string
            translated_data[key] = translate_text(value, target_language)
        else:
            translated_data[key] = value
    return translated_data

def save_json(data, output_path):
    """Save the translated JSON data to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved translated JSON file: {output_path}")
    except Exception as e:
        logging.error(f"Error saving JSON file: {output_path} - {e}")
        raise

def translate_to_multiple_languages(input_path, output_dir, languages):
    """Translate a JSON file to multiple languages and save each translation separately."""
    # Load input JSON
    data = load_json(input_path)

    for lang in languages:
        # Translate JSON data
        translated_data = translate_json(data, lang)
        
        # Define output path
        output_path = os.path.join(output_dir, f'translated_example_{lang}.json')
        
        # Save translated JSON
        save_json(translated_data, output_path)

def main():
    """Main function to load, translate, and save the JSON file."""
    # Define paths (these could be dynamically set or passed as arguments)
    input_path = 'data/input/example.json'
    output_dir = 'data/output'
    languages = ['es', 'fr', 'de', 'zh']  # List of target languages (e.g., 'es' for Spanish, 'fr' for French)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Translate to multiple languages
    translate_to_multiple_languages(input_path, output_dir, languages)

if __name__ == "__main__":
    main()
