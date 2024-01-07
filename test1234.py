import re

def find_first_roman_numeral(text):
    pattern = r'\b(?:I{1,3}|IV|V)\b'
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return None

# Example usage:
input_text = "The Roman numerals in this text are II, XV, and VII."
result = find_first_roman_numeral(input_text)

if result:
    print(f"First Roman numeral found: {result}")
else:
    print("No Roman numerals found.")