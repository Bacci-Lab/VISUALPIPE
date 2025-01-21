# Path to your JSON file
file_path = r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59\protocol.json"
import json

# Load the JSON data
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Initialize lists
protocol_numbers = []
protocol_names = []

# Extract protocol data
for key, value in data.items():
    if key.startswith("Protocol-"):
        # Extract the protocol number
        protocol_number = int(key.split("-")[1])
        protocol_numbers.append(protocol_number)
        protocol_name = value.split("/")[-1].replace(".json", "")
        protocol_names.append(protocol_name)
