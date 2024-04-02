import os
import yaml

# Load the config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

def check_string_in_file(file_path, string_to_search):
    """ Check if the given string is found in the file at file_path """
    try:
        with open(file_path, 'r') as file:
            # Read all lines in the file one by one
            for line in file:
                # For each line, check if string_to_search is present
                if string_to_search in line:
                    return True  # String found
            return False  # String not found
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Path to the file to check
file_path_to_check = config['log_path']
string_to_search_for = 'Bolt app is running'

# Call the function and print the result
is_present = check_string_in_file(file_path_to_check, string_to_search_for)
print(f"String found: {is_present}")
