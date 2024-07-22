def sort_file_by_accuracy(file_path):
    # Function to extract accuracy from a line
    def extract_accuracy(line):
        try:
            # Split the line by 'Accuracy: ' and take the part after it
            accuracy_part = line.split('Accuracy: ')[-1]
            # Further split by comma and take the first part, which should be the accuracy value
            accuracy_value = accuracy_part.split(',')[0]
            return float(accuracy_value)
        except ValueError:
            return 0.0  # Return a default value in case of an error

    # Read the lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the accuracy from each line and sort the lines based on accuracy
    sorted_lines = sorted(lines, key=extract_accuracy, reverse=True)
    
    # Write the sorted lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(sorted_lines)

# Specify the path to your file
file_path = 'breastmnist_forestknn_step3_v4.txt'

# Call the function to sort the file by accuracy
sort_file_by_accuracy(file_path)
