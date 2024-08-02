def sort_file_by_accuracy(file_path):
    def extract_accuracy(line):
        try:
            accuracy_part = line.split('Accuracy: ')[-1]
            accuracy_value = accuracy_part.split(',')[0]
            return float(accuracy_value)
        except ValueError:
            return 0.0
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sorted_lines = sorted(lines, key=extract_accuracy, reverse=True)
    with open(file_path, 'w') as file:
        file.writelines(sorted_lines)

file_path = 'breastmnist_forestknn_step3_v4.txt'
sort_file_by_accuracy(file_path)
