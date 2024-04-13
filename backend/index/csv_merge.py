import csv

csv.field_size_limit(10000000)
# Функция для чтения CSV файла и возвращения данных в виде списка списков
def read_csv(file_name):
    data = []
    with open(file_name, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)  # Пропускаем заголовок
        for row in reader:
            data.append(row)
    return data, header

# Функция для объединения данных из двух CSV файлов
def merge_csv(file1, file2, output_file):
    data1, header = read_csv(file1)
    data2, _ = read_csv(file2)

    # Объединяем данные
    merged_data = data1 + data2

    # Записываем объединенные данные в новый CSV файл
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(merged_data)

    print("Файлы успешно объединены в", output_file)