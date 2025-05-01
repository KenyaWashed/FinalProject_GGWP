import csv
import os

def split_csv(file_path, rows_per_file=1000):
    """
    Tách file CSV thành nhiều file nhỏ hơn theo số dòng.
    file_path: đường dẫn tương đối đến file CSV
    rows_per_file: số dòng tối đa cho mỗi file (không tính header)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # thư mục hiện tại
    
    full_path = os.path.join(base_dir, file_path)


    file_base_name = os.path.splitext(file_path)[0]

    with open(full_path, mode='r', encoding='utf-8', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        rows = []
        file_count = 1
        for i, row in enumerate(reader, 1):
            rows.append(row)
            if i % rows_per_file == 0:
                output_file = f"{file_base_name}_part{file_count}.csv"
                with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    writer.writerows(rows)
                print(f"✅ Đã tạo: {output_file} với {len(rows)} dòng")
                rows = []
                file_count += 1

        # ghi phần còn lại
        if rows:
            output_file = f"{file_base_name}_part{file_count}.csv"
            with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"✅ Đã tạo: {output_file} với {len(rows)} dòng")

# Ví dụ sử dụng (đường dẫn tương đối, file nằm cùng thư mục)
split_csv("movies.csv", rows_per_file=100000)
