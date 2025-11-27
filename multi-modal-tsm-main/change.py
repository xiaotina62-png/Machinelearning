import csv
import os

# 输入文件路径
input_file = 'H:\multi-modal-tsm-main\submission.csv'
# 临时输出文件路径
output_file = 'H:\multi-modal-tsm-main\submission_new.csv'

# 打开输入文件和输出文件
with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 读取第一行（表头）并修改
    header = next(reader)
    # 将第一列名称修改为您需要的名称
    header[0] = 'video id'  # 修改为您需要的列名
    writer.writerow(header)

    # 复制其余所有行
    for row in reader:
        writer.writerow(row)

# 替换原始文件
os.replace(output_file, input_file)
print(f"已成功修改CSV文件第一列名称为：{header[0]}")