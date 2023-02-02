import os
import velocity
from multiprocessing import Process, Queue


class MatrixFile:
    def __init__(self, file_path, matrix):
        self.file_path = file_path
        self.matrix = matrix


def read(file_path, val_type):
    rows = []
    # 初始化64x64矩阵
    for k in range(64):
        rows.append(velocity.get_init_row(8, 0))
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            cols = line.split(",")
            grid = int(cols[0])
            rows[grid - 1] = list(map(val_type, cols[1:]))
    return rows


def mapping(file_name):
    results = read(f"PDF\\{file_name.replace(' ', '')}", float)
    predicts = read(f"predict\\{file_name}", int)
    adj_matrix = []
    for idx, row in enumerate(predicts):
        marked = velocity.get_init_row(8, True)
        matrix_row = velocity.get_init_row(8, float(0))
        result_row = results[idx]
        for j, val in enumerate(row):
            if val == 0:
                break
            if marked[val - 1]:
                matrix_row[val - 1] = result_row[j]
                marked[val - 1] = False
        adj_matrix.append(matrix_row)
    return adj_matrix


def mapper(file_queue, res_queue, name):
    print(f"[{name}]开始映射")
    while not file_queue.empty():
        file_name = file_queue.get()
        seq = file_name[len("result") + 1: file_name.rindex(".")]
        file_path = f"adj\\adj_{seq}.csv"
        res_queue.put(MatrixFile(file_path, mapping(file_name)))
        print(f"[{name}]: [{file_name}]处理完成")
    print(f"[{name}]映射结束")


def matrix_writer(res_queue, name):
    print(f"[{name}]开始写数据")
    while True:
        matrix_file = res_queue.get()
        if matrix_file is None:
            print(f"[{name}]数据写入完成")
            break
        else:
            with open(matrix_file.file_path, 'w') as f:
                for row in matrix_file.matrix:
                    f.write(",".join(list(map(str, row))) + "\n")


if __name__ == '__main__':
    files = os.listdir("predict")
    files.sort(key=lambda fn: int(fn[len("result") + 1:fn.index(".")]))
    q = Queue(len(files))
    res_q = Queue(len(files))
    for file in files:
        q.put(file)
    ps = []
    for i in range(5):
        p = Process(target=mapper, args=(q, res_q, f"mapper-{i}"))
        p.start()
        ps.append(p)
    process = Process(target=matrix_writer, args=(res_q, "writer"))
    process.start()
    for p in ps:
        p.join()
    res_q.put(None)
    process.join()
