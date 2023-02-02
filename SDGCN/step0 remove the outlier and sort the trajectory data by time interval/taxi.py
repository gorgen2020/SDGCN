import os
import time
from itertools import groupby
from multiprocessing import Process, Queue
from geopy.distance import geodesic


aa = geodesic((39.83, 116.25), (40.12, 116.64))
print(geodesic((39.83, 116.25), (39.83, 116.64)))
print(geodesic((39.83, 116.25), (40.12, 116.25)))




x0, y0 = 116.25, 39.83
x1, y1 = 116.64, 40.12


class GridRange:
    def __init__(self, grid_id, start_longitude, end_longitude, start_latitude, end_latitude):
        self.grid_id = str(grid_id)
        self.start_longitude = start_longitude
        self.end_longitude = end_longitude
        self.start_latitude = start_latitude
        self.end_latitude = end_latitude

    def is_in_area(self, longitude, latitude):
        return self.start_longitude <= longitude <= self.end_longitude \
               and self.start_latitude <= latitude <= self.end_latitude


class TaxiLog:
    def __init__(self, taxi_id, src_ts, src_longitude, src_latitude, dest_ts, dest_longitude, dest_latitude,
                 src_grid=None, dest_grid=None, seq=None):
        self.taxi_id = taxi_id
        self.src_ts = src_ts
        self.src_longitude = src_longitude
        self.src_latitude = src_latitude
        self.dest_ts = dest_ts
        self.dest_longitude = dest_longitude
        self.dest_latitude = dest_latitude
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.seq = seq

    def get_distance(self):
        return geodesic((self.src_latitude, self.src_longitude), (self.dest_latitude, self.dest_longitude)).km

    def get_avg_velocity(self):
        cost_time = (self.dest_ts - self.src_ts) / 3600
        return round(self.get_distance() / cost_time, 3)

    def to_line(self):
        columns = [self.src_grid, str(self.src_ts), self.src_grid, str(self.src_longitude), str(self.src_latitude),
                   self.src_grid,
                   self.dest_grid, str(self.dest_ts), self.dest_grid, str(self.dest_longitude), str(self.dest_latitude),
                   self.dest_grid
                   ]
        return ",".join(columns)


def get_grids(t="object"):
    x0, y0 = 116.25, 39.83
    x1, y1 = 116.64, 40.12
    rows, cols = 8, 8
    size_x, size_y = (x1 - x0) / rows, (y1 - y0) / cols

    grids = []
    index = 1
    for r in range(rows):
        for c in range(cols):
            _x0, _y0 = c * size_x + x0, (rows - r - 1) * size_y + y0
            _x1, _y1 = (c + 1) * size_x + x0, (rows - r) * size_y + y0
            if t == "object":
                grids.append(GridRange(index, _x0, _x1, _y0, _y1))
                index += 1
            else:
                grids += [[[_y1, _x0], [_y1, _x1], [_y0, _x1], [_y0, _x0]]]
    return grids


def write_grids(file_path):
    """
    生成grids.txt
    """
    grids = get_grids("array")
    with open(file_path, 'w') as f:
        f.write(str(grids))


def to_ts(time_str):
    return int(time.mktime(time.strptime(time_str, "%Y-%m-%d %H:%M:%S")))


def to_taxi_log(prev_line_columns, curr_line_columns):
    src_ts = to_ts(prev_line_columns[1])
    src_long = float(prev_line_columns[2])
    src_lat = float(prev_line_columns[3])
    dest_ts = to_ts(curr_line_columns[1])
    dest_long = float(curr_line_columns[2])
    dest_lat = float(curr_line_columns[3])
    return TaxiLog(curr_line_columns[0], src_ts, src_long, src_lat, dest_ts, dest_long, dest_lat)


def read_taxi_log(file_path):
    taxi_logs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if len(lines) <= 1:
            return None
        prev_line_columns = lines[0].split(",")
        for i in range(1, len(lines)):
            curr_line_columns = lines[i].split(",")
            taxi_logs.append(to_taxi_log(prev_line_columns, curr_line_columns))
            prev_line_columns = curr_line_columns
    return taxi_logs


def cal_sequence(ts):
    # 从2008-02-02 00:00:00(1201881600)开始，梅半个小时为区间
    # [2008-02-02 00:00:00, 2008-02-02 00:29:59] -> 1
    # [2008-02-02 00:30:00, 2008-02-02 00:59:59] -> 2
    idx = (ts - 1201881600) // 300 + 1
    return f'sequences{idx}'


def match_grid(grids, longitude, latitude):
    for grid in grids:
        if grid.is_in_area(longitude, latitude):
            return grid.grid_id
    return None


def filter_and_fill_grid(grids, taxi_logs):
    """
    根据经纬度匹配行政区划，两个地点都需要匹配到，否则丢弃
    :param grids:
    :param taxi_logs:
    :return:
    """
    matched = []
    for taxi_log in taxi_logs:
        src_grid = match_grid(grids, taxi_log.src_longitude, taxi_log.src_latitude)
        if src_grid is None:
            continue
        dest_grid = match_grid(grids, taxi_log.dest_longitude, taxi_log.dest_latitude)
        if dest_grid is None:
            continue
        taxi_log.src_grid = src_grid
        taxi_log.dest_grid = dest_grid
        matched.append(taxi_log)
    return matched


def write(taxi_logs, directory_path):
    sorted_taxi_logs = sorted(taxi_logs, key=lambda t: cal_sequence(t.src_ts))
    ts_group = groupby(sorted_taxi_logs, key=lambda t: cal_sequence(t.src_ts))
    for key, group in ts_group:
        with open(directory_path + f"\\{key}.txt", 'a') as f:
            for taxi_log in group:
                f.write(taxi_log.to_line() + "\n")


def processor(file_queue: Queue, grids, res_queue, name):
    print(f"[{name}]开始执行")
    while not file_queue.empty():
        file_path = file_queue.get()
        taxi_logs = read_taxi_log(file_path)
        if taxi_logs is None:
            print(f"[{name}]: [{file_path}]处理完成")
            continue
        matched_logs = filter_and_fill_grid(grids, taxi_logs)
        if len(matched_logs) > 0:
            res_queue.put(matched_logs)
        print(f"[{name}]: [{file_path}]处理完成")
    print(f"[{name}]执行结束")


def writer(res_queue: Queue, directory_path):
    while True:
        taxi_logs = res_queue.get()
        if taxi_logs is None:
            print(f"数据处理完成，数据目录[{directory_path}]")
            break
        else:
            write(taxi_logs, directory_path)


if __name__ == '__main__':
    directory = "taxi_log_2008_by_id"
    files = os.listdir(directory)
    match_grids = get_grids()
    write_grids('./grid.txt')
    q = Queue(len(files))
    res_q = Queue(len(files))
    for f in files:
        q.put(directory + "\\" + f)
    ps = []
    for i in range(16):
        p = Process(target=processor, args=(q, match_grids, res_q, f"processor-{i}"))
        p.start()
        ps.append(p)
    process = Process(target=writer, args=(res_q, "taxi_data_2008"))
    process.start()
    for p in ps:
        p.join()
    res_q.put(None)
    process.join()
