

def get_num_from_string(text):
    res = 0
    for c in text:
        if '0' <= c <= '9':
            res *= 10
            res += int(c)
        if c == '/':
            res = 0

    return res


def read_file(file_directory):
    fp = open(file_directory, 'r')
    lines = fp.readlines()
    fp.close()
    return list(map(lambda s: s.strip(), lines))


def write_file(file_directory, text):
    fp = open(file_directory, 'w')
    fp.write(text)
    fp.close()


def classify_result(text):
    n_source = len(text) - 4
    delays_list = []
    for i in range(0, n_source):
        delays = list(map(lambda t: int(t) / 1000.0, text[i].split()))
        delays_list.append(delays)
    aoi_list = list(map(float, text[n_source].split()))
    paoi_list = list(map(float, text[n_source + 1].split()))
    sum_aoi = float(text[n_source + 2])
    sum_paoi = float(text[n_source + 3])

    return delays_list, aoi_list, paoi_list, sum_aoi, sum_paoi


def classify_metadata(text):
    res = []
    for s in text:
        s_list = s.split()
        data_size = get_num_from_string(s_list[0])
        data_size = data_size if data_size >= 10 else data_size * 1024
        freq = get_num_from_string(s_list[1])
        dura = get_num_from_string(s_list[2])
        is_realtime = s_list[3][0] == 'r'
        res.append((data_size, freq, dura, is_realtime))

    return res


def read_analysis_result_1d():
    text_1d = read_file('analysis.txt')
    return [list(map(float, text.split()))[1] for text in text_1d]