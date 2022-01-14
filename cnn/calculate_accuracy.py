import numpy as np


def read_csv_as_dict(csv_path):
    out_dict = {}
    with open(csv_path) as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line_split = line.strip().split(',')  # split
            key = line_split[0]  # get key
            value = [int(v) for v in line_split[1:]]  # get output or gound truth
            out_dict[key] = value  # set dictionary
    return out_dict


if __name__ == '__main__':
    gt_path = 'bread_test/gt.csv'
    output_path = 'output.csv'

    gt_dict = read_csv_as_dict(gt_path)
    output_dict = read_csv_as_dict(output_path)

    top1_scores = []
    top5_scores = []
    for key, output in output_dict.items():
        gt = gt_dict[key][0]
        top1_scores.append(1.0 if gt == output[0] else 0.0)
        top5_scores.append(1.0 if gt in output[:5] else 0.0)

    print('* Top1: {:.5f}'.format(np.mean(top1_scores) * 100))
    print('* Top5: {:.5f}'.format(np.mean(top5_scores) * 100))
