import time
import re
import os
import sys
import json

import argparse
import matlab.engine
_CURR_DIR = os.path.dirname(os.path.realpath(__file__))


def run_cmd(cmd):
    print("cmd: %s" % cmd)
    os.system(cmd)


def get_matlab_eng(works_num=0):
    eng = matlab.engine.start_matlab()
    # eng.myCluster = eng.parcluster('local')
    # eng.delete(eng.myCluster.Jobs)
    eng.addpath(eng.genpath(_CURR_DIR))
    if works_num == 0:
        eng.parpool('local')
    else:
        eng.parpool('local', works_num)
    return eng


def eval_dir(result_dir, dataset='PIOD', eval_occ=1, matlab_eng=None, maxdist=0.0075):
    if matlab_eng is None:
        eng = get_matlab_eng()
    else:
        eng = matlab_eng
    if dataset == 'bsds500':
        eval_res = eng.EdgesEval(result_dir, dataset, 0.0075)
    elif dataset == 'nyud' or dataset == 'nyud_dd':
        print(0.011)
        eval_res = eng.EdgesEval(result_dir, dataset, 0.011)
    else:
        eval_res = eng.Evaluate(result_dir, dataset, eval_occ, maxdist)
    edge_best_t, edge_ods, edge_ois, edge_ap = eval_res[0][
        0], eval_res[0][3], eval_res[0][6], eval_res[0][7]
    ori_ods, ori_ois, ori_ap = eval_res[1][3], eval_res[1][6], eval_res[1][7]
    occ_ods, occ_ois, occ_ap = eval_res[2][3], eval_res[2][6], eval_res[2][7]
    if matlab_eng is None:
        eng.quit()
    return edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods, ori_ois, ori_ap, occ_ods, occ_ois, occ_ap


def eval_zip(tar_file, dataset='PIOD', eval_occ=1, matlab_eng=None, maxdist=0.0075):

    file_name = os.path.basename(tar_file).split('.')[0]
    suffix = os.path.basename(tar_file).split('.')[-1]
    file_dir = os.path.dirname(tar_file)
    result_dir = file_dir + '/' + file_name

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if suffix == 'zip':
        run_cmd('unzip -o %s -d %s > /dev/null' % (tar_file, result_dir))
    elif suffix == 'tar':
        run_cmd('tar -xf %s -C %s' % (tar_file, result_dir))
    else:
        raise RuntimeError("=> suffix: %s is not supported!" % suffix)

    edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods, ori_ois, ori_ap, occ_ods, occ_ois, occ_ap = eval_dir(
        result_dir, dataset, eval_occ, matlab_eng, maxdist)

    # run_cmd('tar -cf %s/%s_%s_eval.tar %s/*.txt' %
    #         (file_dir, file_name, str(maxdist), result_dir))
    run_cmd('cd %s && tar -cf %s_%s_eval.tar %s/*.txt' %
            (file_dir, file_name, str(maxdist), file_name))
    # run_cmd('rm -r %s' % result_dir)
    return edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods, ori_ois, ori_ap, occ_ods, occ_ois, occ_ap


def get_epoch(file_name):
    name_split = re.split(r'[_.]', file_name)
    epoch = int(name_split[name_split.index('epoch') + 1])
    return epoch


def get_file_name_by_suffix(dest_dir, suffix):
    file_list = []
    s_len = len(suffix)
    for file_name in os.listdir(dest_dir):
        if file_name[-s_len:] == suffix:
            file_list.append(file_name)
    file_list.sort(key=lambda x: get_epoch(x))
    return file_list


def test_main(args):
    zip_dir = args.zip_dir

    zip_file_list = get_file_name_by_suffix(
        zip_dir, '_result.zip') + get_file_name_by_suffix(zip_dir, '_result.tar')

    if args.epochs is not None and args.epochs != '':
        end_model = zip_file_list[-1]
        zip_file_list = eval(f'zip_file_list[{args.epochs}]')
        if end_model not in zip_file_list:
            zip_file_list.append(end_model)

    if len(zip_file_list) == 0:
        raise RuntimeError("=> zip file list is less! ")
    print('\n zip file list: \n', zip_file_list)

    eval_json_file = zip_dir + '/eval.json'
    eval_result = []
    if os.path.exists(eval_json_file):
        with open(eval_json_file) as f:
            eval_result = json.load(f)

    log_f = None
    format_str = '' + '{:<6}	' * 10
    format_str1 = '{:.4f}	' * 10 + '{}	{}_nyud0.011'

    eng = None

    print(format_str.format('best_t', 'B-ODS', 'B-OIS', 'B-AP', 'O-ODS',
                                        'O-OIS', 'O-AP', 'occODS', 'occOIS', 'occ_AP'))
    for file_name in zip_file_list:
        # 在 json 中查找
        eval_json_res = None
        for eval_r in eval_result:
            if file_name == eval_r['zip_file_name']:
                eval_json_res = eval_r
                break

        if eval_json_res is not None:
            edge_best_t = eval_json_res['edge_best_t']
            edge_ods, edge_ois, edge_ap = eval_json_res['edge_ods'], eval_json_res['edge_ois'], eval_json_res['edge_ap']
            ori_ods, ori_ois, ori_ap = eval_json_res['ori_ods'], eval_json_res['ori_ois'], eval_json_res['ori_ap']
            occ_ods, occ_ois, occ_ap = eval_json_res['occ_ods'], eval_json_res['occ_ois'], eval_json_res['occ_ap']

            print(format_str1.format(edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods,
                                     ori_ois, ori_ap, occ_ods, occ_ois, occ_ap, file_name, args.dataset))
            continue

        print('-' * 80)
        print('\n epochs: \n', [get_epoch(f_name) for f_name in zip_file_list])
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        start_time = time.time()
        epoch = get_epoch(file_name)
        print("Epoch: %s" % epoch)

        if eng is None:
            eng = get_matlab_eng(works_num=args.works_num)
        eval_json_res = {'zip_file_name': file_name, 'epoch': epoch}
        edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods, ori_ois, ori_ap, occ_ods, occ_ois, occ_ap = \
            eval_zip(zip_dir + '/' + file_name, dataset=args.dataset, eval_occ=args.occ, matlab_eng=eng)
        eval_json_res['edge_best_t'] = edge_best_t
        eval_json_res['edge_ods'], eval_json_res['edge_ois'], eval_json_res['edge_ap'] = edge_ods, edge_ois, edge_ap
        eval_json_res['ori_ods'], eval_json_res['ori_ois'], eval_json_res['ori_ap'] = ori_ods, ori_ois, ori_ap
        eval_json_res['occ_ods'], eval_json_res['occ_ois'], eval_json_res['occ_ap'] = occ_ods, occ_ois, occ_ap

        # 记录到 json 中
        eval_result.append(eval_json_res)
        eval_result.sort(key=lambda x: x["epoch"])
        with open(eval_json_file, 'w') as f:
            json.dump(eval_result, f)

        # 记录到 log_f 中
        if log_f is None:
            log_f = open(zip_dir + '/eval.log', 'a')
            log_f.write(time.strftime(
                "\n%Y-%m-%d %H:%M:%S\n", time.localtime()))
            log_f.write(zip_dir)
            log_f.write('	'.join([i for i in zip_dir.split('_') if i != '']))
            log_f.write('\n')
            log_f.write(
                format_str.format('best_t', 'edgODS', 'edgOIS', 'edgeAP', 'oriODS',
                                  'oriOIS', 'ori_AP', 'occODS', 'occOIS', 'occ_AP'))

        log_f.write('------	' * 11 + '\n')
        log_f.write(
            format_str1.format(edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods, ori_ois, ori_ap, occ_ods, occ_ois, occ_ap, file_name, args.dataset))
        log_f.write('\n')
        log_f.flush()

        print('time using: %f' % (time.time() - start_time))
    if eng is not None:
        eng.quit()
    if log_f is not None:
        log_f.close()


def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('--occ', type=int, default=0, help='occ')
    parser.add_argument('--zip-dir', type=str, default=None, help='zip-dir')
    parser.add_argument('--result_dir', type=str, default=None, help='result_dir')
    parser.add_argument('--zipfile', type=str, default=None, help='zipfile')
    parser.add_argument('--dataset', type=str, default='PIOD', help='dataset')
    parser.add_argument('--epochs', type=str, default='', help='epochs, examples -- 10 or 1:10 or 1:10:2')
    parser.add_argument('--maxdist', type=float, default=0.0075, help='maxdist')
    parser.add_argument('--works_num', type=int, default=0, help='works_num')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.result_dir is not None:
        print('@' * 80 + '\n')
        start_time = time.time()
        edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods, ori_ois, ori_ap, occ_ods, occ_ois, occ_ap = eval_dir(
            args.result_dir, dataset=args.dataset, eval_occ=args.occ, maxdist=args.maxdist)
        print(edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods,
              ori_ois, ori_ap, occ_ods, occ_ois, occ_ap)
        print('time using: %f' % (time.time() - start_time))
        print('@' * 80 + '\n')
    elif args.zipfile is not None:
        print('@' * 80 + '\n')
        start_time = time.time()
        edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods, ori_ois, ori_ap, occ_ods, occ_ois, occ_ap = eval_zip(
            args.zipfile, dataset=args.dataset, eval_occ=args.occ, matlab_eng=None, maxdist=args.maxdist)
        print(edge_best_t, edge_ods, edge_ois, edge_ap, ori_ods,
              ori_ois, ori_ap, occ_ods, occ_ois, occ_ap)
        print('time using: %f' % (time.time() - start_time))
        print('@' * 80 + '\n')
    else:
        test_main(args)
