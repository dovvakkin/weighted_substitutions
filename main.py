from pathlib import Path
import pandas as pd
import numpy as np
from time import time
import os
import regex as re
from tqdm import tqdm
import argparse


def load_substs(substs_fname, limit=None, drop_duplicates=True,
                data_name=None):
    if substs_fname.endswith('+'):
        split = substs_fname.strip('+').split('+')
        p1 = '+'.join(split[:-1])
        s = float(split[-1])
        p2 = re.sub(r'((<mask>)+)(.*?)T', r'T\3\1', p1)
        if p2 == p1:
            p2 = re.sub(r'T(.*?)((<mask>)+)', r'\2\1T', p1)
        print(f'Combining {p1} and {p2}')
        if p1 == p2:
            raise Exception('Cannot conver fname to symmetric one:', p1)
        dfinp1, dfinp2 = (load_substs_(p, limit, drop_duplicates, data_name)
                          for p in (p1, p2))
        dfinp = dfinp1.merge(dfinp2, on=['context', 'positions'], how='inner',
                             suffixes=('', '_y'))
        res = bcomb3(dfinp, nmasks=len(substs_fname.split('<mask>')) - 1, s=s)
        return res
    else:
        return load_substs_(substs_fname, limit, drop_duplicates, data_name)


def load_substs_(substs_fname, limit=None, drop_duplicates=True,
                 data_name=None):
    st = time()
    p = Path(substs_fname)
    npz_filename_to_save = None
    print(time() - st, 'Loading substs from ', p)
    if substs_fname.endswith('.npz'):
        arr_dict = np.load(substs_fname, allow_pickle=True)
        ss, pp = arr_dict['substs'], arr_dict['probs']
        print(ss.shape, ss.dtype, pp.shape, pp.dtype)
        ss, pp = [list(s) for s in ss], [list(p) for p in pp]
        substs_probs = pd.DataFrame({'substs': ss, 'probs': pp})
        substs_probs = substs_probs.apply(
            lambda r: [(p, s) for s, p in zip(r.substs, r.probs)], axis=1)
        print(substs_probs.head(3))
    else:
        substs_probs = pd.read_csv(p, index_col=0, nrows=limit)['0']

        print(time() - st, 'Eval... ', p)
        substs_probs = substs_probs.apply(pd.eval)
        print(time() - st, 'Reindexing... ', p)
        substs_probs.reset_index(inplace=True, drop=True)

        szip = substs_probs.apply(lambda l: zip(*l)).apply(list)
        res_probs, res_substs = szip.str[0].apply(list), szip.str[1].apply(
            list)
        print(type(res_probs))

        npz_filename_to_save = p.parent / (p.name.replace('.bz2', '.npz'))
        if not os.path.isfile(npz_filename_to_save):
            print('saving npz to %s' % npz_filename_to_save)
            np.savez_compressed(p.parent / (p.name.replace('.bz2', '.npz')),
                                probs=res_probs, substs=res_substs)

        # pd.DataFrame({'probs':res_probs, 'substs':res_substs}).to_csv(p.parent/(p.name.replace('.bz2', '.npz')),sep='\t')

    p_ex = p.parent / (p.name + '.input')
    if os.path.isfile(p_ex):
        print(time() - st, 'Loading examples from ', p_ex)
        dfinp = pd.read_csv(p_ex, nrows=limit)
        dfinp['positions'] = dfinp['positions'].apply(pd.eval)
        dfinp['word_at'] = dfinp.apply(
            lambda r: r.context[slice(*r.positions)], axis=1)
    else:
        assert data_name is not None, "no input file %s and no data name provided" % p_ex
        dfinp, _ = load_data(data_name)
        if npz_filename_to_save is not None:
            input_filename = npz_filename_to_save.parent / (
                    npz_filename_to_save.name + '.input')
        else:
            input_filename = p_ex
        print('saving input to %s' % input_filename)
        dfinp.to_csv(input_filename, index=False)

    dfinp['substs_probs'] = substs_probs
    if drop_duplicates:
        dfinp = dfinp.drop_duplicates('context')
    dfinp.reset_index(inplace=True)
    print(dfinp.head())
    return dfinp


def get_words_set(dfs):
    ws = set()
    for df in dfs:
        for context in df['context']:
            df1 = df[df['context'] == context]
            substs = df1['substs_probs']
            substs = list([v for k, v in substs.tolist()[0]])
            ws.update(substs)
    return ws


def probs_list_to_row(probs_list, words_ind, alpha):
    probs = {k: v for v, k in probs_list}
    row = list()
    for word in words_ind:
        if word in probs:
            row.append(probs[word])
        else:
            row.append(alpha)
    return row


def cut_dfs(dfs, words):
    return list([df[df['word'].isin(words)] for df in dfs])


def get_samples(row_words, row_senses, senses_ind, t='pos'):
    positives = list()
    for n, (word, sense) in enumerate(zip(row_words, row_senses)):
        word_positives = list()
        for s in senses_ind[word]:
            if (sense == s and t == 'pos') or (sense != s and t == 'neg'):
                word_positives += list(
                    [ind for ind in senses_ind[word][s] if ind != n])
        positives.append(word_positives)
    return positives


def get_word_senses(row_words, row_senses):
    senses_ind = dict()
    for n, (word, sense) in enumerate(zip(row_words, row_senses)):
        if word not in senses_ind:
            senses_ind[word] = {sense: [n]}
        elif sense not in senses_ind[word]:
            senses_ind[word][sense] = [n]
        else:
            senses_ind[word][sense] += [n]
    return senses_ind


def prepare_dfs(dfs, words, alpha, un_pairs):
    dfs = cut_dfs(dfs, words)
    words_ind = list(get_words_set(dfs))
    probs = np.zeros(
        (len(dfs) + len(un_pairs), dfs[0].shape[0], len(words_ind)))
    row_words = list()
    row_senses = list()
    for n, context in tqdm(enumerate(dfs[0]['context'])):
        row = list()
        for df_n, df in enumerate(dfs):
            probs[df_n, n] = probs_list_to_row(
                df[df['context'] == context]['substs_probs'].tolist()[0],
                words_ind, alpha)

        for n, pair in enumerate(un_pairs):
            probs[len(dfs) + n] = probs[pair[0]] * probs[pair[1]]

        row_words += dfs[0][dfs[0]['context'] == context]['word'].tolist()
        row_senses += dfs[0][dfs[0]['context'] == context][
            'gold_sense_id'].tolist()
    #         print(row)
    #         probs[n] = row
    senses_ind = get_word_senses(row_words, row_senses)
    positives = get_samples(row_words, row_senses, senses_ind, 'pos')
    negatives = get_samples(row_words, row_senses, senses_ind, 'neg')
    return probs, words_ind, positives, negatives, senses_ind


def get_vector(probs, ind, params):
    all_masks = probs[:, ind, :]
    return np.dot(params[np.newaxis], all_masks)


def get_triplet_loss(cur, pos, neg, alpha):
    return np.sum((cur - pos) ** 2) - np.sum((cur - neg) ** 2) + alpha


def get_triplet_gradient(probs, cur_ind, pos_ind, neg_ind, params):
    all_cur = probs[:, cur_ind, :]
    all_pos = probs[:, pos_ind, :]
    all_neg = probs[:, neg_ind, :]

    return 2 * np.sum(
        np.dot(params[np.newaxis], (all_cur - all_pos))) * np.sum(
        (all_cur - all_pos), axis=1) - \
           2 * np.sum(
        np.dot(params[np.newaxis], (all_cur - all_neg))) * np.sum(
        (all_cur - all_neg), axis=1)


def get_hard_cases(probs, cur, positives, negatives, params):
    cur_vec = get_vector(probs, cur, params)
    pos_vectors = np.array(
        [get_vector(probs, ind, params)[0] for ind in positives])
    neg_vectors = np.array(
        [get_vector(probs, ind, params)[0] for ind in negatives])

    pos_dists = np.sum((pos_vectors - cur_vec) ** 2, axis=1)
    neg_dists = np.sum((neg_vectors - cur_vec) ** 2, axis=1)
    pos_hards = np.flip(np.argsort(pos_dists))
    neg_hards = np.argsort(neg_dists)
    return list(np.array(positives)[pos_hards]), list(
        np.array(negatives)[neg_hards])


def train(probs, positives, negatives, EPOCHS, N_ANCH, alpha, cont=False,
          params=None):
    if not cont:
        params = np.random.uniform(0, 10, (probs.shape[0]))
    #         params = np.ones(probs.shape[0])
    for epoch in range(EPOCHS):
        grad = np.zeros(params.shape[0])
        loss = 0
        count = 0
        print('epoch:\t{}/{}'.format(epoch + 1, EPOCHS))
        for i in tqdm(range(probs.shape[1])):
            # pos_hards, neg_hards = get_hard_cases(probs, i, positives[i], negatives[i], params)

            # for pos, neg in zip(pos_hards[:N_ANCH], neg_hards[:N_ANCH]):
            for pos, neg in zip(positives[i][:N_ANCH], negatives[i][:N_ANCH]):
                loss += get_triplet_loss(get_vector(probs, i, params),
                                         get_vector(probs, pos, params),
                                         get_vector(probs, neg, params), 0.01)
                grad += get_triplet_gradient(probs, i, pos, neg, params)
                count += 1
        params -= alpha * (grad / count)
        # print(params)
        # print(np.sum(params))
        # print(loss)
        print('triplet_loss:\t{}'.format(loss / count))
    return params


def get_substs_probs(df):
    substs = list()
    probs = list()
    for _, i in df.iterrows():
        substs_probs = i['substs_probs']
        s = list([v for k, v in substs_probs])
        p = list([k for k, v in substs_probs])
        substs.append(s)
        probs.append(p)
    return np.array(substs), np.array(probs)


def save_subst(df, fname):
    s, p = get_substs_probs(df)
    np.savez_compressed(fname, substs=s, probs=p)
    df.positions = df.positions.apply(tuple)
    df[['context_id', 'word', 'gold_sense_id', 'predict_sense_id', 'positions',
        'context', 'word_at']].to_csv(fname + '.input', index=False)


def get_words_from_probs(probs):
    words = set()
    for i in probs:
        for v, k in i.tolist()[0]:
            words.update([k])
    return list(words)


def probs_to_matrix(probs, alpha, words, un_pairs):
    m = np.ones((len(probs) + len(un_pairs), len(words)))
    m *= alpha
    for line_num, line in enumerate(probs):
        line_words = {k: v for v, k in line.tolist()[0]}
        for n, word in enumerate(words):
            if word in line_words:
                m[line_num, n] = line_words[word]

    for n, pair in enumerate(un_pairs):
        m[len(probs) + n] = m[pair[0]] * m[pair[1]]

    return m


def unite_with_params(params, alpha, dfs, un_pairs):
    new_df = pd.DataFrame(columns=dfs[0].columns)
    for context in tqdm(dfs[0]['context']):
        _df = dfs[0][dfs[0]['context'] == context]
        new = _df.copy()
        probs = [df[df['context'] == context]['substs_probs'] for df in dfs]
        words = get_words_from_probs(probs)
        probs_matrix = probs_to_matrix(probs, alpha, words, un_pairs)
        united_probs = np.dot(params, probs_matrix)
        inds = np.flip(np.argsort(united_probs))[:600]
        res = list([(k, v) for k, v in
                    zip(united_probs[inds], np.array(words)[inds])])
        new['substs_probs'] = [res]
        new_df = pd.concat([new, new_df])
    return new_df


small_words = {'винт', 'обед', 'хвост'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reproduce', action='store_true',
                        help='reproduce report')

    parser.add_argument('-s', '--substs', nargs='+',
                        help='<Required> substitutions files paths',
                        required=True)

    parser.add_argument('-u', '--substs-union-pairs', nargs='+',
                        help='<Required> pair of substitutions'
                             ' files indexes to be united')

    parser.add_argument('-n', '--n-epochs',
                        help='<Required> n epochs to train')

    parser.add_argument('--n-samples',
                        help='<Required> n positive and negative'
                             ' samples for each context')

    parser.add_argument('-a', '--alpha',
                        help='<Required> train step')

    parser.add_argument('-o', '--output',
                        help='<Required> path to save united substitutions',
                        required=True)

    args = parser.parse_args()
    reproduce = args.reproduce
    substs = args.substs
    substs_union_pairs = args.substs_union_pairs
    if not reproduce:
        alpha = float(args.alpha)
        n_epochs = int(args.n_epochs)
        n_samples = int(args.n_samples)
    output = args.output

    if len(substs_union_pairs) % 2 != 0 or len(substs_union_pairs) > 2 * len(
            substs):
        raise ValueError('incorrect number of union pairs')

    dfs = list(map(load_substs, substs))
    substs_union_pairs = list(map(int, substs_union_pairs))
    substs_union_pairs = list(
        zip(substs_union_pairs[0::2], substs_union_pairs[1::2]))

    print('prepairing data')
    probs, words_ind, positives, negatives, senses_ind \
        = prepare_dfs(dfs,
                      small_words,
                      1e-4,
                      substs_union_pairs)

    if not reproduce:
        params = train(probs, positives, negatives, n_epochs, n_samples, alpha)
    else:
        repr_params_init = np.array(
            [7.99171959, 7.23257908, 6.52963882, 6.97519871, 1.29005091,
             8.40169683, 9.2151098, 5.24719852, 2.22550272, 2.1030637,
             6.1217757, 1.81652457, 3.80302804, 4.25486093, 5.53369063,
             6.78357457, 1.06457464, 4.6030889])
        params = train(probs, positives, negatives, 5, 5, 2, True,
                       repr_params_init)

    print('uniting substitutions')
    united = unite_with_params(params, 1e-4, dfs,
                               substs_union_pairs)

    print('saving substitutions to {}'.format(output))
    save_subst(united, output)
