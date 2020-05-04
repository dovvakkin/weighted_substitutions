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
    probs = {k:v for v, k in probs_list}
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
    for n, (word, sense) in enumerate(zi(row_words, row_senses)):
        word_positives = list()
        for s in senses_ind[word]:
            if (sense == s and t == 'pos') or (sense != s and t == 'neg'):
                word_positives += list([ind for ind in senses_ind[word][s] if ind != n])
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

def prepare_dfs(dfs, words, alpha):
    dfs = cut_dfs(dfs, words)
    words_ind = list(get_words_set(dfs))
    probs = np.zeros((len(dfs), dfs[0].shape[0], len(words_ind)))
    row_words = list()
    row_senses = list()
    for n, context in enumerate(dfs[0]['context']):
        row = list()
        for df_n, df in enumerate(dfs):
            probs[df_n, n] =  probs_list_to_row(df[df['context'] == context]['substs_probs'].tolist()[0], words_ind, alpha)
        row_words += dfs[0][dfs[0]['context'] == context]['word'].tolist()
        row_senses += dfs[0][dfs[0]['context'] == context]['gold_sense_id'].tolist()
#         print(row)
#         probs[n] = row
    senses_ind = get_word_senses(row_words, row_senses)
    positives = get_samples(row_words, row_senses, senses_ind, 'pos')
    negatives = get_samples(row_words, row_senses, senses_ind, 'neg')
    return probs, words_ind, positives, negatives, senses_ind
