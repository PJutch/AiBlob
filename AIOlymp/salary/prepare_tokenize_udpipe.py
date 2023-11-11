import wget
import os
import ufal.udpipe
import pandas
import corpy.udpipe

data_file = input('Data path: ')
result_file = input('Result path: ')

# print('Loading udpipe model...')
# model_url = 'https://rusvectores.org/static/models/udpipe_syntagrus.model'
#
# model_file = os.path.basename(model_url)
# if not os.path.exists(model_file):
#     model_file = wget.download(model_url)
#
# model = corpy.udpipe.Model(model_file)
# pipeline = ufal.udpipe.Pipeline(model, 'tokenize', ufal.udpipe.Pipeline.DEFAULT, ufal.udpipe.Pipeline.DEFAULT, 'conllu')


def num_replace(word):
    newtoken = 'x' * len(word)
    return newtoken


def clean_token(token, misc):
    out_token = token.strip().replace(' ', '')
    if token == 'Файл' and 'SpaceAfter=No' in misc:
        return None
    return out_token


def clean_lemma(lemma, pos):
    out_lemma = lemma.strip().replace(' ', '').replace('_', '').lower()
    if '|' in out_lemma or out_lemma.endswith('.jpg') or out_lemma.endswith('.png'):
        return None
    if pos != 'PUNCT':
        if out_lemma.startswith('«') or out_lemma.startswith('»'):
            out_lemma = ''.join(out_lemma[1:])
        if out_lemma.endswith('«') or out_lemma.endswith('»'):
            out_lemma = ''.join(out_lemma[:-1])
        if out_lemma.endswith('!') or out_lemma.endswith('?') or out_lemma.endswith(',') \
                or out_lemma.endswith('.'):
            out_lemma = ''.join(out_lemma[:-1])
    return out_lemma


n_processed = 0


def process(text, keep_pos=True, keep_punct=False):
    global n_processed
    n_processed += 1
    if n_processed % 10000 == 0:
        print(f'Processing item {n_processed}')
    return list(model.process(text))

    # entities = {'PROPN'}
    # named = False
    # memory = []
    # mem_case = None
    # mem_number = None
    # tagged_propn = []
    #
    # processed = pipeline.process(text)
    #
    # # пропускаем строки со служебной информацией:
    # content = [l for l in processed.split('\n') if not l.startswith('#')]
    #
    # # извлекаем из обработанного текста леммы, тэги и морфологические характеристики
    # tagged = [w.split('\t') for w in content if w]
    #
    # for t in tagged:
    #     if len(t) != 10:
    #         continue
    #     (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
    #     token = clean_token(token, misc)
    #     lemma = clean_lemma(lemma, pos)
    #     if not lemma or not token:
    #         continue
    #     if pos in entities:
    #         if '|' not in feats:
    #             tagged_propn.append('%s_%s' % (lemma, pos))
    #             continue
    #         morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}
    #         if 'Case' not in morph or 'Number' not in morph:
    #             tagged_propn.append('%s_%s' % (lemma, pos))
    #             continue
    #         if not named:
    #             named = True
    #             mem_case = morph['Case']
    #             mem_number = morph['Number']
    #         if morph['Case'] == mem_case and morph['Number'] == mem_number:
    #             memory.append(lemma)
    #             if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
    #                 named = False
    #                 past_lemma = '::'.join(memory)
    #                 memory = []
    #                 tagged_propn.append(past_lemma + '_PROPN ')
    #         else:
    #             named = False
    #             past_lemma = '::'.join(memory)
    #             memory = []
    #             tagged_propn.append(past_lemma + '_PROPN ')
    #             tagged_propn.append('%s_%s' % (lemma, pos))
    #     else:
    #         if not named:
    #             if pos == 'NUM' and token.isdigit():  # Заменяем числа на xxxxx той же длины
    #                 lemma = num_replace(token)
    #             tagged_propn.append('%s_%s' % (lemma, pos))
    #         else:
    #             named = False
    #             past_lemma = '::'.join(memory)
    #             memory = []
    #             tagged_propn.append(past_lemma + '_PROPN ')
    #             tagged_propn.append('%s_%s' % (lemma, pos))
    #
    # if not keep_punct:
    #     tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PUNCT']
    # if not keep_pos:
    #     tagged_propn = [word.split('_')[0] for word in tagged_propn]
    # return tagged_propn


print('Loading data...')
data = pandas.read_csv(data_file, sep='\t')

print('Preparing data...')
columns_to_drop = [
                   # 'id',
                   'unified_address_city', 'unified_address_country', 'languages_name',
                   # 'lemmaized_wo_stopwords_raw_description', 'lemmaized_wo_stopwords_raw_branded_description',
                   # 'name_clean',
                   'raw_description', 'raw_branded_description', 'name',
                   'employer_id'
                  ]
data = data.drop(columns_to_drop, axis=1)

cat_columns = ['schedule_name',  'unified_address_region', 'if_foreign_language',
               'is_branded_description', 'employment_name', ]
for cat_column in cat_columns:
    dummies = pandas.get_dummies(data[cat_column])
    dummies.columns = [cat_column + ' ' + col for col in dummies.columns]
    data = pandas.concat([data.drop(cat_column, axis=1),
                         dummies], axis=1)

data = data.assign(is_Moscow=(data['unified_address_state'] == 'Москва').values)
data.drop('unified_address_state', axis=1, inplace=True)

# print('Tokenizing data...')
# text_columns = ['name', 'employer_name', 'experience_name', 'key_skills_name',
#                 'specializations_profarea_name', 'professional_roles_name',
#                 'raw_description', 'raw_branded_description',
#                 'employer_industries']
#
# for text_column in text_columns:
#     data[text_column] = data[text_column].apply(process)

print('Saving data...')
data.to_csv(result_file)
