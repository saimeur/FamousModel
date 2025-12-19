class Tokenizer:
  def __init__(self, num_merges):
    self.OOV = 0
    self.car_to_id = {}
    self.id_to_car = {}
    self.id_to_car[self.OOV] = '[u]'
    self.num_merges = num_merges
    self.merge_pair = []
    self.pair_to_id = {}

  def fit(self, corpus:str):
    id = len(self.id_to_car) + 1
    enc = ''
    for c in corpus:
      if c not in self.car_to_id:
        self.car_to_id[c] = id
        self.id_to_car[id] = c
        id += 1
    enc = self.encode(corpus)
    for _ in range(self.num_merges):
      target_pair = self.most_frequent_pair(enc)
      if target_pair is None:
        break
      else:
        enc = self.merge(enc, target_pair, id)
        self.merge_pair.append(target_pair)
        self.id_to_car[id] = self.id_to_car[target_pair[0]] + self.id_to_car[target_pair[1]]
        self.pair_to_id[tuple(target_pair)] = id
        id += 1
    
  def encode(self, text:str):
    tokens_id = []
    for c in text:
      tokens_id.append(self.car_to_id.get(c, self.OOV))

    ind_corpus = 0
    while ind_corpus < len(tokens_id)-1:
      ind_merge_pair = 0
      current_pair  = tuple(tokens_id[ind_corpus:ind_corpus+2])
      while ind_merge_pair < len(self.merge_pair) and current_pair != self.merge_pair[ind_merge_pair]:
        ind_merge_pair += 1

      # Change pair_id to new_id if pair is found   
      if ind_merge_pair != len(self.merge_pair):
        tokens_id[ind_corpus:ind_corpus+2] = [self.pair_to_id[tuple(current_pair)]]
      else:
        ind_corpus += 1
        
    return tokens_id
      
  def decode(self, ids:list[int]):
    text = ''
    for token_id in ids:
      text += self.id_to_car[token_id]
    return text

  def most_frequent_pair(self, ids:list[int]):
    n = len(ids)
    dic_f = {}
    m = 0
    pair_id = None
    for i in range(n-1):
      dic_f[tuple(ids[i:i+2])] = dic_f.get(tuple(ids[i:i+2]), 0) + 1
      if dic_f[tuple(ids[i:i+2])] > m:
        m = dic_f[tuple(ids[i:i+2])]
        pair_id = tuple(ids[i:i+2])
    return pair_id
    
  def merge(self, ids:list[int], pair, id):
    ids_merge = []
    i = 0
    while i < len(ids):
      if i< len(ids)-1 and tuple(ids[i:i+2]) == pair:
        ids_merge.append(id)
        i += 1
      else:
        ids_merge.append(ids[i])
      i += 1
    return ids_merge
      
corpus = """
A Programmer’s Introduction to Unicode
March 3, 2017 · Coding · 25 Comments

Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.

A few months ago, I got interested in Unicode and decided to spend some time learning more about it in detail. In this article, I’ll give an introduction to it from a programmer’s point of view.

I’m going to focus on the character set and what’s involved in working with strings and files of Unicode text. However, in this article I’m not going to talk about fonts, text layout/shaping/rendering, or localization in detail—those are separate issues, beyond my scope (and knowledge) here.
"""

print('0 Merge')
T = Tokenizer(0)
T.fit(corpus)
toy_sample = 'Hello everyone, it\'s a pleasure to be here in the best city in the world'
toy_en = T.encode(toy_sample)
print(f'Length of the encoded sequence : {len(toy_en)}')
toy_dec = T.decode(toy_en)
print(f'Decoded sequence : {toy_dec}')
# print(f'Orignal seq == Decodec seq : {toy_sample == toy_dec}')
print(f'Size of the vocabulary : {len(T.id_to_car)}')
print(f'Number of new token created by merging : {len(T.pair_to_id)} \n')

print('10 Merge')
T = Tokenizer(10)
T.fit(corpus)
toy_sample = 'Hello everyone, it\'s a pleasure to be here in the best city in the world'
toy_en = T.encode(toy_sample)
print(f'Length of the encoded sequence : {len(toy_en)}')
toy_dec = T.decode(toy_en)
print(f'Decoded sequence : {toy_dec}')
# print(f'Orignal seq == Decodec seq : {toy_sample == toy_dec}')
print(f'Size of the vocabulary : {len(T.id_to_car)}')
print(f'Number of new token created by merging : {len(T.pair_to_id)} \n')

print('100 Merge')
T = Tokenizer(100)
T.fit(corpus)
toy_sample = 'Hello everyone, it\'s a pleasure to be here in the best city in the world'
toy_en = T.encode(toy_sample)
print(f'Length of the encoded sequence : {len(toy_en)}')
toy_dec = T.decode(toy_en)
print(f'Decoded sequence : {toy_dec}')
# print(f'Orignal seq == Decodec seq : {toy_sample == toy_dec}')
print(f'Size of the vocabulary : {len(T.id_to_car)}')
print(f'Number of new token created by merging : {len(T.pair_to_id)} \n')

print('1000 Merge')
T = Tokenizer(1000)
T.fit(corpus)
toy_sample = 'Hello everyone, it\'s a pleasure to be here in the best city in the world'
toy_en = T.encode(toy_sample)
print(f'Length of the encoded sequence : {len(toy_en)}')
toy_dec = T.decode(toy_en)
print(f'Decoded sequence : {toy_dec}')
# print(f'Orignal seq == Decodec seq : {toy_sample == toy_dec}')
print(f'Size of the vocabulary : {len(T.id_to_car)}')
print(f'Number of new token created by merging : {len(T.pair_to_id)} \n')