import streamlit as st
# これまでに定義した関数の読み込み
from chapter01 import get_string_from_file
from chapter02 import get_words_from_file
from chapter03 import get_words, bows_to_cfs, load_aozora_corpus, get_bows,\
    add_to_corpus, get_weights, translate_bows, get_tfidfmodel_and_weights

#検索のための関数
from gensim import corpora, models

from gensim.similarities import MatrixSimilarity

def vsm_search(texts, query):
    tfidf_model, dic, text_weights = get_tfidfmodel_and_weights(texts)

    index = MatrixSimilarity(text_weights,  num_features=len(dic))

    # queryのbag-of-wordsを作成し，重みを計算
    query_bows = get_bows([query], dic)
    query_weights = get_weights(query_bows, dic, tfidf_model)

    # 類似度計算
    sims = index[query_weights[0]]

    # 類似度で降順にソート
    return sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

def get_list_from_file(file_name):
    with open(file_name, 'r', encoding='UTF-8') as f:
        return f.read().split()

#メインの検索部分
st.title("簡易文書検索システム")
with st.form(key='profile form'):
    #タイトルやテキストのコーパスたち
    texts = [get_string_from_file('data/ch04/%d.txt' % i) for i in range(10)]
    titles = get_list_from_file('data/ch04/book-titles.txt')
    #テキストボックス
    query=st.text_input("クエリ")
    #ベクトル空間の計算
    result = vsm_search(texts, query)
    #ボタン
    submit_btn=st.form_submit_button("検索")
    cancel_btn=st.form_submit_button("リセット")
    if submit_btn:
        for x in range(len(result)):
            st.text('%s %.4f' % (titles[result[x][0]], result[x][1]))