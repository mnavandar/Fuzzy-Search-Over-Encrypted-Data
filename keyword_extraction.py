from nltk import tokenize
from operator import itemgetter
import math
import nltk


#stop_words = set(stopwords.words('english'))
#print(stop_words)
new_stopwords = {'+--------------+','(','<-','want','never','would',')','[]','**','issued','&',']','+','+---------------+---------------+','#','-----','+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+','+---------------+---------------+---------------+---------------+','+-----+-----+-----+-----+-----+-----+-----+-----+','[page','"','The','Re','University','+-----+-----+','+---------+','+---------------+---------------+-----------------+---------------+','\\','^','|/|','A','B','primary','||','*','V','=','BB&N','/','Need','-','_','3','HOSTs','link','Links','+---------+---------+---------+---------+', '[Page','Working','+','-','|','+-----------+','+-+','1','IMP','HOST','7','1969','RFC','Page','UCLA','Crocker'}
#stop_words.add(new_stopwords)
stpwrd = nltk.corpus.stopwords.words('english')
stpwrd.extend(new_stopwords)

def check_sent(word, sentences):
    final = [all([w in x for w in word]) for x in sentences]
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n])
    return result

def keywords(doc):
    total_words = doc.split()
    total_word_length = len(total_words)
    #print(total_word_length)

    total_sentences = tokenize.sent_tokenize(doc)
    total_sent_len = len(total_sentences)
    #print(total_sent_len)
    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.' , '')
        if each_word not in stpwrd:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x , y / int(total_word_length)) for x , y in tf_score.items())
    #print(tf_score)

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.' , '')
        if each_word not in stpwrd:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word , total_sentences)
            else:
                idf_score[each_word] = 1


    idf_score.update((x ,math.log(int(total_sent_len) / y) if y!=0 else 0 ) for x, y in idf_score.items())

    #print(idf_score)

    tf_idf_score = {key: tf_score[key] * idf_score.get(key , 0) for key in tf_score.keys()}
    #print(tf_idf_score)

    r=get_top_n(tf_idf_score, 10)
    return r


