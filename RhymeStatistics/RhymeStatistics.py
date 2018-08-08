import pandas as pd
from pypinyin import lazy_pinyin, Style

class rhyme_count :
    def __init__(self) :
        self.df_all = pd.read_csv("df_all.csv", names=['lyric', 'isBad'])
        self.df_all['rhyme'] = self.df_all.loc[:, 'lyric'].apply(lambda x: lazy_pinyin(x, style= Style.FINALS, strict=False))
        self.list_rhyme = self.df_all.loc[:, 'rhyme'].tolist()


    def paiyun(self, lastn) :
        list_lastn = [line[-lastn:] for line in self.list_rhyme if len(line) > lastn]
        head, tail = 0, 0
        paiyun_count = {}
        while head < len(list_lastn) :
            while tail < len(list_lastn) and list_lastn[tail] == list_lastn[head] :
                tail+=1
            key = '0' + str(tail - head) if tail - head < 10 else str(tail - head)
            paiyun_count[key] = paiyun_count.get(key, 0) + 1
            head = tail

        print([(_, paiyun_count[_]) for _ in sorted(paiyun_count.keys())])

    def gehangyun(self, lastn) :

        list_lastn = [line[-lastn:] for line in self.list_rhyme if len(line) > lastn]
        gehangyun_count = {}

        head, tail = 0, 0
        while head < len(list_lastn) :
            while tail < len(list_lastn) and list_lastn[tail] == list_lastn[head] :
                tail+=2
            key = '0' + str(int((tail - head) / 2)) if (tail - head) / 2 < 10 else str(int((tail - head) / 2))
            gehangyun_count[key] = gehangyun_count.get(key, 0) + 1
            head = tail

        head, tail = 1, 1
        while head < len(list_lastn) :
            while tail < len(list_lastn) and list_lastn[tail] == list_lastn[head] :
                tail+=2
            key = '0' + str(int((tail - head) / 2)) if (tail - head) / 2 < 10 else str(int((tail - head) / 2))
            gehangyun_count[key] = gehangyun_count.get(key, 0) + 1
            head = tail

        print([(_, gehangyun_count[_]) for _ in sorted(gehangyun_count.keys())])

    def jiaoyun(self, lastn) :
        list_lastn = [line[-lastn:] for line in self.list_rhyme if len(line) > lastn]
        jiaoyun_count = {}
        head, tail = 0, 0
        while head < len(list_lastn) :
            while tail < len(list_lastn) and list_lastn[head] == list_lastn[tail] and list_lastn[head+1] == list_lastn[tail+1] :
                tail+=2
            key = '0' + str(int((tail - head) / 2)) if (tail - head) / 2 < 10 else str(int((tail - head) / 2))
            jiaoyun_count[key] = jiaoyun_count.get(key, 0) + 1
            head = tail

        print([(_, jiaoyun_count[_]) for _ in sorted(jiaoyun_count.keys())])

    def baoyun(self, lastn) :
        list_lastn = [line[-lastn:] for line in self.list_rhyme if len(line) > lastn]
        baoyun_count = 0
        head = 0
        while head < len(list_lastn) - 3 :
            if list_lastn[head] == list_lastn[head+3] and list_lastn[head+1] == list_lastn[head+2] :
                baoyun_count+=1
            head+=1

        print(baoyun_count)


if __name__ == '__main__' :
    rc = rhyme_count()
    rc.baoyun(1)