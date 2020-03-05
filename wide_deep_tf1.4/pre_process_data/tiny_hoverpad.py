#coding=utf-8
import luigi
import random
import collections
import cPickle

class RawMetaData(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../../data/meta_Books.json')

class RawReviewData(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../../data/tiny_reviews_Books.json')

class ItemInfo(luigi.Task):
    n = luigi.IntParameter(default=10)
    def requires(self):
        return RawMetaData()

    def output(self):
        return luigi.LocalTarget('item_info.data')

    def run(self):
        with self.input().open('r') as f_in, self.output().open('w') as f_out:
            for line in f_in:
                obj = eval(line)
                cat = obj['categories'][0][-1]
                print >> f_out, obj['asin'] + '\t' + cat
        #end-with

class ReviewInfo(luigi.Task):
    def requires(self):
        return RawReviewData()

    def output(self):
        return luigi.LocalTarget('review_info.data')

    def run(self):
        user_map = {}
        with self.input().open('r') as f_in, self.output().open('w') as f_out:
            for line in f_in:
                obj = eval(line)
                userID = obj["reviewerID"]
                itemID = obj["asin"]
                rating = obj["overall"]
                time = obj["unixReviewTime"]
                print>>f_out, userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time)
        #end-with

class OnlineJoiner(luigi.Task):
    def requires(self):
        return [ItemInfo(), ReviewInfo()]

    def output(self):
        return luigi.LocalTarget('instances.data')

    def run(self):
        user_map = collections.defaultdict(list)
        item_map = collections.defaultdict(list)
        reviewed_item_list = []
        with self.input()[0].open('r') as fin_item, self.input()[1].open('r') as fin_rev, \
            self.output().open('w') as fout:
            for line in fin_rev:
                items = line.strip().split("\t")
                user_map[items[0]].append(("\t".join(items), float(items[-1])))
                reviewed_item_list.append(items[1])

            for line in fin_item:
                arr = line.strip().split("\t")
                item_map[arr[0]] = arr[1]

            for key in user_map:
                sorted_user_bh = sorted(user_map[key], key=lambda x:x[1]) #对用户评论数据，按照时间来排序
                for line,t in sorted_user_bh:
                    items = line.split("\t")
                    asin = items[1]
                    j = 0
                    while True:
                        asin_neg_index = random.randint(0, len(reviewed_item_list)-1)
                        asin_neg = reviewed_item_list[asin_neg_index]
                        if asin_neg == asin: #要找跟asin不一样的
                            continue
                        items[1] = asin_neg
                        print >> fout, "0" + "\t" + "\t".join(items) + "\t" + item_map[asin_neg]
                        j += 1
                        if j==1:
                            break
                    if asin in item_map:
                        print>>fout, "1" + "\t" + line + "\t" + item_map[asin]
                    else:
                        print>>fout, "1" + "\t" + line + "\t" + "default_cat"
                #end-for
            #end-for
        #end-with

class SplitInstances(luigi.Task):
    def requires(self):
        return OnlineJoiner()

    def output(self):
        return luigi.LocalTarget('split_instances.data')

    def run(self):
        with self.input().open('r') as fin, self.output().open('w') as fout:
            user_count = collections.defaultdict(int)
            for line in fin:
                line = line.strip()
                user = line.split("\t")[1]
                user_count[user] += 1
            fin.seek(0)
            i = 0
            last_user = "A26ZDKC53OP6JD"
            for line in fin:
                line = line.strip()
                user = line.split("\t")[1]
                if user == last_user:
                    if i < user_count[user] - 2:  # 1 + negative samples
                        print>>fout, "20180118" + "\t" + line
                    else:
                        print>>fout, "20190119" + "\t" + line
                else:
                    last_user = user
                    i = 0
                    if i < user_count[user] - 2:
                        print>>fout, "20180118" + "\t" + line
                    else:
                        print>>fout, "20190119" + "\t" + line
                i += 1
        #end-with


class Aggregator(luigi.Task):
    def requires(self):
        return SplitInstances()

    def output(self):
        return [luigi.LocalTarget('train.data'), luigi.LocalTarget('test.data')]

    def run(self):
        with self.input().open('r') as fin, self.output()[0].open('w') as ftrain, self.output()[1].open('w') as ftest:
            last_user = "0"
            common_fea = ""
            line_idx = 0
            for line in fin:
                items = line.strip().split("\t")
                ds = items[0]
                clk = int(items[1])
                user = items[2]
                movie_id = items[3]
                dt = items[5]
                cat1 = items[6]

                if ds=="20180118":
                    fo = ftrain
                else:
                    fo = ftest
                if user != last_user:
                    movie_id_list = []
                    cate1_list = []
                    #print >> fo, items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + "" + "\t" + ""
                else:
                    history_clk_num = len(movie_id_list)
                    cat_str = ""
                    mid_str = ""
                    for c1 in cate1_list:
                        cat_str += c1 + ""
                    for mid in movie_id_list:
                        mid_str += mid + ""
                    if len(cat_str) > 0: cat_str = cat_str[:-1]
                    if len(mid_str) > 0: mid_str = mid_str[:-1]
                    if history_clk_num >= 1:    # 8 is the average length of user behavior
                        print >> fo, items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + mid_str + "\t" + cat_str
                last_user = user
                if clk:
                    movie_id_list.append(movie_id)
                    cate1_list.append(cat1)
                line_idx += 1
            #end-for
        #end-with

class Split(luigi.Task):
    def requires(self):
        return Aggregator()

    def output(self):
        return [luigi.LocalTarget('train_split_by_user.data'), luigi.LocalTarget('test_split_by_user.data')]

    def run(self):
        with self.input()[0].open('r') as fin, self.output()[0].open('w') as ftrain, self.output()[1].open('w') as ftest:
            while True:
                rand_int = random.randint(1, 10)
                noclk_line = fin.readline().strip()
                clk_line = fin.readline().strip()
                if noclk_line == "" or clk_line == "":
                    break
                if rand_int == 2:
                    print >> ftest, noclk_line
                    print >> ftest, clk_line
                else:
                    print >> ftrain, noclk_line
                    print >> ftrain, clk_line
            #end-while
        #end-with

class GenerateVocabulary(luigi.Task):
    def requires(self):
        return Split()

    def output(self):
        return [luigi.LocalTarget('uid_voc.pkl'), luigi.LocalTarget('mid_voc.pkl'), luigi.LocalTarget('cat_voc.pkl')]

    def run(self):
        with self.input()[0].open('r') as fin, self.output()[0].open('w') as fout1, \
            self.output()[1].open('w') as fout2, self.output()[2].open('w') as fout3:
            uid_dict = {}
            mid_dict = {}
            cat_dict = {}

            iddd = 0
            for line in fin:
                arr = line.strip("\n").split("\t")
                clk = arr[0]
                uid = arr[1]
                mid = arr[2]
                cat = arr[3]
                mid_list = arr[4]
                cat_list = arr[5]
                if uid not in uid_dict:
                    uid_dict[uid] = 0
                uid_dict[uid] += 1
                if mid not in mid_dict:
                    mid_dict[mid] = 0
                mid_dict[mid] += 1
                if cat not in cat_dict:
                    cat_dict[cat] = 0
                cat_dict[cat] += 1
                if len(mid_list) == 0:
                    continue
                for m in mid_list.split("^B"):
                    if m not in mid_dict:
                        mid_dict[m] = 0
                    mid_dict[m] += 1
                #print iddd
                iddd += 1
                for c in cat_list.split("^B"):
                    if c not in cat_dict:
                        cat_dict[c] = 0
                    cat_dict[c] += 1

            sorted_uid_dict = sorted(uid_dict.iteritems(), key=lambda x:x[1], reverse=True)
            sorted_mid_dict = sorted(mid_dict.iteritems(), key=lambda x:x[1], reverse=True)
            sorted_cat_dict = sorted(cat_dict.iteritems(), key=lambda x:x[1], reverse=True)

            uid_voc = {}
            index = 0
            for key, value in sorted_uid_dict:
                uid_voc[key] = index
                index += 1

            mid_voc = {}
            mid_voc["default_mid"] = 0
            index = 1
            for key, value in sorted_mid_dict:
                mid_voc[key] = index
                index += 1

            cat_voc = {}
            cat_voc["default_cat"] = 0
            index = 1
            for key, value in sorted_cat_dict:
                cat_voc[key] = index
                index += 1
            cPickle.dump(uid_voc, fout1)
            cPickle.dump(mid_voc, fout2)
            cPickle.dump(cat_voc, fout3)
        #end-with

if __name__ == '__main__':
    luigi.run()

