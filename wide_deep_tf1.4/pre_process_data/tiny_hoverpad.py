#coding=utf-8
import luigi
import random
import collections

class RawMetaData(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../../data/meta_Books.json')

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

class RawReviewData(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../../data/tiny_reviews_Books.json')

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
        item_list = []
        with self.input()[0].open('r') as fin_item, self.input()[1].open('r') as fin_rev, \
            self.output().open('w') as fout:
            for line in fin_rev:
                items = line.strip().split("\t")
                user_map[items[0]].append(("\t".join(items), float(items[-1])))
                item_list.append(items[1])

            for line in fin_item:
                arr = line.strip().split("\t")
                item_map[arr[0]] = arr[1]

            for key in user_map:
                sorted_user_bh = sorted(user_map[key], key=lambda x:x[1])
                for line,t in sorted_user_bh:
                    items = line.split("\t")
                    asin = items[1]
                    j = 0
                    while True:
                        asin_neg_index = random.randint(0, len(item_list)-1)
                        asin_neg = item_list[asin_neg_index]
                        if asin_neg == asin:
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


if __name__ == '__main__':
    luigi.run()

