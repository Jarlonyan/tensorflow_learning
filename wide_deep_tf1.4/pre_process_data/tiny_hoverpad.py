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
    n = luigi.IntParameter(default=10)
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

if __name__ == '__main__':
    luigi.run()

