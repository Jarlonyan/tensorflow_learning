#coding=utf-8
import luigi

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
        with self.output().open('w') as f_out:
            with self.input().open('r') as f_in:
                for line in f_in:
                    obj = eval(line)
                    cat = obj['categories'][0][-1]
                    print >> f_out, obj['asin'] + '\t' + cat
        #end-with

#class SquaredNumbers(luigi.Task):
#    n = luigi.IntParameter(default=10)
#
#    def requires(self):
#        return [PrintNumbers(n=self.n)]
#
#    def output(self):
#        return luigi.LocalTarget('squares.txt')
#
#    def run(self#):
#        with self.input()[0].open() as fin, self.output().open('w') as fout:
#@           for line in fin:
#                n = int(line.strip())
#                out = n*n
#                fout.write('{}:{}\n'.format(n,out))

if __name__ == '__main__':
    luigi.run()

