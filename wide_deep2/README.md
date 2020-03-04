
+ 环境
```
sudo pip install tensorflow==1.4
sudo pip install keras=2.2.0
```


+ 参考：
https://github.com/mouna99/dien.git


+ 数据直接用上面这个链接下面导出的数据
    + http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
    + 原始评论数据
```json
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}

reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
asin - ID of the product, e.g. 0000013714
reviewerName - name of the reviewer
helpful - helpfulness rating of the review, e.g. 2/3
reviewText - text of the review
overall - rating of the product
summary - summary of the review
unixReviewTime - time of the review (unix time)
reviewTime - time of the review (raw)
```



+ 数据信息参考：http://jmcauley.ucsd.edu/data/amazon/1

