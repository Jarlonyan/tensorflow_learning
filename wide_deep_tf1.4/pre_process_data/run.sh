
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
#gunzip reviews_Books.json.gz
#gunzip meta_Books.json.gz
#
#python process_data.py meta_Books.json reviews_Books_5.json
#python local_aggretor.py
#python split_by_user.py
#python generate_voc.py

#luigi --module=tiny_hoverpad ItemInfo --local-scheduler
#luigi --module=tiny_hoverpad ReviewInfo --local-scheduler
#luigi --module=tiny_hoverpad OnlineJoiner --local-scheduler
#luigi --module=tiny_hoverpad SplitInstances --local-scheduler
#luigi --module=tiny_hoverpad Aggregator --local-scheduler
luigi --module=tiny_hoverpad Split --local-scheduler

