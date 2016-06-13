# source domain prefix path
# trn set: .trn.libsvm
# val set: .val.libsvm
# tst set: .tst.libsvm
#  codeHome = '/usr0/home/wchang2/research/NIPS2016/'
codeHome = '/Users/crickwu/Work/Research Transfer Learning/code/mnist/view1_view2/digit9/fold0/'

#srcPath = '/usr0/home/wchang2/research/NIPS2016/data/binary/src.100'
#srcPath = '/usr0/home/wchang2/research/NIPS2016/CLS_data/cls-acl10-postprocess/en_books_de_books/src.1024'
#  srcPath = '/Users/crickwu/Work/Research Transfer Learning/code/cls/en_books_de_books/src.1024'
#  srcPath = '/Users/crickwu/Work/Research Transfer Learning/code/mnist/view1_view2/digit9/fold0/src'
srcPath = codeHome + 'h10.src'

# target domain prefix path
#tgtPath = '/usr0/home/wchang2/research/NIPS2016/data/binary/tgt.40'
#tgtPath = '/usr0/home/wchang2/research/NIPS2016/CLS_data/cls-acl10-postprocess/en_books_de_books/tgt.16'
#  tgtPath = '/Users/crickwu/Work/Research Transfer Learning/code/cls/en_books_de_books/tgt.16'
#  tgtPath = '/Users/crickwu/Work/Research Transfer Learning/code/mnist/view1_view2/digit9/fold0/tgt'
tgtPath = codeHome + 'h10.tgt.only'

# parallel data prefix path
# prlPath = '/usr0/home/wchang2/research/NIPS2016/CLS_data/cls-acl10-postprocess/en_books_de_books/'
#  prlPath = '/Users/crickwu/Work/Research Transfer Learning/code/cls/en_books_de_books/'
#  prlPath = '/Users/crickwu/Work/Research Transfer Learning/code/mnist/view1_view2/digit9/fold0/10000'
prlPath = codeHome + 'h10'
