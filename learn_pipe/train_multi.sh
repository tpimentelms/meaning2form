####### Celex

data='celex'

is_opt=''
run_mode=''
reverse=''
for context in 'none' 'pos' 'word2vec' 'mixed'
do
    echo $context
    python learn_pipe/train${run_mode}.py --data $data --model lstm $is_opt --context $context $reverse
done

is_opt=''
run_mode='_bayesian'
reverse=''
for context in 'none' 'pos' 'word2vec' 'mixed'
do
    echo $context
    python learn_pipe/train${run_mode}.py --data $data --model lstm $is_opt --context $context $reverse
done

is_opt='--opt'
is_bayesian=''
run_mode='_cv'
reverse=''
for context in 'none' 'pos' 'word2vec' 'mixed'
do
    echo $context
    python learn_pipe/train${run_mode}.py --data $data --model lstm $is_opt --context $context $reverse
done

####### Celex reverse

data='celex'

is_opt=''
run_mode=''
reverse='--reverse'
for context in 'none' 'pos' 'word2vec' 'mixed'
do
    echo $context
    python learn_pipe/train${run_mode}.py --data $data --model lstm $is_opt --context $context $reverse
done

is_opt=''
run_mode='_bayesian'
reverse='--reverse'
for context in 'none' 'pos' 'word2vec' 'mixed'
do
    echo $context
    python learn_pipe/train${run_mode}.py --data $data --model lstm $is_opt --context $context $reverse
done

is_opt='--opt'
is_bayesian=''
run_mode='_cv'
reverse='--reverse'
for context in 'none' 'pos' 'word2vec' 'mixed'
do
    echo $context
    python learn_pipe/train${run_mode}.py --data $data --model lstm $is_opt --context $context $reverse
done

####### Northeuralex

data='northeuralex'

is_opt=''
run_mode=''
reverse=''
for context in 'none' 'pos' 'word2vec' 'mixed'
do
    echo $context
    python learn_pipe/train${run_mode}.py --data $data --model lstm $is_opt --context $context $reverse
done

is_opt=''
run_mode='_bayesian'
reverse=''
for context in 'none' 'pos' 'word2vec' 'mixed'
do
    echo $context
    python learn_pipe/train${run_mode}.py --data $data --model lstm $is_opt --context $context $reverse
done

is_opt='--opt'
is_bayesian=''
run_mode='_cv'
reverse=''
for context in 'none' 'pos' 'word2vec' 'mixed'
do
    echo $context
    python learn_pipe/train${run_mode}.py --data $data --model lstm $is_opt --context $context $reverse
done
