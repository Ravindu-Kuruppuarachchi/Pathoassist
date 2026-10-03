cd /home/ravindu/rashmika/pathoasisst
P=~/miniconda3/envs/pathoassist/bin/python
$P eval/evaluate_ihc_deepliif.py models/stardist_he_ihc_ft_int8.xml 2>&1 | grep -v Warn | tail -5
$P eval/evaluate.py models/stardist_he_ihc_ft_int8.xml > logs/eval_he_ihc_ft.log 2>&1; tail -3 logs/eval_he_ihc_ft.log
