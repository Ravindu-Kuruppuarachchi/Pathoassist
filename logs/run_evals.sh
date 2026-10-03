cd /home/ravindu/rashmika/pathoasisst
for m in stardist_he_pretrained.onnx stardist_he_pretrained_int8.xml stardist_he_ft.onnx stardist_he_ft_int8.xml; do
  ~/miniconda3/envs/pathoassist/bin/python eval/evaluate.py models/$m > logs/eval_$m.log 2>&1 &
done
wait
for m in stardist_he_pretrained.onnx stardist_he_pretrained_int8.xml stardist_he_ft.onnx stardist_he_ft_int8.xml; do tail -4 logs/eval_$m.log; done
