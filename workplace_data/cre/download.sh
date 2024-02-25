for dataset in test train dev
do
  wget https://raw.githubusercontent.com/jiangfeng1124/ChemRxnExtractor/main/tests/data/role/$dataset.txt
done
cat train.txt dev.txt test.txt > role.txt
rm train.txt dev.txt test.txt

for dataset in test train dev
do
  wget https://raw.githubusercontent.com/jiangfeng1124/ChemRxnExtractor/main/tests/data/prod/$dataset.txt
done
cat train.txt dev.txt test.txt > prod.txt
rm train.txt dev.txt test.txt
