for ((runid=20; runid>=10; runid--))
do
     python unified_test.py --script_name anytrack --yaml_name anytrack --dataset_name RGBDT500 --threads 4 --num_gpus 4 --epoch $runid --mode parallel && echo "Command for runid $runid executed successfully"
done

for ((runid=20; runid>=10; runid--))
do
     python unified_test.py --script_name anytrack --yaml_name anytrack --dataset_name LasHeR --threads 4 --num_gpus 4 --epoch $runid --mode parallel && echo "Command for runid $runid executed successfully"
done

for ((runid=20; runid>=10; runid--))
do
    python unified_test.py --script_name anytrack --yaml_name anytrack --dataset_name VisEvent --threads 4 --num_gpus 4 --epoch $runid --mode parallel && echo "Command for runid $runid executed successfully"
done

