## Installation

# pip install required packages
pip install requirements.txt
(if this does not work look at what is in the text file and do latest version manually)

# go to code folder
cd /ImageProcessing

#Run
python detect.py --weights best.pt --source [image location] --no-trace

example
python detect.py --weights best.pt --source testimages\CreepyMen.jpg --no-trace

go look in runs\detect to see results

video example
python detect.py --weights best.pt --source 0 --no-trace

if you have more than 1 device looking to take/give images this 0 might change to another number like 1 or 2