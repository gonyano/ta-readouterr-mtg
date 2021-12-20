
   
MYPYTHON=/home/hyano/.local/share/virtualenvs/ic4qse-Wv0pccD4/bin/python

# SHOTS=100

OUTPUTDIR=/home/hyano/local/readout-mtg/outputs

# 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000
for SHOTS in 5000 6000 7000 8000 9000 10000
do
    OUTPUTFILENAME=s${SHOTS}.txt
    echo "Running top.py --shots ${SHOTS} OUTPUT to ${OUTPUTDIR}/${OUTPUTFILENAME}"
    ${MYPYTHON} top.py --shots ${SHOTS} > ${OUTPUTDIR}/${OUTPUTFILENAME}  2>&1
done