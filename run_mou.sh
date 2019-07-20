#!/bin/bash


python prepare_data.py 
python main_dnn.py
python evaluate.py 
# Calculate overall stats. 
#python evaluate.py get_stats






 	
