import os
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Easy video feature extractor')
parser.add_argument("-input_file_list", type=str, default='sample_video_extract_list.csv', help="Should be a csv file of a single columns, each row is the input video path.")
parser.add_argument("-target_fold", type=str, default='./sample_audio/', help="The place to store the video frames.")
parser.add_argument("-part", type=int, default=1, help="Which fifth of the list to process")
parser.add_argument("-dataset", type=str, default='VGG', help="Which dataset")

args = parser.parse_args()

input_filelist = np.loadtxt(args.input_file_list, delimiter=',', dtype=str)
if os.path.exists(args.target_fold) == False:
    os.makedirs(args.target_fold)

part = args.part
print('Part ',part)
n =len(input_filelist)

input_filelist = input_filelist[(part-1)*int(n/5):part*int(n/5)]

if args.dataset == 'AS':
    AS = pd.read_csv('/tsi/dcase/AudioSet/metadata.csv')
    AS =AS.dropna()
    AS = AS.drop_duplicates(subset='file_id')
else:
    AS = pd.read_csv('/tsi/dcase/VGGSound/metadata.csv')
    AS =AS.dropna()
    AS = AS.drop_duplicates(subset='file_id')

# first resample audio
'''for i in range(input_filelist.shape[0]):
    try:
        input_f = input_filelist[i]
        ext_len = len(input_f.split('/')[-1].split('.')[-1])
        video_id = input_f.split('/')[-1][:-ext_len-1]
        if args.dataset == 'AS':
            rel_path = input_f.split('/')[-4:]
        else:
            rel_path = input_f.split('/')[-2:]
        rel_path = os.path.join(*rel_path)
        video_id = AS[AS['video_file_path']==rel_path]['file_id'].values[0]
        video_id = str(video_id)
        #video_id = extract_id(video_id)

        #id = AS[]
        output_f_1 = args.target_fold + '/' + video_id + '_intermediate.wav'
        os.system('ffmpeg -i {:s} -vn -ar 16000 {:s}'.format(input_f, output_f_1)) # save an intermediate file
    except:
        print('\n \n ERROR WITH FILE \n \n',input_f)'''

# then extract the first channel
for i in range(input_filelist.shape[0]):
    try:
        input_f = input_filelist[i]
        ext_len = len(input_f.split('/')[-1].split('.')[-1])
        video_id = input_f.split('/')[-1][:-ext_len-1]
        if args.dataset == 'AS':
            rel_path = input_f.split('/')[-4:]
        else:
            rel_path = input_f.split('/')[-2:]    
        rel_path = os.path.join(*rel_path)
        video_id = AS[AS['video_file_path']==rel_path]['file_id'].values[0]
        video_id = str(video_id)

        output_f_1 = args.target_fold + '/' + video_id + '_intermediate.wav'
        output_f_2 = args.target_fold + '/' + video_id + '.wav'
        os.system('sox {:s} {:s} remix 1'.format(output_f_1, output_f_2))
        # remove the intermediate file
        os.remove(output_f_1)
    except:
        print('\n \n ERROR WITH FILE \n \n',input_f)