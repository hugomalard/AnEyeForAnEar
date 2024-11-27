import os 
import sys 
from trainPrefixTuningAudio import train
import sys
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2",aligned_path = sys.argv[3])
