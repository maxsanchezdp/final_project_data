from feature_extraction import create_song_feats, create_train_feats
from model_train_save import execute_mts
from demo1 import execute_demo1
from music import play_song, execute_spoti

import argparse
import threading
import warnings
warnings.filterwarnings("ignore")

SONG = './Data/test_songs/classical.mp3'


def parse():
    """
    Arg_parser
    """
    parser = argparse.ArgumentParser(description="Music classification system")

    grupo = parser.add_mutually_exclusive_group()

    grupo.add_argument('-f', '--fext', help='Extracts features from training data',
                       action='store_true')
    grupo.add_argument('-t', '--train', help='Trains and save the model using training data', action='store_true')
    grupo.add_argument('-d', '--demo', help='Test model using song in path', action='store_true')
    grupo.add_argument('-s', '--spoti', help='Test model using song from spotify', action='store_true')

    parser.add_argument('-p', '--path', help='Path to local file.', type=str, default=SONG)

    return parser.parse_args()


def main():
    """
    Main: either extract features from train data, train the model or test with single song (from path or spotify)
    """
    args = parse()

    if args.fext:
        print("Extracting features from training data...")
        p = create_train_feats()
        print(f'Done! you may find the .csv with training data features at: {p}')

    if args.train:
        execute_mts()

    if args.demo:
        print('\nDEMO')
        thread = threading.Thread(target=play_song, args=(args.path,))
        thread.start()
        print('\nExtracting song features...')
        create_song_feats(args.path)
        print('\nPredicting genre...')
        execute_demo1()
        thread.join()
        print("\nDone!")

    if args.spoti:
        print('\nSPOTI-DEMO')
        while True:
            spath = execute_spoti()
            thread = threading.Thread(target=play_song, args=(spath,))
            thread.start()
            print('\nExtracting song features...')
            create_song_feats(spath)
            print('\nPredicting genre...')
            execute_demo1()
            thread.join()
            cont = input("Would you like to try another song? (y/n): ")
            if cont == "n":
                break
        print("\nDone!")



if __name__ == "__main__":
    main()








