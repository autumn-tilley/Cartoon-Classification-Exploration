import argparse
import os
from proj1_part1 import filter_test
from proj1_part2 import hybrid_img_generation

def main():
    """
    To test your program you can run the tests we created - "filter" and "hybrid". To run each test, you must add the corresponding
    flags (outlined below) to specify which test you are running, and the image paths you are running the tests on

    Command line usage: python3 main.py -t | --test <filter or hybrid> -i | --image <image path(s) separated by comma (no spaces)>

    -t | --task - flag - required. specifies which test to run (filter - image filtering or hybrid - hybrid image generation)
    -i | --image - flag - required. specifies which image to filter or images to create a hybrid. If running hybrid should be two image
    paths separated by a comma (no spaces)

    e.g.
    python3 main.py -t filter -i ../data/dog.bmp
    python3 main.py -t hybrid -i ../data/cat.bmp,../data/dog.bmp

        """

    # create the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task",
                        required=True,
                        choices=['cartoon','edge', 'both'],
                        help="Either 'filter' to run image "
                             "filtering or 'hybrid' to run "
                             "hybrid image generation")
    parser.add_argument("-i", "--image",
                        required=True,
                        help="Paths to image(s). If running "
                             "hybrid images separate "
                             "paths by a comma (no space)")
    args = parser.parse_args()

    if args.task == 'cartoon':
        dir_list = os.listdir(args.image) 

        for i in range(0, len(dir_list)):
            if ".DS_Store" != dir_list[i]:
                img = args.image + "/" + dir_list[i]
                filter_test(img, args.task)
    

    elif args.task == 'edge':
        dir_list = os.listdir(args.image) 

        for i in range(0, len(dir_list)):
            if ".DS_Store" != dir_list[i]:
                img = args.image + "/" + dir_list[i]
                filter_test(img, args.task)
    

    elif args.task == 'both':
        dir_list = [x[0] for x in os.walk(args.image)]
        for i in range(1, len(dir_list)):
            print("folder " + str(i))
            if ".DS_Store" != dir_list[i]:
                dir_list2 = os.listdir(dir_list[i]) 
                for j in range(0, len(dir_list2)):
                    if ".DS_Store" != dir_list2[j]:
                        print(str(j) + " out of " + str(len(dir_list2)))
                        img = dir_list[i] + "/" + dir_list2[j]
                        print(img)
                        filter_test(img, args.task)


    # user didn't specify whether testing filtering or hybrid image generation
    else:
        print("You must specify what you are testing (either 'filter' or 'hybrid')"
              " for e.g. try running: \n python3 main.py -t filter -i ../data/dog.bmp")


if __name__ == '__main__':
    main()
