import argparse
import os
from proj1_part1 import filter_test

def main():
    
    # create the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task",
                        required=True,
                        choices=['cartoon','edge', 'both'],
                        help="Either 'filter' to run image "
                             "filtering or 'hybrid' to run "
                             "hybrid image generation")
    parser.add_argument("-q", "--quantity",
                        required=True,
                        choices=['single', 'many'],
                        help="Run on single or many images")
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
                filter_test(img, args.task, args.quantity)
    

    elif args.task == 'edge':
        dir_list = os.listdir(args.image) 

        for i in range(0, len(dir_list)):
            if ".DS_Store" != dir_list[i]:
                img = args.image + "/" + dir_list[i]
                filter_test(img, args.task, args.quantity)
    

    elif args.task == 'both':
        if args.quantity == "many":
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
                            filter_test(img, args.task, args.quantity)
        else:
            filter_test(args.image, args.task, args.quantity)

    else:
        print("Unrecognized task entered")


if __name__ == '__main__':
    main()
