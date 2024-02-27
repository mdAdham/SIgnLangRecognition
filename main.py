import os
from app import main

if __name__ == '__main__':
    #main()

    videos_dir = os.path.join("D:\\__yakku\\smart intern\\machine learning\\sign lang\\dataset-mp4")    
    p=0
    for (i, word_folder) in enumerate(os.listdir(videos_dir)):
        print(i,end="\n")
        folder_path = os.path.join(videos_dir, word_folder)
        if os.path.isdir(folder_path):
            video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4'))]
            for a, video_file in enumerate(video_files):
                video_path=os.path.join(folder_path, video_file)
                main(video_path,i)
                print(p,end='\r')
                #p+=1
    print("\n")