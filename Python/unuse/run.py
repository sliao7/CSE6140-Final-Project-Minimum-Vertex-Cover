import subprocess
import os

if __name__ == '__main__':

    file_list = os.listdir('../DATA')
    file_list = [file for file in file_list if 'graph' in file]

    for file in file_list:
        file_path = os.path.join('../DATA',file)
        print(file)
        subprocess.call(['python','approx.py','-inst',file_path, '-alg','Approx'])