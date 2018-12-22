import sys
import os
if __name__ == '__main__':
    root_dir = sys.argv[1]
    sub_dirs = os.listdir(root_dir)
    for dir_name in sub_dirs:
        f = open(dir_name+'.txt','w')
        l = open(dir_name+'_label.txt','w')
        person_list = os.listdir(root_dir+'/'+dir_name)
        for person_name in person_list:
            img_list = os.listdir(root_dir+'/'+dir_name+'/'+person_name)
            for image_name in img_list:
                f.write(root_dir+'/'+dir_name+'/'+person_name+'/'+image_name+'\n')
                l.write(person_name+'\n')
        f.close()
        l.close()