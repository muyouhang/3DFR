import os

if __name__ == '__main__':
    root_dir = 'depth_image'
    img_lst = os.listdir(root_dir)
    fp = open('depth.txt','w')
    for fname in img_lst:
        fp.write('%s %s/%s\n'%(fname.split('.p')[0],root_dir,fname))
    fp.close()