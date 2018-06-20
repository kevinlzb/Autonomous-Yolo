import os
import json
import io
import mxnet as mx
import numpy as np
import cv2
from random import shuffle

def load_notatoin(notation_path):
    notation_list = []
    with io.open(notation_path,'r') as n:
        notation_content = n.readlines()

    for img_entry in notation_content:
        notation_list.append(json.loads(img_entry))

    print("There are {} images loaded into the list".format(len(notation_list)))
    return notation_list


def get_filename_list(notation_list):
    file_names = []
    for entry in notation_list:
        for key in entry.keys():
            file_names.append(key)

    return file_names


class Data:
    def __init__(self,notation_file,dataset_folder,split_ratio = 0.8):
        self.dataset_folder = dataset_folder
        self.split_ratio = split_ratio
        self.notation_list = load_notatoin(notation_file)
        self.rec_folder_name = './Data_txt/Data_rec'
        if not os.path.exists(self.rec_folder_name):
            os.mkdir(self.rec_folder_name)


    def generate_rec_file(self,image_list = None):

        if image_list is not None:
            rec_file_name = 'small_dataset.rec'
            idx_file_name = 'small_dataset.idx'
            overfit_dataset = mx.recordio.MXIndexedRecordIO(os.path.join(self.rec_folder_name,idx_file_name),
                                                            os.path.join(self.rec_folder_name,rec_file_name),'w')

            for i in range(len(image_list)):
                image_name = image_list[i]
                print("Now is processing {}".format(image_name))
                image_notation = self.get_notation(image_name)
                image_content = cv2.imread(os.path.join(self.dataset_folder,image_name))
                resized_image, new_notation = self.image_resize_notation_transform(image_content,image_notation)
                header = mx.recordio.IRHeader(flag=0, label = new_notation.flatten(), id = i, id2 = 0)
                pack = mx.recordio.pack_img(header,resized_image,quality=100,img_fmt=".jpg")
                overfit_dataset.write_idx(i,pack)
                print("Picture {} has been added into rec-file".format(image_name))

            overfit_dataset.close()

            print("Threre {} pictures have been added into {}".format(str(len(image_list)),rec_file_name))

        else:
            train_rec_name = 'trainRoadImages.rec'
            train_idx_name = 'trainRoadImages.idx'
            val_rec_name = 'valRoadImages.rec'
            val_idx_name = 'valRoadImages.idx'
            train_dataset = mx.recordio.MXIndexedRecordIO(os.path.join(self.rec_folder_name,train_idx_name),
                                                            os.path.join(self.rec_folder_name,train_rec_name),'w')

            val_dataset = mx.recordio.MXIndexedRecordIO(os.path.join(self.rec_folder_name,val_idx_name),
                                                        os.path.join(self.rec_folder_name,val_rec_name),'w')

            train_image_num = int(len(self.notation_list)) * self.split_ratio
            shuffle(self.notation_list)
            for i in range(len(self.notation_list)):
                print i
                for key in self.notation_list[i].keys():
                    image_name = key
                print("Now is processing {}".format(image_name))
                image_notation = self.get_notation(image_name)
                # print os.path.join(self.dataset_folder,image_name)
                image_content = cv2.imread(os.path.join(self.dataset_folder,image_name))
                resized_image, new_notation = self.image_resize_notation_transform(image_content,image_notation)
                header = mx.recordio.IRHeader(flag = 0, label = new_notation.flatten(), id = i,id2 = 0)
                pack = mx.recordio.pack_img(header,resized_image,quality=100,img_fmt=".jpg")
                if i < train_image_num:
                    train_dataset.write_idx(i,pack)
                    print("Picture {} has been added into trainning dataset." .format(image_name))
                else:
                    val_dataset.write_idx(i-train_image_num,pack)
                    print("Picture {} has been added into validatoin dataset.".format(image_name))

            print("Threr are {} images have been added into training dataset".format(str(train_image_num)))
            print("There are {} images have been added into validation dataset".format(str(len(self.notation_list)-train_image_num)))
            train_dataset.close()
            val_dataset.close()

    def get_notation(self, image_name):
        for notation in self.notation_list:
            if image_name in notation.keys():
                return notation[image_name]

    def image_resize_notation_transform(self,image,notations,grid_size=(7,7,9),target_size=224,dscale=32):

        resized_image,new_notations = self.coordinate_transform_on_score_map(image,notations,grid_size,dscale,target_size)
        return resized_image,new_notations


    def get_YOLO_xy(self,bxy,grid_size=(7,7,9),dscale=32,sizet = 224):
        cx,cy = bxy
        assert cx<=1 and cy <=1,"All should be < 1, but get {},and {}".format(cx,cy)
        j = int(np.floor(cx/(1.0/grid_size[0])))
        i = int(np.floor(cy/(1.0/grid_size[1])))
        xyolo = (cx * sizet - j * dscale) / dscale
        yyolo = (cy * sizet - i * dscale) / dscale

        return [i, j, xyolo, yyolo]

    def coordinate_transform_on_score_map(self,image,bbox,grid_size=(7,7,9),dscale=32,target_size = 224):
        # print type(image)
        himg,wimg = image.shape[:2]
        imgR = cv2.resize(image,dsize=(target_size,target_size))
        bboxyolo = np.zeros(grid_size)
        for eachbox in bbox:
            tl_x, tl_y, br_x, br_y,cls = eachbox
            h = int(br_y - tl_y)
            w = int(br_x - tl_x)
            cy = tl_y + h / 2.0
            cx = tl_x + w / 2.0
            cxt = 1.0*cx/wimg
            cyt = 1.0*cy/himg
            wt = 1.0*w/wimg
            ht = 1.0*h/himg
            assert wt<1 and ht<1
            i,j,xyolo,yyolo = self.get_YOLO_xy([cxt,cyt],grid_size,dscale,target_size)
            print "one yolo box is {}".format((i,j,xyolo,yyolo,wt,ht))
            if cls is 1:
                label_vec = np.asarray([1, xyolo, yyolo, wt, ht, 1, 0, 0, 0])
            elif cls is 2:
                label_vec = np.asarray([1, xyolo, yyolo, wt, ht, 0, 1, 0, 0])
            elif cls is 3:
                label_vec = np.asarray([1, xyolo, yyolo, wt, ht, 0, 0, 1, 0])
            else:
                label_vec = np.asarray([1, xyolo, yyolo, wt, ht, 0, 0, 0, 1])

            bboxyolo[i,j,:] = label_vec

        return imgR,bboxyolo



if __name__ == '__main__':
    from random import shuffle
    # import random
    train_dataset_folder = './Data_txt/train'
    val_dataset_folder = './Data_txt/val'
    dataset_root = 'Data_txt/RoadImages'
    val_ratio = 0.2
    notation_file_path = os.path.join('Data_txt/RoadImages','all_label.idl')
    val_label_name = 'val_label.idl'
    train_label_name = 'train_label.idl'
    small_overfit_images = ["60091.jpg", "67996.jpg","68015.jpg", "70076.jpg", "68785.jpg", "68808.jpg",
                            "68040.jpg", "67999.jpg", "68192.jpg", "70070.jpg"]
    data = Data(notation_file_path,dataset_root,split_ratio = 0.8)
    data.generate_rec_file()
    # notation_content = load_notatoin(notation_file_path)
    # shuffle(notation_content)
    # val_notaion = []
    # train_notation = []
    # val_image_num= int(len(notation_content) * 0.2)
    # for i in xrange(len(notation_content)):
    #     if i < val_image_num:
    #         val_notaion.append(notation_content[i])
    #         # for key in notation_content[i].keys():
    #         #     path_from = os.path.join(train_dataset_folder,key)
    #         #     path_to = os.path.join(val_dataset_folder,key)
    #         #     os.rename(path_from,path_to)
    #     else:
    #         train_notation.append(notation_content[i])
    #
    # with open(os.path.join(dataset_root,val_label_name),'w') as n:
    #     for item in val_notaion:
    #         json_str = json.dumps(item)
    #         n.write(json_str + '\n')
    #
    # with open(os.path.join(dataset_root,train_label_name),'w') as n:
    #     for item in train_notation:
    #         json_str = json.dumps(item)
    #         n.write(json_str + '\n')

