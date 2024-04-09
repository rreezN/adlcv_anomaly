import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from scipy import ndimage

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False, one_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            if one_flag:
                self.seqtypes = ['t1']
            else:
                self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            if one_flag:
                self.seqtypes = ['t1', 'seg']
            else:
                self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[2]
                    datapoint[seqtype] = os.path.join(root, f)
                # print(f.split('_'), datapoint)
                # Sort datapoint by sequence type
                datapoint = {k: datapoint[k] for k in self.seqtypes}
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            number=filedict['t1'].split('_')[3].split('.')[0]
            nu = filedict['t1'].split('_')[1]
            n = filedict['t1'].split('\\')[1]
            nib_img = nibabel.load(filedict[seqtype])
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        out_dict = {}
        if self.test_flag:
            path2 = './data/brats21/processed/testing/' + str(n) + '/BraTS2021_' + str(
                nu) + '_seg_' + str(number) + '.nii.gz'

            seg=nibabel.load(path2)
            seg=seg.get_fdata()
            image = torch.zeros(4, 256, 256)
            image[:, 8:-8, 8:-8] = out
            label = seg[None, ...]
            if seg.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"]=weak_label
        else:
            image = torch.zeros(4,256,256)
            image[:,8:-8,8:-8]=out[:-1,...]		#pad to a size of (256,256)
            label = out[-1, ...][None, ...]
            if label.max()>0:
                weak_label=1
            else:
                weak_label=0
            out_dict["y"] = weak_label

        return (image, out_dict, weak_label, label, number)

    def __len__(self):
        return len(self.database)


if __name__ == '__main__':
    ds = BRATSDataset('data/brats21/processed/testing', test_flag=True)
    datal = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=True)
    datal = iter(datal)
    batch, cond, label, _, _ = next(datal)
    print(batch.shape, cond, label)
    # for batch, cond, label in datal:
    #     print(batch.shape, cond, label)
    #     break