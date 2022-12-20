import glob

images = glob.glob("/playpen-raid2/lin.tian/data/BraTS-Reg/BraTSReg_Training_Data_v3/BraTSReg_*")
images = sorted(images)
#/playpen-raid2/lin.tian/data/BraTS-Reg/BraTSReg_Training_Data_v3/BraTSReg_001/BraTSReg_001_00_0000_t1.nii.gz
#/playpen-raid2/lin.tian/data/BraTS-Reg/BraTSReg_Training_Data_v3/BraTSReg_001/BraTSReg_001_01_0106_t1.nii.gz
pair1 = [glob.glob(i + "/BraTSReg_*_00_0000_t1.nii.gz")[0] for i in images]
pair2 = [glob.glob(i + "/BraTSReg_*_01_*_t1.nii.gz")[0] for i in images]

images = list(zip(pair1, pair2))
[print(i) for i in images]


train = images[:-10]
test = images[10:]



