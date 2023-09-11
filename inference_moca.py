import os
import glob

model = 'C:\\Users\\xyli45\\Desktop\\git\\RAFT\\model\\raft-things.pth'
root = 'C:\\Users\\xyli45\\Desktop\\datasets\\MoCA-Mask-Pseudo\\MoCA-Video-Test'
folders = sorted(glob.glob(os.path.join(root, '*')))
cnt = 0
for f in folders:
    cnt += 1
    print(f'[{cnt}/{len(folders)}] {f}')
    path_in = f + '\Frame'
    path_out = f + '\Flow'
    mask = f + '\GT'
    assert os.path.exists(path_in)
    # assert os.path.exists(path_out)
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    os.system('python inference.py --model {} --path_in {} --path_out {} --mask {}'.format(
        model, path_in, path_out, mask
    ))


