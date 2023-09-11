import os
import glob

roots = ['C:\\Users\\xyli45\\Desktop\\datasets\\MoCA-Mask-Pseudo\\MoCA-Video-Test',
         'C:\\Users\\xyli45\\Desktop\\datasets\\MoCA-Mask-Pseudo\\MoCA-Video-Train']
orig_dirs = ['\\Frame', '\\Flow', '\\GT']
dirs = ['\\frame', '\\flow', '\\mask']
split = ['\\val', '\\train']
dst_root = 'C:\\Users\\xyli45\\Desktop\\datasets\\MoCA-Mask-Flow'
# Test
for is_test in [0, 1]:
    root = roots[is_test]
    cnt = 0
    cases = sorted(glob.glob(os.path.join(root, '*')))
    for case in cases:
        cnt += 1
        case_name = os.path.basename(case)
        print(f'[{cnt}/{len(cases)}] {case_name}')
        frames = sorted(glob.glob(os.path.join(case + orig_dirs[0], '*.jpg')))
        flows = sorted(glob.glob(os.path.join(case + orig_dirs[1], '*.png')))
        masks = sorted(glob.glob(os.path.join(case + orig_dirs[2], '*.png')))
        assert len(frames) == len(masks)
        assert len(frames)-1 == len(flows)
        # n+1
        for frame in frames[:-1]:
            file_name = os.path.basename(frame)
            dst = dst_root + dirs[0] + split[is_test] + '\\' + case_name + '_' + file_name
            assert os.path.exists(frame)
            os.system(f'copy {frame} {dst}')
            assert os.path.exists(dst)
        # n
        for flow in flows:
            file_name = os.path.basename(flow)
            dst = dst_root + dirs[1] + split[is_test] + '\\' + case_name + '_' + file_name
            assert os.path.exists(flow)
            os.system(f'copy {flow} {dst}')
            assert os.path.exists(dst)
        # n+1
        for mask in masks[:-1]:
            file_name = os.path.basename(mask)
            dst = dst_root + dirs[2] + split[is_test] + '\\' + case_name + '_' + file_name
            assert os.path.exists(mask)
            os.system(f'copy {mask} {dst}')
            assert os.path.exists(dst)




