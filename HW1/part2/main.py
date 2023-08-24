import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    if not os.path.exists('./output'):
        os.makedirs('./output')

    # parse setting file
    with open(args.setting_path, 'r') as f:
        lines = f.readlines()
        params = lines[-1].strip().split(',')
        sigma_s = float(params[params.index('sigma_s')+1].strip())
        sigma_r = float(params[params.index('sigma_r')+1].strip())
        del lines[-1]
        del lines[0]
        settings = []
        for line in lines:
            settings.append(np.array(line.strip().split(',')).astype('float32'))

    
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
    print(np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32'))))

    guidances = []
    jbf_outs = []
    errs = []
    for setting in settings:
        print(setting)
        guidance = np.sum(img_rgb * setting, axis=-1)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.uint8)
        guidances.append(guidance)
        jbf_outs.append(jbf_out)
        errs.append(np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32'))))
        print(np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32'))))
    
    # save min error and its guidance
    min_err = np.min(errs)
    min_err_idx = errs.index(min_err)
    min_err_guidance = guidances[min_err_idx]
    min_err_jbf_out = jbf_outs[min_err_idx]
    cv2.imwrite('./output/min_err.png', cv2.cvtColor(min_err_jbf_out, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./output/min_err_guidance.png', cv2.cvtColor(min_err_guidance, cv2.COLOR_RGB2BGR))

    # save max error and its guidance
    max_err = np.max(errs)
    max_err_idx = errs.index(max_err)
    max_err_guidance = guidances[max_err_idx]
    max_err_jbf_out = jbf_outs[max_err_idx]
    cv2.imwrite('./output/max_err.png', cv2.cvtColor(max_err_jbf_out, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./output/max_err_guidance.png', cv2.cvtColor(max_err_guidance, cv2.COLOR_RGB2BGR))
    


if __name__ == '__main__':
    main()