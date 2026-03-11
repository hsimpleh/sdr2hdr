import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
from GBDT_RF.train import train_from_dir
from GBDT_RF.infer import infer_sdr_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer'], required=True)
    parser.add_argument('--sdr_dir', type=str, required=True, help="SDR 图像目录")
    parser.add_argument('--hdr', type=str, help="HDR 图像目录（推理时可选，训练时必须）")
    parser.add_argument('--out', type=str, default='output/', help="输出目录")
    parser.add_argument('--model_dir', type=str, default='models/model1', help="模型目录")
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.hdr:
            print("❌ 训练模式必须提供 HDR 目录")
        else:
            train_from_dir(args.sdr_dir, args.hdr, model_dir=args.model_dir)

    elif args.mode == 'infer':
        os.makedirs(args.out, exist_ok=True)

        sdr_files = sorted(glob.glob(os.path.join(args.sdr_dir, '*')))
        hdr_files = sorted(glob.glob(os.path.join(args.hdr, '*')) if args.hdr else [])
        sdr_dict = {os.path.basename(f): f for f in sdr_files}
        hdr_dict = {os.path.basename(f): f for f in hdr_files}
        common_files = sorted(set(sdr_dict) & set(hdr_dict)) if args.hdr else list(sdr_dict.keys())

        psnr_list, ssim_list, delta_e_list = [], [], []

        for fname in tqdm(common_files):
            sdr_path = sdr_dict[fname]
            gt_hdr_path = hdr_dict.get(fname) if args.hdr else None
            out_path = os.path.join(args.out, fname)

            res = infer_sdr_image(sdr_path, model_dir=args.model_dir, gt_hdr_path=gt_hdr_path, output_path=out_path)
            if res:
                psnr, ssim, delta_e = res
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                delta_e_list.append(delta_e)
                print(f"{fname} ➤ PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | ΔE: {delta_e:.2f}")

        if psnr_list:
            print("\n📊 推理完成！")
            print(f"✅ 平均 PSNR: {np.mean(psnr_list):.2f}")
            print(f"✅ 平均 SSIM: {np.mean(ssim_list):.4f}")
            print(f"✅ 平均 ΔE: {np.mean(delta_e_list):.2f}")
