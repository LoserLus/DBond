import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s]-%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
def mgf2csv(mgf_path:str,csv_path:str) -> None:
    """Convert MGF file to CSV format for Multi-Label Classifaction.

    Args:
        mgf_path (str): Path to the input MGF file.
        csv_path (str): Path to save the output CSV file.
    """
    import pandas as pd
    from pyteomics import mgf
    with  open(mgf_path, "r") as f:
            sp_cnt = 0
            mgf_name_list = []
            mgf_seq_list = []
            mgf_charge_list = []
            mgf_pep_mass_list = []
            mgf_intensity_list = []
            mgf_nce_list = []
            mgf_scan_num_list = []
            mgf_rt_list = []
            mgf_fbr_list = []
            mgf_tb_list = []
            mgf_fb_list = []
            mgf_mb_list = []
            mgf_true_multi_list = []
          
            data = mgf.read(f, convert_arrays=1, read_charges=False,read_ions=False, dtype="float32", use_index=False)
            for sp in data:
                # if sp_cnt > 1: break
                
                seq = sp["params"]["seq"]
                nce = int(sp["params"]["nce"])
                scan_num = int(sp["params"]["scans"])
                charge = int(sp["params"]["charge"][0])
                pep_mass = float(sp["params"]["pepmass"][0])
                intensity = float(sp["params"]["pepmass"][1])
                rt = float(sp["params"]["rtinseconds"])

                name = sp["params"]["title"]
                fbr = float(sp["params"]["fbr"])
                tb = int(sp["params"]["tb"] )
                fb = int(sp["params"]["fb"] )
                mb:str = sp["params"]["mb"] 
                mb_list = [int(index)-1 for index in mb.split(';')[0:-1]]
        
         
             
                bond_label_list = [1]*tb
                for index in mb_list:
                    bond_label_list[index] = 0
         
                mgf_name_list.append(name)
                mgf_true_multi_list.append(';'.join(map(str,bond_label_list)))
                mgf_seq_list.append(seq)
                mgf_charge_list.append(charge)
                mgf_pep_mass_list.append(pep_mass)
                mgf_intensity_list.append(intensity)
                mgf_nce_list.append(nce)
                mgf_scan_num_list.append(scan_num)
                mgf_rt_list.append(rt)
                mgf_fbr_list.append(fbr)
                mgf_tb_list.append(tb)
                mgf_fb_list.append(fb)
                mgf_mb_list.append(mb)
          
                sp_cnt+=1
            # name,seq,charge,pep_mass,intensity,nce,scan_num,rt,fbr,tb,fb,mb,true_multi
            mgf_df = pd.DataFrame({
                'name': mgf_name_list,
                'seq': mgf_seq_list,
                'charge': mgf_charge_list,
                'pep_mass': mgf_pep_mass_list,
                'intensity': mgf_intensity_list,
                'nce': mgf_nce_list,
                'scan_num': mgf_scan_num_list,
                'rt': mgf_rt_list,
                'fbr': mgf_fbr_list,
                'tb': mgf_tb_list,
                'fb': mgf_fb_list,
                'mb': mgf_mb_list,
                'true_multi': mgf_true_multi_list
            })
            mgf_df.to_csv(csv_path,index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert MGF to CSV for Multi-Label Classification")
    parser.add_argument(
        "--mgf_path",
        type=str,
        default="./mgf_dataset/example_out.mgf",
        help="Path to the input MGF file."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./mgf_dataset/example.multi.csv",
        help="Path to save the output CSV file."
    )
    args = parser.parse_args()
    mgf_path = args.mgf_path
    csv_path = args.csv_path
    mgf2csv(mgf_path,csv_path)
    logging.info(f"MGF file {mgf_path} has been converted to CSV file {csv_path}.")