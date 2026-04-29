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
            mgf_bond_aa_list = []
            mgf_bond_pos_list = []
            mgf_bond_label_list = []
            mgf_seq_list = []
            mgf_charge_list = []
            mgf_pep_mass_list = []
            mgf_intensity_list = []
            mgf_nce_list = []
            mgf_scan_num_list = []
            mgf_rt_list = []

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
        
                bond_pos_list = [index for index in range(tb)]
                bond_aa_list = [seq[index:index+2] for index in bond_pos_list]
                bond_label_list = [1]*tb
                for index in mb_list:
                    bond_label_list[index] = 0
                seq_list = [seq]*tb
                charge_list = [charge]*tb
                pep_mass_list = [pep_mass]*tb
                intensity_list = [intensity]*tb
                nce_list = [nce]*tb
                scan_num_list = [scan_num]*tb
                rt_list = [rt]*tb
                mgf_bond_aa_list.extend(bond_aa_list)
                mgf_bond_pos_list.extend(bond_pos_list)
                mgf_bond_label_list.extend(bond_label_list)
                mgf_seq_list.extend(seq_list)
                mgf_charge_list.extend(charge_list)
                mgf_pep_mass_list.extend(pep_mass_list)
                mgf_intensity_list.extend(intensity_list)
                mgf_nce_list.extend(nce_list)
                mgf_scan_num_list.extend(scan_num_list)
                mgf_rt_list.extend(rt_list)
          
                sp_cnt+=1
            
            mgf_df = pd.DataFrame({
                'bond_aa': mgf_bond_aa_list,
                'bond_pos': mgf_bond_pos_list,
                'bond_label': mgf_bond_label_list,
                'seq': mgf_seq_list,
                'charge': mgf_charge_list,
                'pep_mass': mgf_pep_mass_list,
                'intensity': mgf_intensity_list,
                'nce': mgf_nce_list,
                'scan_num': mgf_scan_num_list,
                'rt': mgf_rt_list
            })
            mgf_df.to_csv(csv_path,index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert MGF to CSV for Multi-Label Classification")
    parser.add_argument(
        "--mgf_path",
        type=str,
        help="Path to the input MGF file",
        default="./mgf_dataset/example_out.mgf",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to save the output CSV file",
        default="./mgf_dataset/example.csv",
    )
    args = parser.parse_args()
    mgf_path = args.mgf_path
    csv_path = args.csv_path
    mgf2csv(mgf_path,csv_path)
    logging.info(f"MGF file {mgf_path} has been converted to CSV file {csv_path}.")