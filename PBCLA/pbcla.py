
from pyteomics import mgf
from utils import Peptide,get_threshold, float_binary_search_with_threshold
from typing import List,Tuple,Dict
import logging
import datetime


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s]-%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# set timezone to Beijing time
logging.Formatter.converter = lambda *args: (datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).timetuple()


ppm = 20


pep_cache_dict :Dict[str,Peptide] = {}
def pbcla(sp:mgf.MGF)->Tuple[int,int,str]:
    """PBCLA算法的核心函数,用于计算多肽的碎片键数和缺失的碎片键
    Args:
        sp (mgf.MGF): MGF格式的谱图数据,要求至少包含以下3类信息:
            - 该质谱图的对应的多肽序列 seq
            - 该质谱图的实际数据 m/z intensity array
    Returns:
        Tuple[int,int,str]: 返回一个元组,包含以下3个元素:
            - total_fragment_bonds (int): 多肽的总碎片键数
            - fragment_bonds (int): 实际检测到的碎片键数
            - missing_bonds (str): 缺失的碎片键,以分号分隔的字符串形式返回,如'1;3;5;'表示第1、3、5个碎片键缺失
    """
    pep_seq = str(sp["params"]["seq"])
    logging.info(f'Processing peptide sequence: {pep_seq}')
    if pep_seq not in pep_cache_dict.keys():

        pep_cache_dict[pep_seq] = Peptide(seq=pep_seq)
        pep_cache_dict[pep_seq].calc_m_z()
    
    logging.info(f"Compute the theoretical m/z for {pep_seq}")
    ion_list = pep_cache_dict[pep_seq].get_m_z_dict()
    ion_list.sort(key=lambda x: x['m_z_t'])
    def match_fragment(item:dict):
        threshold = get_threshold(item['m_z_t'], ppm)
        index = float_binary_search_with_threshold(
            sp["m/z array"], item['m_z_t'], threshold)
        if index >= 0 and  sp["intensity array"][index] > 0 :
            return item['ion_type']
        else:
            return None
        
    logging.info("Match fragments")
    matched_ion_type_list = list(map(match_fragment,ion_list))
    # matched_ion_type_list = list(filter(lambda x: x is not None,matched_ion_type_list))
    matched_ion_type_list = list(filter(None,matched_ion_type_list))

    logging.info("Start Peptide bond labeling algorithm")
    total_fragment_bonds=len(pep_seq)-1
    fragment_bonds = 0
    b_pattern = 'b{}'
    y_pattern = 'y{}'
    b_pattern_nl = 'b{}-'
    y_pattern_nl = 'y{}-'
    bonds_cnt_list = [0] * total_fragment_bonds
    for bond in range(1,total_fragment_bonds+1):
        
        b_str = b_pattern.format(bond)
        y_str = y_pattern.format(total_fragment_bonds+1-bond)
        b_str_nl = b_pattern_nl.format(bond)
        y_str_nl = y_pattern_nl.format(total_fragment_bonds+1-bond)
        for _ion_type in matched_ion_type_list:
            if _ion_type != '':
                # print(_ion_type)
                if b_str == _ion_type or y_str == _ion_type :
                    fragment_bonds+=1
                    bonds_cnt_list[bond-1]+=1
                    # print(_ion_type)
                    break
                elif b_str_nl in _ion_type or y_str_nl in _ion_type:
                    fragment_bonds+=1
                    bonds_cnt_list[bond-1]+=1
                    # print(_ion_type)
                    break
    missing_bonds=''
    for _index in range(len(bonds_cnt_list)):
        if bonds_cnt_list[_index]==0:
            missing_bonds+=str(_index+1)+';'
    return total_fragment_bonds,fragment_bonds,missing_bonds

def process_one_mgf(in_mgf_path:str,out_mgf_path:str) -> None:
    logging.info(f"Process mgf: {in_mgf_path}")
    sps = []
    with  open(in_mgf_path, "r") as f:
        data = mgf.read(f, convert_arrays=1, read_charges=False, dtype="float32", use_index=False)
        for sp in data:
            tb,fb,mb = pbcla(sp)
            sp["params"]["fbr"] = fb/tb
            sp["params"]["tb"] = tb
            sp["params"]["fb"] = fb
            sp["params"]["mb"] = mb
            sps.append(sp)
    logging.info(f"Write mgf to {out_mgf_path}")
    with open(out_mgf_path, "w") as f:
        mgf.write(sps,f)

  
if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser(description="Args for PBCLA")
    parse.add_argument(
        "--in_mgf_path",
        type=str,
        help="Path to the input MGF file",
        default="./mgf_dataset/example.mgf",
    )
    parse.add_argument(
        "--out_mgf_path",
        type=str,
        help="Path to save the output MGF file",
        default="./mgf_dataset/example_out.mgf",
    )
    args = parse.parse_args()
    in_mgf_path = args.in_mgf_path
    out_mgf_path = args.out_mgf_path
    process_one_mgf(in_mgf_path, out_mgf_path)