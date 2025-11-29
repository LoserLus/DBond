
from typing import List,Dict
from pyteomics import mass

mass.std_aa_comp.update({
    # BELOW NEW
    'B':   mass.mass.Composition({'H': 6, 'C': 3, 'O': 1, 'N': 2}),
    'O':   mass.Composition({'H': 10, 'C': 5, 'O': 1, 'N': 2}),
    # Z -> X2 X->X1
    'Z':   mass.Composition({'H': 15, 'C': 9, 'O': 1, 'N': 1}),
    'X':   mass.Composition({'H': 8, 'C': 8, 'O': 1, 'N': 2}),
})
mass.std_aa_mass.update({
      # BELOW NEW
    'B': 86.04801,
    'O': 114.07931,
    # X->X1 Z->X2
    'X': 148.06366,
    'Z': 153.11536
})
class Peptide:
    ion_max_charge = 2
    ion_types_left = [ 'b{}','b{}-H2O','b{}-NH3']
    ion_types_right = ['y{}','y{}-H2O','y{}-NH3']


    def __init__( self, seq: str):
        """多肽实例化

        Args:
            seq (str): 多肽的序列(包括B、O)
        """
        self.seq = seq
        self.length = len(seq)
        self.ion_dict_list:List[Dict] = []
        self.ifCalc = False

    def calc_m_z(self) -> None:
        """计算多肽可能产生的所有离子的理论质核比
        """
        seq = self.seq
        if self.ifCalc:
            return
        # a，b，c离子需要从左向右数,如pep='ABCD',b3在相当于'ABC'
        for ion_type in Peptide.ion_types_left:
            for charge in range(1, Peptide.ion_max_charge+1):
                for length in range(1, self.length):
                    # tmp = ion_type.format(length)
                    m_z_t =  mass.fast_mass(sequence=seq[:length], ion_type=ion_type.format(''), charge=charge)
                    self.ion_dict_list.append({'ion_type':ion_type.format(length),'ion_len':length,'charge':charge,'m_z_t':m_z_t})

        # x,y,z离子需要从右向左数,如pep='ABCD',y3在相当于'BCD'
        for ion_type in Peptide.ion_types_right:
            for charge in range(1, Peptide.ion_max_charge+1):
                for length in range(1, self.length):
                    m_z_t = mass.fast_mass(sequence=seq[-1*length:], ion_type=ion_type.format(''), charge=charge)
                    self.ion_dict_list.append({'ion_type':ion_type.format(length),'ion_len':length,'charge':charge,'m_z_t':m_z_t})

        self.ifCalc = True

    def get_m_z_dict(self) -> List[dict]:
        """以list形式返回3维字典d中的所有元素
        Returns:
            List[dict]: 以dict形式保存的结果，每个元素形如{'ion_type':'b6-NH3','ion_len':6,'charge':2,'m_z_t':114.987}
        """
        if self.ifCalc:
            return self.ion_dict_list
        else:
            self.calc_m_z()
            return self.ion_dict_list


def get_threshold(m_z: float, ppm: int = 20) -> float:
    """对于给定的质核比m_z,根据ppm计算阈值并返回

    Args:
        m_z (float): 给定的质核比
        ppm (int): ppm值

    Returns:
        threshold (float): 计算得到的阈值
    """
    # bottom = int(m_z)
    # threshold = bottom*ppm/1e6
    threshold = ppm*m_z/1e6
    return threshold



def float_binary_search_with_threshold(arr: List[float], key: float, threshold: float, eps: float = 1e-6) -> int:
    """这个函数通过二分查找,在数组arr中,找到在threshold范围内最接近key的值的下标并返回,如果找不到,则返回值为-1;
    Args:
        arr (List[float]): 待查找的数组
        key (float): 待查找的key
        threshold (float): 比较阈值,在阈值范围内认为匹配成功
        eps (float, optional): 浮点数比较精度,当两个浮点数的差值的绝对值小于eps时,认为它们相等 Defaults to 1e-6.

    Returns:
        int: 如果匹配成功,返回下标;否则返回-1。
     示例用法1:
        arr = [1.0, 2.50, 2.51, 2.52, 2.53, 2.54, 2.55, 2.56, 2.6]
        key = 2.50
        threshold = 0.2
        返回值为:2.5所对应的下标1
    示例用法2:
        arr = [1.0, 2.5, 2.6, 2.7]
        key = 0.799999
        threshold = 0.2
        返回值为1.0所对应的下标0
    """
    left = 0
    right = len(arr) - 1

    if len(arr) == 0:  # 列表为空
        return -1

    # 当key和arr中的最小值min相等
    if abs(arr[left]-key) <= eps:
        return left

    # 当key和arr中的最大值max相等
    if abs(arr[right]-key) <= eps:
        return right

    # 处理边界情况，arr中的最小值min大于key
    # 当|min-key|<=threshold时，认为key和min匹配
    if arr[left] > key:
        # 计算差值delta，delta>0
        delta = arr[left]-key
        # 当差值和阈值在精度范围内不相等
        if abs(delta - threshold) > eps:
            # 差值小于阈值
            if delta < threshold:
                return left
            else:
                return -1
        # 当差值和阈值在精度范围内相等
        else:
            return left

   # 处理边界情况，arr中的最大值max小于key
    # 当|max-key|<=threshold时，认为key和max匹配
    if arr[right] < key:
        # 计算差值delta，delta>0
        delta = key - arr[right]
        # 当差值和阈值在精度范围内不相等
        if abs(delta - threshold) > eps:
            # 差值小于阈值
            if delta < threshold:
                return right
            else:
                return -1
        # 当差值和阈值在精度范围内相等
        else:
            return right
    # 此时arr中的最小值min，最大值max满足：min<key<max
    closest_idx = -1
    min_diff = float("inf")

    while left <= right:
        mid = (left + right) // 2
        diff = arr[mid] - key

        # 此时arr[mid]和key在精度范围内相等，不考虑在精度范围内再排序找最接近
        if abs(diff) <= eps:
            closest_idx = mid
            return closest_idx

        # 此时在精度范围内arr[mid]>key
        elif diff > 0:
            # 此时精度范围内key<arr[mid]<=key+threshold
            # 此时可以认为key与arr[mid]匹配
            if diff < threshold or abs(diff - threshold) <= eps:
                # 考虑key与arr[mid]是否是最接近的匹配
                if min_diff > diff:
                    # 出现了key与arr[mid]更接近的匹配
                    closest_idx = mid
                    min_diff = diff
                right = mid - 1
            # 此时精度范围内arr[mid]>key+threshold
            else:
                right = mid - 1
        # 此时在精度范围内arr[mid]<key
        elif diff < 0:
            # 此时精度范围内key-threshold<=arr[mid]<key
            # 此时可以认为key与arr[mid]匹配
            if -1*diff < threshold or abs(-1*diff - threshold) <= eps:
                # 考虑key与arr[mid]是否是最接近的匹配
                if min_diff > -1*diff:
                    # 出现了key与arr[mid]更接近的匹配
                    closest_idx = mid
                    min_diff = -1*diff
                left = mid + 1
            else:
                left = mid + 1

    return closest_idx
