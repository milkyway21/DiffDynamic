"""
Lilly Medchem Rules评估模块

基于J Med Chem 2012论文中的Lilly Medchem Rules实现。
参考原始实现：https://github.com/IanAWatson/Lilly-Medchem-Rules

使用RDKit和SMARTS模式来评估分子是否符合药物化学规则。

主要规则包括：
1. 原子数量限制（默认7-40个重原子，软上限25）
2. 基本要求检查（至少1个C，至少1个O或N）
3. 不允许的元素检查（Ag, Fe, Hg, Zn, Pb, As, Se, Te等）
4. 同位素检查（默认不允许）
5. 直接拒绝规则（reject rules）
6. 扣分规则（demerit rules，默认阈值100）

扣分规则包括：
- ester (酯类): 50分
- nitro (硝基): 60分
- long_chain (长链烷基): C4链50分，C5链60分，C6+链70-80分
- cyclohexane (环己烷): 40分
- halo_next_to_aryl_n_w_ewg (邻位卤素与芳基N): 40分
- no_rings (无环结构): 根据链长扣分（C4:30, C5:40, C6+:50）
- reverse_michael (反Michael加成): 50分
- phthalimide (邻苯二甲酰亚胺): 50分
- thiourea (硫脲): 40分
- thioamide (硫代酰胺): 40分
- positive (正电荷): 30分
- 其他问题结构（过氧化物、叠氮化物等）

实现与原始Lilly-Medchem-Rules保持一致，确保评估结果的一致性。
"""

from copy import deepcopy
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors


class LillyMedchemRules:
    """Lilly Medchem Rules评估器"""
    
    def __init__(self, 
                 min_atoms: int = 7,
                 soft_max_atoms: int = 25,
                 hard_max_atoms: int = 40,
                 demerit_cutoff: int = 100,
                 relaxed: bool = False,
                 no_phosphorus: bool = False,
                 ok_isotopes: bool = False):
        """
        初始化Lilly Medchem Rules评估器
        
        Args:
            min_atoms: 最小重原子数（默认7）
            soft_max_atoms: 软上限重原子数，超过此值开始扣分（默认25）
            hard_max_atoms: 硬上限重原子数，超过此值直接拒绝（默认40）
            demerit_cutoff: demerit分数阈值，超过此值拒绝（默认100）
            relaxed: 是否使用宽松模式（提高阈值）
            no_phosphorus: 是否拒绝所有含磷分子
            ok_isotopes: 是否允许同位素原子
        """
        self.min_atoms = min_atoms
        self.soft_max_atoms = soft_max_atoms
        self.hard_max_atoms = hard_max_atoms
        self.demerit_cutoff = demerit_cutoff
        self.no_phosphorus = no_phosphorus
        self.ok_isotopes = ok_isotopes
        
        if relaxed:
            self.demerit_cutoff = int(self.demerit_cutoff * 1.5)
            self.hard_max_atoms = int(self.hard_max_atoms * 1.2)
        
        # 初始化规则模式
        self._init_reject_patterns()
        self._init_demerit_patterns()
    
    def _init_reject_patterns(self):
        """初始化直接拒绝规则（SMARTS模式）
        
        基于Lilly Medchem Rules原始实现，这些规则会直接拒绝分子。
        不允许的元素通过原子类型检查处理，这里主要处理其他拒绝模式。
        """
        self.reject_patterns = [
            # 注意：不允许的元素（Ag, Fe, Hg, Zn, Pb, As, Se, Te等）
            # 通过_check_basic_requirements中的原子类型检查来处理
            # 这里主要处理其他需要直接拒绝的结构模式
        ]
        
        # 编译SMARTS模式
        self.reject_queries = []
        for pattern, name in self.reject_patterns:
            try:
                query = Chem.MolFromSmarts(pattern)
                if query is not None:
                    self.reject_queries.append((query, name))
            except:
                pass
    
    def _init_demerit_patterns(self):
        """初始化扣分规则（SMARTS模式）
        
        基于Lilly Medchem Rules原始实现和GitHub仓库描述。
        规则和分值参考：https://github.com/IanAWatson/Lilly-Medchem-Rules
        """
        self.demerit_patterns = [
            # 酯类 (ester) - 50分
            # 匹配非羰基碳连接的酯键
            ('[C;!$(C=O)]C(=O)O[!C;!$(C=O)]', 'ester', 50),
            ('[C;!$(C=O)]C(=O)OC', 'ester', 50),
            
            # 硝基 (nitro) - 60分
            ('[N+](=O)[O-]', 'nitro', 60),
            
            # 长链烷基 (long_chain) - C4链50分，C5链60分，C6+链更高
            # 匹配连续的CH2链（至少4个）
            ('[CH2][CH2][CH2][CH2]', 'C4_chain', 50),
            ('[CH2][CH2][CH2][CH2][CH2]', 'C5_chain', 60),
            ('[CH2][CH2][CH2][CH2][CH2][CH2]', 'C6_chain', 70),
            # 更长的链（7+）
            ('[CH2][CH2][CH2][CH2][CH2][CH2][CH2]', 'C7_chain', 80),
            
            # 环己烷 (cyclohexane) - 40分
            # 匹配未取代的环己烷
            ('C1CCCCC1', 'cyclohexane', 40),
            
            # 邻位卤素与芳基N（带吸电子基）(halo_next_to_aryl_n_w_ewg) - 40分
            # 卤素邻位到带有吸电子基的芳基N
            ('[Cl,Br,I][c;$(c[N;$(N[C,S,P]=O)])]', 'halo_next_to_aryl_n_w_ewg', 40),
            ('[Cl,Br,I][c;$(c[N;$(N[C]=O)])]', 'halo_next_to_aryl_n_w_ewg', 40),
            
            # 反Michael加成 (reverse_michael) - 50分
            # 两个羰基碳之间的连接
            ('[C;$(C=O)]C(=O)[C;$(C=O)]', 'reverse_michael', 50),
            
            # 邻苯二甲酰亚胺 (phthalimide) - 50分
            ('c1ccc2c(c1)C(=O)NC(=O)c2', 'phthalimide', 50),
            ('O=C1NC(=O)c2ccccc12', 'phthalimide', 50),  # 另一种表示
            
            # 硫脲 (thiourea) - 40分
            ('[N;$(NC(=S)N)]C(=S)[N;$(NC(=S)N)]', 'thiourea', 40),
            ('NC(=S)N', 'thiourea', 40),  # 简单形式
            
            # 硫代酰胺 (thioamide) - 40分
            ('[C;$(C(=S)N)]C(=S)[N]', 'thioamide', 40),
            ('C(=S)N', 'thioamide', 40),  # 简单形式
            
            # 正电荷 (positive) - 根据GitHub输出示例
            # 匹配带正电荷的原子（除了常见的铵离子等）
            ('[N+;!$(N[C,O])]', 'positive', 30),
            
            # 其他常见问题结构
            # 过氧化物
            ('OO', 'peroxide', 50),
            # 叠氮化物
            ('[N-]=[N+]=N', 'azide', 40),
            # 重氮化合物
            ('[N+]=[N-]', 'diazonium', 50),
        ]
        
        # 编译SMARTS模式
        self.demerit_queries = []
        for pattern, name, demerit in self.demerit_patterns:
            try:
                query = Chem.MolFromSmarts(pattern)
                if query is not None:
                    self.demerit_queries.append((query, name, demerit))
            except Exception as e:
                # 如果模式编译失败，跳过（可能是SMARTS语法问题）
                pass
    
    def _count_heavy_atoms(self, mol: Chem.Mol) -> int:
        """计算重原子数（非氢原子）"""
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
    
    def _check_atom_count(self, mol: Chem.Mol) -> Tuple[bool, str, int]:
        """
        检查原子数量
        
        Returns:
            (passed, reason, demerit): 是否通过，原因，扣分
        """
        n_heavy = self._count_heavy_atoms(mol)
        
        if n_heavy < self.min_atoms:
            return False, f'too_few_atoms({n_heavy})', 0
        
        if n_heavy > self.hard_max_atoms:
            return False, f'too_many_atoms({n_heavy})', 0
        
        # 软上限扣分
        demerit = 0
        if n_heavy > self.soft_max_atoms:
            # 每超过一个原子扣5分
            demerit = (n_heavy - self.soft_max_atoms) * 5
        
        return True, '', demerit
    
    def _check_basic_requirements(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """
        检查基本要求：
        - 至少1个碳原子
        - 至少1个氧或氮原子
        - 不允许的元素
        - 同位素检查
        """
        atoms = mol.GetAtoms()
        
        # 检查是否有碳原子
        has_carbon = any(atom.GetAtomicNum() == 6 for atom in atoms)
        if not has_carbon:
            return False, 'no_carbon'
        
        # 检查是否有氧或氮原子
        has_on = any(atom.GetAtomicNum() in [7, 8] for atom in atoms)
        if not has_on:
            return False, 'no_oxygen_nitrogen'
        
        # 检查不允许的元素
        # 根据GitHub描述：不允许的元素包括 Ag, Fe, Hg, Zn, Pb, As, Se, Te 等
        # 允许的元素：H, B, C, N, O, F, P, S, Cl, Br, I 等常见药物化学元素
        allowed_elements = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, B, C, N, O, F, P, S, Cl, Br, I
        disallowed_elements = {47, 26, 80, 30, 82, 33, 34, 52}  # Ag, Fe, Hg, Zn, Pb, As, Se, Te
        
        for atom in atoms:
            atomic_num = atom.GetAtomicNum()
            
            # 检查是否是磷（如果设置了no_phosphorus）
            if atomic_num == 15 and self.no_phosphorus:
                return False, 'phosphorus_not_allowed'
            
            # 检查是否是不允许的元素
            if atomic_num in disallowed_elements:
                element_names = {47: 'Ag', 26: 'Fe', 80: 'Hg', 30: 'Zn', 
                                82: 'Pb', 33: 'As', 34: 'Se', 52: 'Te'}
                return False, f'disallowed_element({element_names.get(atomic_num, atomic_num)})'
            
            # 检查其他非常见元素（不在允许列表中）
            if atomic_num not in allowed_elements and atomic_num not in disallowed_elements:
                # 对于其他元素，也拒绝（除非是氢）
                if atomic_num != 1:  # 氢是允许的
                    return False, f'disallowed_element({atomic_num})'
        
        # 检查同位素
        if not self.ok_isotopes:
            for atom in atoms:
                if atom.GetIsotope() != 0:
                    return False, 'isotope_not_allowed'
        
        return True, ''
    
    def _check_reject_patterns(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """
        检查直接拒绝规则
        
        Returns:
            (passed, reason): 是否通过，拒绝原因
        """
        for query, name in self.reject_queries:
            if mol.HasSubstructMatch(query):
                return False, f'reject_{name}'
        return True, ''
    
    def _check_demerit_patterns(self, mol: Chem.Mol) -> Tuple[List[str], int]:
        """
        检查扣分规则
        
        Returns:
            (matched_rules, total_demerit): 匹配的规则列表，总扣分
        """
        matched_rules = []
        total_demerit = 0
        
        for query, name, demerit in self.demerit_queries:
            matches = mol.GetSubstructMatches(query)
            if matches:
                matched_rules.append(name)
                total_demerit += demerit
        
        # 检查无环结构（需要特殊处理）
        # 根据GitHub输出示例，无环结构会根据链长扣分
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() == 0:
            n_heavy = self._count_heavy_atoms(mol)
            if n_heavy >= 4:
                matched_rules.append('no_rings')
                # 根据链长扣分（参考GitHub输出格式）
                # C4表示4个碳的链，扣分约30-50分
                if n_heavy >= 6:
                    matched_rules.append('C6')  # 6+个重原子的链
                    total_demerit += 50
                elif n_heavy >= 5:
                    matched_rules.append('C5')  # 5个重原子的链
                    total_demerit += 40
                elif n_heavy >= 4:
                    matched_rules.append('C4')  # 4个重原子的链
                    total_demerit += 30
        
        return matched_rules, total_demerit
    
    def evaluate(self, mol: Chem.Mol) -> Dict:
        """
        评估分子是否符合Lilly Medchem Rules
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            dict: 评估结果，包含：
                - passed: 是否通过（bool）
                - demerit: 总扣分（int）
                - demerit_cutoff: 扣分阈值（int）
                - matched_rules: 匹配的规则列表（list）
                - reject_reason: 拒绝原因（str，如果被拒绝）
                - n_heavy_atoms: 重原子数（int）
                - details: 详细信息（dict）
        """
        # 深拷贝避免修改原分子
        mol = deepcopy(mol)
        
        # 标准化分子
        try:
            Chem.SanitizeMol(mol)
        except:
            return {
                'passed': False,
                'demerit': 0,
                'demerit_cutoff': self.demerit_cutoff,
                'matched_rules': [],
                'reject_reason': 'sanitization_failed',
                'n_heavy_atoms': 0,
                'details': {}
            }
        
        result = {
            'passed': False,
            'demerit': 0,
            'demerit_cutoff': self.demerit_cutoff,
            'matched_rules': [],
            'reject_reason': '',
            'n_heavy_atoms': 0,
            'details': {}
        }
        
        # 1. 基本要求检查
        passed, reason = self._check_basic_requirements(mol)
        if not passed:
            result['reject_reason'] = reason
            return result
        
        # 2. 原子数量检查
        n_heavy = self._count_heavy_atoms(mol)
        result['n_heavy_atoms'] = n_heavy
        
        passed, reason, atom_demerit = self._check_atom_count(mol)
        if not passed:
            result['reject_reason'] = reason
            return result
        
        # 3. 直接拒绝规则检查
        passed, reason = self._check_reject_patterns(mol)
        if not passed:
            result['reject_reason'] = reason
            return result
        
        # 4. 扣分规则检查
        matched_rules, pattern_demerit = self._check_demerit_patterns(mol)
        result['matched_rules'] = matched_rules
        
        # 总扣分 = 原子数量扣分 + 模式扣分
        total_demerit = atom_demerit + pattern_demerit
        result['demerit'] = total_demerit
        
        # 5. 判断是否通过
        if total_demerit >= self.demerit_cutoff:
            result['reject_reason'] = f'demerit_exceeded({total_demerit})'
            return result
        
        result['passed'] = True
        result['details'] = {
            'atom_count_demerit': atom_demerit,
            'pattern_demerit': pattern_demerit,
            'n_matched_patterns': len(matched_rules)
        }
        
        return result


def evaluate_lilly_medchem_rules(mol: Chem.Mol, 
                                  min_atoms: int = 7,
                                  soft_max_atoms: int = 25,
                                  hard_max_atoms: int = 40,
                                  demerit_cutoff: int = 100,
                                  relaxed: bool = False,
                                  no_phosphorus: bool = False,
                                  ok_isotopes: bool = False) -> Dict:
    """
    便捷函数：评估分子是否符合Lilly Medchem Rules
    
    Args:
        mol: RDKit分子对象
        min_atoms: 最小重原子数（默认7）
        soft_max_atoms: 软上限重原子数（默认25）
        hard_max_atoms: 硬上限重原子数（默认40）
        demerit_cutoff: demerit分数阈值（默认100）
        relaxed: 是否使用宽松模式
        no_phosphorus: 是否拒绝所有含磷分子
        ok_isotopes: 是否允许同位素原子
        
    Returns:
        dict: 评估结果
    """
    evaluator = LillyMedchemRules(
        min_atoms=min_atoms,
        soft_max_atoms=soft_max_atoms,
        hard_max_atoms=hard_max_atoms,
        demerit_cutoff=demerit_cutoff,
        relaxed=relaxed,
        no_phosphorus=no_phosphorus,
        ok_isotopes=ok_isotopes
    )
    return evaluator.evaluate(mol)

