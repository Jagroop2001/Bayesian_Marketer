import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import math

class StatisticalTests:
    """
    Performs traditional statistical tests for A/B testing analysis.
    """
    
    def __init__(self, data: pd.DataFrame, group_a: str, group_b: str):
        self.data = data
        self.group_a = group_a
        self.group_b = group_b
        
        # Prepare group data
        self.data_a = data[data['test group'] == group_a]
        self.data_b = data[data['test group'] == group_b]
        
        # Basic statistics
        self.n_a = len(self.data_a)
        self.n_b = len(self.data_b)
        self.conv_a = self.data_a['converted'].sum()
        self.conv_b = self.data_b['converted'].sum()
        self.rate_a = self.conv_a / self.n_a if self.n_a > 0 else 0
        self.rate_b = self.conv_b / self.n_b if self.n_b > 0 else 0
    
    def z_test(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Perform two-proportion z-test.
        
        Args:
            confidence_level: Confidence level for the test
            
        Returns:
            Dictionary containing z-test results
        """
        # Pooled proportion
        p_pool = (self.conv_a + self.conv_b) / (self.n_a + self.n_b)
        
        # Standard error
        se_pool = math.sqrt(p_pool * (1 - p_pool) * (1/self.n_a + 1/self.n_b))
        
        # Z-statistic
        z_stat = (self.rate_b - self.rate_a) / se_pool if se_pool > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Critical value
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'z_critical': z_critical,
            'is_significant': abs(z_stat) > z_critical,
            'pooled_proportion': p_pool,
            'standard_error': se_pool
        }
    
    def chi_square_test(self) -> Dict[str, float]:
        """
        Perform chi-square test of independence.
        
        Returns:
            Dictionary containing chi-square test results
        """
        # Contingency table
        contingency_table = np.array([
            [self.conv_a, self.n_a - self.conv_a],
            [self.conv_b, self.n_b - self.conv_b]
        ])
        
        # Chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected,
            'contingency_table': contingency_table
        }
    
    def confidence_interval(self, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for the difference in proportions.
        
        Args:
            confidence_level: Confidence level for the interval
            
        Returns:
            Tuple containing (lower_bound, upper_bound)
        """
        # Difference in proportions
        diff = self.rate_b - self.rate_a
        
        # Standard error for difference
        se_diff = math.sqrt(
            (self.rate_a * (1 - self.rate_a) / self.n_a) + 
            (self.rate_b * (1 - self.rate_b) / self.n_b)
        )
        
        # Critical value
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval
        margin_error = z_critical * se_diff
        lower_bound = diff - margin_error
        upper_bound = diff + margin_error
        
        return lower_bound, upper_bound
    
    def effect_size(self) -> Dict[str, float]:
        """
        Calculate effect size measures.
        
        Returns:
            Dictionary containing various effect size measures
        """
        # Absolute difference
        absolute_diff = self.rate_b - self.rate_a
        
        # Relative difference (lift)
        relative_diff = (absolute_diff / self.rate_a) if self.rate_a > 0 else float('inf')
        
        # Cohen's h (effect size for proportions)
        cohens_h = 2 * (math.asin(math.sqrt(self.rate_b)) - math.asin(math.sqrt(self.rate_a)))
        
        # Risk ratio
        risk_ratio = self.rate_b / self.rate_a if self.rate_a > 0 else float('inf')
        
        # Odds ratio
        odds_a = self.rate_a / (1 - self.rate_a) if self.rate_a < 1 else float('inf')
        odds_b = self.rate_b / (1 - self.rate_b) if self.rate_b < 1 else float('inf')
        odds_ratio = odds_b / odds_a if odds_a > 0 else float('inf')
        
        return {
            'absolute_difference': absolute_diff,
            'relative_difference': relative_diff,
            'cohens_h': cohens_h,
            'risk_ratio': risk_ratio,
            'odds_ratio': odds_ratio
        }
    
    def sample_size_calculation(self, alpha: float = 0.05, power: float = 0.8, 
                              effect_size: float = 0.01) -> Dict[str, int]:
        """
        Calculate required sample size for given parameters.
        
        Args:
            alpha: Type I error rate
            power: Statistical power (1 - Type II error rate)
            effect_size: Expected effect size (difference in proportions)
            
        Returns:
            Dictionary containing sample size calculations
        """
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Baseline conversion rate
        p1 = self.rate_a
        p2 = p1 + effect_size
        
        # Pooled proportion
        p_avg = (p1 + p2) / 2
        
        # Sample size calculation
        numerator = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) + 
                    z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = effect_size ** 2
        
        n_per_group = math.ceil(numerator / denominator) if denominator > 0 else float('inf')
        
        return {
            'sample_size_per_group': n_per_group,
            'total_sample_size': n_per_group * 2,
            'current_sample_size': self.n_a + self.n_b,
            'is_adequately_powered': (self.n_a >= n_per_group and self.n_b >= n_per_group)
        }
    
    def perform_all_tests(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Perform comprehensive statistical testing.
        
        Args:
            confidence_level: Confidence level for tests and intervals
            
        Returns:
            Dictionary containing all test results
        """
        # Z-test
        z_results = self.z_test(confidence_level)
        
        # Chi-square test
        chi2_results = self.chi_square_test()
        
        # Confidence interval
        ci_lower, ci_upper = self.confidence_interval(confidence_level)
        
        # Effect size
        effect_results = self.effect_size()
        
        # Sample size analysis
        sample_size_results = self.sample_size_calculation()
        
        return {
            # Basic statistics
            'group_a_users': self.n_a,
            'group_b_users': self.n_b,
            'group_a_conversions': self.conv_a,
            'group_b_conversions': self.conv_b,
            'group_a_rate': self.rate_a,
            'group_b_rate': self.rate_b,
            
            # Z-test results
            'z_stat': z_results['z_statistic'],
            'z_pvalue': z_results['p_value'],
            'z_critical': z_results['z_critical'],
            'z_significant': z_results['is_significant'],
            
            # Chi-square results
            'chi2_stat': chi2_results['chi2_statistic'],
            'chi2_pvalue': chi2_results['p_value'],
            'chi2_dof': chi2_results['degrees_of_freedom'],
            
            # Confidence interval
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            
            # Effect sizes
            'effect_size': effect_results['absolute_difference'],
            'relative_lift': effect_results['relative_difference'],
            'cohens_h': effect_results['cohens_h'],
            'risk_ratio': effect_results['risk_ratio'],
            'odds_ratio': effect_results['odds_ratio'],
            
            # Sample size analysis
            'required_sample_size': sample_size_results['sample_size_per_group'],
            'is_adequately_powered': sample_size_results['is_adequately_powered']
        }
    
    def interpret_results(self, test_results: Dict[str, float], 
                         confidence_level: float = 0.95) -> Dict[str, str]:
        """
        Provide interpretation of statistical test results.
        
        Args:
            test_results: Results from perform_all_tests()
            confidence_level: Confidence level used
            
        Returns:
            Dictionary containing interpretations
        """
        alpha = 1 - confidence_level
        
        interpretations = {}
        
        # Statistical significance
        if test_results['z_pvalue'] < alpha:
            interpretations['significance'] = f"Statistically significant at {confidence_level:.0%} confidence level"
        else:
            interpretations['significance'] = f"Not statistically significant at {confidence_level:.0%} confidence level"
        
        # Effect magnitude
        abs_effect = abs(test_results['effect_size'])
        if abs_effect < 0.01:
            interpretations['effect_magnitude'] = "Small effect size"
        elif abs_effect < 0.05:
            interpretations['effect_magnitude'] = "Medium effect size"
        else:
            interpretations['effect_magnitude'] = "Large effect size"
        
        # Direction
        if test_results['effect_size'] > 0:
            interpretations['direction'] = f"Group B ({self.group_b}) performs better than Group A ({self.group_a})"
        else:
            interpretations['direction'] = f"Group A ({self.group_a}) performs better than Group B ({self.group_b})"
        
        # Sample size adequacy
        if test_results['is_adequately_powered']:
            interpretations['power'] = "Sample size is adequate for detecting meaningful differences"
        else:
            interpretations['power'] = f"Sample size may be insufficient. Consider collecting more data (recommended: {test_results['required_sample_size']} per group)"
        
        return interpretations
