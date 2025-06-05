import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
import math

class BayesianAnalysis:
    """
    Performs Bayesian inference for A/B testing using Beta-Binomial conjugate priors.
    """
    
    def __init__(self, data: pd.DataFrame, group_a: str, group_b: str, 
                 prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.data = data
        self.group_a = group_a
        self.group_b = group_b
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Prepare group data
        self.data_a = data[data['test group'] == group_a]
        self.data_b = data[data['test group'] == group_b]
        
        # Observed data
        self.n_a = len(self.data_a)
        self.n_b = len(self.data_b)
        self.successes_a = self.data_a['converted'].sum()
        self.successes_b = self.data_b['converted'].sum()
        self.failures_a = self.n_a - self.successes_a
        self.failures_b = self.n_b - self.successes_b
        
        # Posterior parameters
        self.posterior_alpha_a = self.prior_alpha + self.successes_a
        self.posterior_beta_a = self.prior_beta + self.failures_a
        self.posterior_alpha_b = self.prior_alpha + self.successes_b
        self.posterior_beta_b = self.prior_beta + self.failures_b
    
    def posterior_statistics(self) -> Dict[str, float]:
        """
        Calculate posterior distribution statistics.
        
        Returns:
            Dictionary containing posterior statistics for both groups
        """
        # Group A posterior
        mean_a = self.posterior_alpha_a / (self.posterior_alpha_a + self.posterior_beta_a)
        var_a = (self.posterior_alpha_a * self.posterior_beta_a) / \
                ((self.posterior_alpha_a + self.posterior_beta_a)**2 * 
                 (self.posterior_alpha_a + self.posterior_beta_a + 1))
        std_a = math.sqrt(var_a)
        
        # Group B posterior
        mean_b = self.posterior_alpha_b / (self.posterior_alpha_b + self.posterior_beta_b)
        var_b = (self.posterior_alpha_b * self.posterior_beta_b) / \
                ((self.posterior_alpha_b + self.posterior_beta_b)**2 * 
                 (self.posterior_alpha_b + self.posterior_beta_b + 1))
        std_b = math.sqrt(var_b)
        
        return {
            'posterior_a_alpha': self.posterior_alpha_a,
            'posterior_a_beta': self.posterior_beta_a,
            'posterior_a_mean': mean_a,
            'posterior_a_var': var_a,
            'posterior_a_std': std_a,
            'posterior_b_alpha': self.posterior_alpha_b,
            'posterior_b_beta': self.posterior_beta_b,
            'posterior_b_mean': mean_b,
            'posterior_b_var': var_b,
            'posterior_b_std': std_b
        }
    
    def probability_b_better_than_a(self, n_samples: int = 10000) -> float:
        """
        Calculate the probability that group B is better than group A.
        
        Args:
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Probability that B > A
        """
        # Sample from posterior distributions
        samples_a = np.random.beta(self.posterior_alpha_a, self.posterior_beta_a, n_samples)
        samples_b = np.random.beta(self.posterior_alpha_b, self.posterior_beta_b, n_samples)
        
        # Calculate probability B > A
        prob_b_better = np.mean(samples_b > samples_a)
        
        return prob_b_better
    
    def expected_loss(self, n_samples: int = 10000) -> Dict[str, float]:
        """
        Calculate expected loss for choosing each variant.
        
        Args:
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary containing expected losses
        """
        # Sample from posterior distributions
        samples_a = np.random.beta(self.posterior_alpha_a, self.posterior_beta_a, n_samples)
        samples_b = np.random.beta(self.posterior_alpha_b, self.posterior_beta_b, n_samples)
        
        # Expected loss if we choose A (when B is actually better)
        loss_choose_a = np.mean(np.maximum(0, samples_b - samples_a))
        
        # Expected loss if we choose B (when A is actually better)
        loss_choose_b = np.mean(np.maximum(0, samples_a - samples_b))
        
        return {
            'expected_loss_choose_a': loss_choose_a,
            'expected_loss_choose_b': loss_choose_b
        }
    
    def credible_interval(self, group: str, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate credible interval for a group's conversion rate.
        
        Args:
            group: 'A' or 'B'
            confidence_level: Confidence level for the interval
            
        Returns:
            Tuple containing (lower_bound, upper_bound)
        """
        alpha = 1 - confidence_level
        
        if group.upper() == 'A':
            lower = stats.beta.ppf(alpha/2, self.posterior_alpha_a, self.posterior_beta_a)
            upper = stats.beta.ppf(1 - alpha/2, self.posterior_alpha_a, self.posterior_beta_a)
        else:  # Group B
            lower = stats.beta.ppf(alpha/2, self.posterior_alpha_b, self.posterior_beta_b)
            upper = stats.beta.ppf(1 - alpha/2, self.posterior_alpha_b, self.posterior_beta_b)
        
        return lower, upper
    
    def lift_distribution(self, n_samples: int = 10000) -> np.ndarray:
        """
        Generate samples from the lift distribution (B - A).
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of lift samples
        """
        samples_a = np.random.beta(self.posterior_alpha_a, self.posterior_beta_a, n_samples)
        samples_b = np.random.beta(self.posterior_alpha_b, self.posterior_beta_b, n_samples)
        
        lift_samples = (samples_b - samples_a) / samples_a
        
        return lift_samples
    
    def lift_credible_interval(self, confidence_level: float = 0.95, 
                              n_samples: int = 10000) -> Tuple[float, float]:
        """
        Calculate credible interval for the relative lift.
        
        Args:
            confidence_level: Confidence level for the interval
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple containing (lower_bound, upper_bound) for lift
        """
        lift_samples = self.lift_distribution(n_samples)
        
        alpha = 1 - confidence_level
        lower = np.percentile(lift_samples, 100 * alpha/2)
        upper = np.percentile(lift_samples, 100 * (1 - alpha/2))
        
        return lower, upper
    
    def minimum_detectable_effect(self, power: float = 0.8, 
                                 confidence_level: float = 0.95) -> float:
        """
        Calculate the minimum detectable effect given current sample sizes.
        
        Args:
            power: Desired statistical power
            confidence_level: Confidence level
            
        Returns:
            Minimum detectable effect (absolute difference in rates)
        """
        posterior_stats = self.posterior_statistics()
        
        # Current posterior means
        p_a = posterior_stats['posterior_a_mean']
        p_b = posterior_stats['posterior_b_mean']
        
        # Approximate standard errors
        se_a = posterior_stats['posterior_a_std']
        se_b = posterior_stats['posterior_b_std']
        se_diff = math.sqrt(se_a**2 + se_b**2)
        
        # Critical values
        alpha = 1 - confidence_level
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # MDE calculation
        mde = (z_alpha + z_beta) * se_diff
        
        return mde
    
    def bayesian_factor(self, null_hypothesis_range: Tuple[float, float] = (-0.001, 0.001),
                       n_samples: int = 10000) -> float:
        """
        Calculate Bayes factor for the hypothesis that there's no practical difference.
        
        Args:
            null_hypothesis_range: Range considered as "no practical difference"
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Bayes factor (BF10 - evidence for alternative vs null)
        """
        # Sample from the difference distribution
        samples_a = np.random.beta(self.posterior_alpha_a, self.posterior_beta_a, n_samples)
        samples_b = np.random.beta(self.posterior_alpha_b, self.posterior_beta_b, n_samples)
        diff_samples = samples_b - samples_a
        
        # Count samples in null hypothesis range
        null_count = np.sum((diff_samples >= null_hypothesis_range[0]) & 
                           (diff_samples <= null_hypothesis_range[1]))
        
        # Count samples outside null hypothesis range (alternative)
        alt_count = n_samples - null_count
        
        # Bayes factor (alternative / null)
        if null_count > 0:
            bayes_factor = alt_count / null_count
        else:
            bayes_factor = float('inf')
        
        return bayes_factor
    
    def perform_analysis(self, confidence_level: float = 0.95, 
                        n_samples: int = 10000) -> Dict[str, float]:
        """
        Perform comprehensive Bayesian analysis.
        
        Args:
            confidence_level: Confidence level for intervals
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary containing all Bayesian analysis results
        """
        # Posterior statistics
        posterior_stats = self.posterior_statistics()
        
        # Probability B > A
        prob_b_better = self.probability_b_better_than_a(n_samples)
        
        # Expected losses
        losses = self.expected_loss(n_samples)
        
        # Credible intervals
        ci_a_lower, ci_a_upper = self.credible_interval('A', confidence_level)
        ci_b_lower, ci_b_upper = self.credible_interval('B', confidence_level)
        
        # Lift analysis
        lift_samples = self.lift_distribution(n_samples)
        expected_lift = np.mean(lift_samples)
        lift_ci_lower, lift_ci_upper = self.lift_credible_interval(confidence_level, n_samples)
        
        # Minimum detectable effect
        mde = self.minimum_detectable_effect()
        
        # Bayes factor
        bf = self.bayesian_factor(n_samples=n_samples)
        
        # Practical significance (lift > 1% with high confidence)
        practical_threshold = 0.01
        prob_practical_significance = np.mean(lift_samples > practical_threshold)
        
        # Combine all results
        results = {
            **posterior_stats,
            'prob_b_better': prob_b_better,
            'prob_a_better': 1 - prob_b_better,
            'expected_loss_choose_a': losses['expected_loss_choose_a'],
            'expected_loss_choose_b': losses['expected_loss_choose_b'],
            'ci_a_lower': ci_a_lower,
            'ci_a_upper': ci_a_upper,
            'ci_b_lower': ci_b_lower,
            'ci_b_upper': ci_b_upper,
            'expected_lift': expected_lift,
            'ci_lower': lift_ci_lower,
            'ci_upper': lift_ci_upper,
            'minimum_detectable_effect': mde,
            'bayes_factor': bf,
            'prob_practical_significance': prob_practical_significance,
            'practical_significance': prob_practical_significance > 0.9,
            'lift_samples': lift_samples  # For plotting
        }
        
        return results
    
    def generate_business_recommendations(self, traditional_results: Dict[str, float],
                                        bayesian_results: Dict[str, float],
                                        confidence_level: float = 0.95) -> Dict[str, any]:
        """
        Generate actionable business recommendations based on analysis results.
        
        Args:
            traditional_results: Results from traditional statistical tests
            bayesian_results: Results from Bayesian analysis
            confidence_level: Confidence level used in analysis
            
        Returns:
            Dictionary containing business recommendations
        """
        recommendations = {
            'primary_action': '',
            'primary_reason': '',
            'key_insights': [],
            'risks': [],
            'next_steps': [],
            'risk_level': ''
        }
        
        # Extract key metrics
        prob_b_better = bayesian_results['prob_b_better']
        expected_lift = bayesian_results['expected_lift']
        practical_sig = bayesian_results['practical_significance']
        statistical_sig = traditional_results.get('z_significant', False)
        
        # Decision logic
        if prob_b_better > 0.95 and practical_sig and expected_lift > 0.01:
            recommendations['primary_action'] = 'implement'
            recommendations['primary_reason'] = f"Strong evidence favoring {self.group_b} strategy with {prob_b_better:.1%} probability of being better and {expected_lift:.1%} expected lift."
            recommendations['risk_level'] = 'Low'
        elif prob_b_better > 0.8 and expected_lift > 0.005:
            recommendations['primary_action'] = 'continue_testing'
            recommendations['primary_reason'] = f"Moderate evidence favoring {self.group_b} strategy ({prob_b_better:.1%} probability), but consider more data for stronger confidence."
            recommendations['risk_level'] = 'Medium'
        elif prob_b_better < 0.2:
            recommendations['primary_action'] = 'do_not_implement'
            recommendations['primary_reason'] = f"Evidence suggests {self.group_a} strategy is likely better ({1-prob_b_better:.1%} probability)."
            recommendations['risk_level'] = 'Low'
        else:
            recommendations['primary_action'] = 'continue_testing'
            recommendations['primary_reason'] = "Inconclusive results. Need more data to make a confident decision."
            recommendations['risk_level'] = 'High'
        
        # Key insights
        if statistical_sig:
            recommendations['key_insights'].append("Traditional statistical tests show significant difference")
        else:
            recommendations['key_insights'].append("Traditional statistical tests do not show significant difference")
        
        recommendations['key_insights'].append(f"Bayesian analysis shows {prob_b_better:.1%} probability that {self.group_b} is better")
        recommendations['key_insights'].append(f"Expected relative lift: {expected_lift:.1%}")
        
        if bayesian_results['ci_lower'] > 0:
            recommendations['key_insights'].append(f"95% confident that lift is at least {bayesian_results['ci_lower']:.1%}")
        
        # Risk assessment
        if not traditional_results.get('is_adequately_powered', True):
            recommendations['risks'].append("Sample size may be insufficient for reliable conclusions")
        
        if abs(expected_lift) < 0.005:
            recommendations['risks'].append("Effect size is very small and may not be practically significant")
        
        if prob_b_better > 0.2 and prob_b_better < 0.8:
            recommendations['risks'].append("High uncertainty in which variant is better")
        
        # Next steps
        if recommendations['primary_action'] == 'implement':
            recommendations['next_steps'] = [
                f"Roll out {self.group_b} strategy to broader audience",
                "Monitor key metrics closely during rollout",
                "Set up tracking for long-term performance",
                "Consider segmentation analysis for optimization"
            ]
        elif recommendations['primary_action'] == 'continue_testing':
            sample_size_needed = traditional_results.get('required_sample_size', 'unknown')
            recommendations['next_steps'] = [
                f"Continue testing until reaching {sample_size_needed} users per group",
                "Monitor for early stopping criteria",
                "Consider adjusting test parameters if needed",
                "Plan for follow-up experiments"
            ]
        else:
            recommendations['next_steps'] = [
                f"Stick with {self.group_a} strategy",
                "Investigate why alternative didn't perform better",
                "Consider testing different variations",
                "Analyze user segments for insights"
            ]
        
        return recommendations
    
    def calculate_revenue_impact(self, bayesian_results: Dict[str, float],
                               projected_users: int, revenue_per_conversion: float) -> Dict[str, float]:
        """
        Calculate potential revenue impact of implementing the better variant.
        
        Args:
            bayesian_results: Results from Bayesian analysis
            projected_users: Number of users expected per month
            revenue_per_conversion: Revenue generated per conversion
            
        Returns:
            Dictionary containing revenue impact calculations
        """
        # Current conversion rates
        rate_a = bayesian_results['posterior_a_mean']
        rate_b = bayesian_results['posterior_b_mean']
        
        # Expected conversions
        conversions_a = projected_users * rate_a
        conversions_b = projected_users * rate_b
        
        # Revenue calculations
        revenue_a = conversions_a * revenue_per_conversion
        revenue_b = conversions_b * revenue_per_conversion
        
        # Lift calculations
        expected_monthly_lift = (revenue_b - revenue_a)
        
        # Conservative estimate (using lower bound of credible interval)
        conservative_lift_rate = bayesian_results['ci_lower']
        conservative_conversions = projected_users * rate_a * (1 + conservative_lift_rate)
        conservative_revenue = conservative_conversions * revenue_per_conversion
        conservative_estimate = conservative_revenue - revenue_a
        
        return {
            'baseline_monthly_revenue': revenue_a,
            'expected_monthly_revenue': revenue_b,
            'expected_monthly_lift': expected_monthly_lift,
            'conservative_estimate': max(0, conservative_estimate),
            'annual_potential': expected_monthly_lift * 12,
            'baseline_conversion_rate': rate_a,
            'expected_conversion_rate': rate_b
        }
