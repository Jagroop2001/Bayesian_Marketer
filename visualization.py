import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class Visualizer:
    """
    Creates interactive visualizations for A/B testing analysis using Plotly.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
        # Color palette for consistent theming
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9500',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def plot_conversion_by_group(self) -> go.Figure:
        """
        Create bar chart showing conversion rates by test group.
        
        Returns:
            Plotly figure object
        """
        # Calculate conversion rates
        conv_data = self.data.groupby('test group').agg({
            'converted': ['count', 'sum', 'mean']
        }).round(4)
        conv_data.columns = ['Total_Users', 'Conversions', 'Conversion_Rate']
        conv_data = conv_data.reset_index()
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=conv_data['test group'],
            y=conv_data['Conversion_Rate'],
            text=[f'{rate:.2%}' for rate in conv_data['Conversion_Rate']],
            textposition='auto',
            marker_color=[self.colors['primary'], self.colors['secondary']],
            hovertemplate='<b>%{x}</b><br>' +
                         'Conversion Rate: %{y:.2%}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Conversion Rate by Test Group',
            xaxis_title='Test Group',
            yaxis_title='Conversion Rate',
            yaxis_tickformat='.1%',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_ad_exposure_distribution(self) -> go.Figure:
        """
        Create histogram showing distribution of ad exposure.
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for group in self.data['test group'].unique():
            group_data = self.data[self.data['test group'] == group]
            
            fig.add_trace(go.Histogram(
                x=group_data['total ads'],
                name=group,
                opacity=0.7,
                nbinsx=50,
                histnorm='probability'
            ))
        
        fig.update_layout(
            title='Distribution of Total Ads by Test Group',
            xaxis_title='Total Ads',
            yaxis_title='Probability',
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def plot_conversion_by_hour(self) -> go.Figure:
        """
        Create line chart showing conversion rates by hour of day.
        
        Returns:
            Plotly figure object
        """
        # Calculate hourly conversion rates
        hourly_data = self.data.groupby(['most ads hour', 'test group']).agg({
            'converted': ['count', 'sum', 'mean']
        }).round(4)
        hourly_data.columns = ['Total_Users', 'Conversions', 'Conversion_Rate']
        hourly_data = hourly_data.reset_index()
        
        fig = go.Figure()
        
        for group in hourly_data['test group'].unique():
            group_data = hourly_data[hourly_data['test group'] == group]
            
            fig.add_trace(go.Scatter(
                x=group_data['most ads hour'],
                y=group_data['Conversion_Rate'],
                mode='lines+markers',
                name=group,
                line=dict(width=3),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Hour: %{x}<br>' +
                             'Conversion Rate: %{y:.2%}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Conversion Rate by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Conversion Rate',
            yaxis_tickformat='.1%',
            height=400
        )
        
        return fig
    
    def plot_conversion_by_day(self) -> go.Figure:
        """
        Create bar chart showing conversion rates by day of week.
        
        Returns:
            Plotly figure object
        """
        # Define day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Calculate daily conversion rates
        daily_data = self.data.groupby(['most ads day', 'test group']).agg({
            'converted': ['count', 'sum', 'mean']
        }).round(4)
        daily_data.columns = ['Total_Users', 'Conversions', 'Conversion_Rate']
        daily_data = daily_data.reset_index()
        
        fig = go.Figure()
        
        groups = daily_data['test group'].unique()
        colors = [self.colors['primary'], self.colors['secondary']]
        
        for i, group in enumerate(groups):
            group_data = daily_data[daily_data['test group'] == group]
            
            # Reorder by day of week
            group_data['day_order'] = group_data['most ads day'].map({day: i for i, day in enumerate(day_order)})
            group_data = group_data.sort_values('day_order')
            
            fig.add_trace(go.Bar(
                x=group_data['most ads day'],
                y=group_data['Conversion_Rate'],
                name=group,
                marker_color=colors[i % len(colors)],
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Day: %{x}<br>' +
                             'Conversion Rate: %{y:.2%}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Conversion Rate by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Conversion Rate',
            yaxis_tickformat='.1%',
            barmode='group',
            height=400
        )
        
        return fig
    
    def plot_conversion_heatmap_hour_day(self) -> go.Figure:
        """
        Create heatmap showing conversion rates by hour and day.
        
        Returns:
            Plotly figure object
        """
        # Prepare data for heatmap
        heatmap_data = self.data.groupby(['most ads day', 'most ads hour'])['converted'].mean().reset_index()
        pivot_data = heatmap_data.pivot(index='most ads day', columns='most ads hour', values='converted')
        
        # Define day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Blues',
            text=np.round(pivot_data.values, 3),
            texttemplate='%{text:.1%}',
            textfont={"size": 10},
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Conversion Rate: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Conversion Rate Heatmap by Day and Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        
        return fig
    
    def plot_conversion_heatmap_by_group(self) -> go.Figure:
        """
        Create side-by-side heatmaps for each test group.
        
        Returns:
            Plotly figure object
        """
        groups = self.data['test group'].unique()
        
        fig = make_subplots(
            rows=1, cols=len(groups),
            subplot_titles=[f'{group.title()} Group' for group in groups],
            shared_yaxes=True,
            horizontal_spacing=0.1
        )
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for i, group in enumerate(groups):
            group_data = self.data[self.data['test group'] == group]
            heatmap_data = group_data.groupby(['most ads day', 'most ads hour'])['converted'].mean().reset_index()
            pivot_data = heatmap_data.pivot(index='most ads day', columns='most ads hour', values='converted')
            pivot_data = pivot_data.reindex(day_order)
            
            fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='Blues',
                    showscale=(i == len(groups) - 1),  # Only show colorbar for last subplot
                    hovertemplate=f'Day: %{{y}}<br>Hour: %{{x}}<br>Conversion Rate: %{{z:.2%}}<extra></extra>'
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Conversion Rate Heatmaps by Test Group',
            height=400
        )
        
        # Update x-axis titles
        for i in range(len(groups)):
            fig.update_xaxes(title_text='Hour of Day', row=1, col=i+1)
        
        # Update y-axis title for first subplot only
        fig.update_yaxes(title_text='Day of Week', row=1, col=1)
        
        return fig
    
    def plot_ad_exposure_by_group(self) -> go.Figure:
        """
        Create box plot showing ad exposure distribution by group.
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = [self.colors['primary'], self.colors['secondary']]
        
        for i, group in enumerate(self.data['test group'].unique()):
            group_data = self.data[self.data['test group'] == group]
            
            fig.add_trace(go.Box(
                y=group_data['total ads'],
                name=group,
                marker_color=colors[i % len(colors)],
                boxpoints='outliers',
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Total Ads: %{y}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Distribution of Ad Exposure by Test Group',
            xaxis_title='Test Group',
            yaxis_title='Total Ads',
            height=400
        )
        
        return fig
    
    def plot_posterior_distributions(self, bayesian_results: Dict[str, Any]) -> go.Figure:
        """
        Create plot showing posterior distributions for both groups.
        
        Args:
            bayesian_results: Results from Bayesian analysis
            
        Returns:
            Plotly figure object
        """
        from scipy import stats
        
        # Generate x values for plotting
        x = np.linspace(0, 0.1, 1000)
        
        # Generate posterior distributions
        alpha_a = bayesian_results['posterior_a_alpha']
        beta_a = bayesian_results['posterior_a_beta']
        alpha_b = bayesian_results['posterior_b_alpha']
        beta_b = bayesian_results['posterior_b_beta']
        
        y_a = stats.beta.pdf(x, alpha_a, beta_a)
        y_b = stats.beta.pdf(x, alpha_b, beta_b)
        
        fig = go.Figure()
        
        # Add posterior distributions
        fig.add_trace(go.Scatter(
            x=x,
            y=y_a,
            mode='lines',
            name='Group A (Control)',
            line=dict(color=self.colors['primary'], width=3),
            fill='tonexty',
            fillcolor=f'rgba(31, 119, 180, 0.3)',
            hovertemplate='Conversion Rate: %{x:.2%}<br>Density: %{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y_b,
            mode='lines',
            name='Group B (Treatment)',
            line=dict(color=self.colors['secondary'], width=3),
            fill='tonexty',
            fillcolor=f'rgba(255, 127, 14, 0.3)',
            hovertemplate='Conversion Rate: %{x:.2%}<br>Density: %{y:.2f}<extra></extra>'
        ))
        
        # Add vertical lines for means
        mean_a = bayesian_results['posterior_a_mean']
        mean_b = bayesian_results['posterior_b_mean']
        
        fig.add_vline(
            x=mean_a, 
            line_dash="dash", 
            line_color=self.colors['primary'],
            annotation_text=f"Mean A: {mean_a:.2%}",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=mean_b, 
            line_dash="dash", 
            line_color=self.colors['secondary'],
            annotation_text=f"Mean B: {mean_b:.2%}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title='Posterior Distributions of Conversion Rates',
            xaxis_title='Conversion Rate',
            yaxis_title='Probability Density',
            xaxis_tickformat='.1%',
            height=400
        )
        
        return fig
    
    def plot_lift_distribution(self, lift_samples: np.ndarray) -> go.Figure:
        """
        Create histogram of lift distribution.
        
        Args:
            lift_samples: Array of lift samples from Bayesian analysis
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=lift_samples,
            nbinsx=50,
            opacity=0.7,
            marker_color=self.colors['info'],
            name='Lift Distribution',
            hovertemplate='Lift: %{x:.1%}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add vertical line at zero
        fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color=self.colors['danger'],
            annotation_text="No Effect",
            annotation_position="top"
        )
        
        # Add vertical line at mean
        mean_lift = np.mean(lift_samples)
        fig.add_vline(
            x=mean_lift, 
            line_dash="dash", 
            line_color=self.colors['success'],
            annotation_text=f"Mean: {mean_lift:.1%}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title='Distribution of Relative Lift (B vs A)',
            xaxis_title='Relative Lift',
            yaxis_title='Frequency',
            xaxis_tickformat='.1%',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_confidence_intervals(self, traditional_results: Dict[str, float],
                                 bayesian_results: Dict[str, float]) -> go.Figure:
        """
        Compare confidence intervals from traditional and Bayesian approaches.
        
        Args:
            traditional_results: Results from traditional statistical tests
            bayesian_results: Results from Bayesian analysis
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Traditional confidence interval
        trad_point = traditional_results['effect_size']
        trad_lower = traditional_results['ci_lower']
        trad_upper = traditional_results['ci_upper']
        
        # Bayesian credible interval  
        bayes_point = bayesian_results['expected_lift']
        bayes_lower = bayesian_results['ci_lower']
        bayes_upper = bayesian_results['ci_upper']
        
        # Add intervals
        fig.add_trace(go.Scatter(
            x=[trad_lower, trad_upper],
            y=['Traditional 95% CI', 'Traditional 95% CI'],
            mode='lines+markers',
            line=dict(color=self.colors['primary'], width=8),
            marker=dict(size=10),
            name='Traditional CI',
            hovertemplate='%{y}<br>Range: %{x:.2%}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[bayes_lower, bayes_upper],
            y=['Bayesian 95% CI', 'Bayesian 95% CI'],
            mode='lines+markers',
            line=dict(color=self.colors['secondary'], width=8),
            marker=dict(size=10),
            name='Bayesian CI',
            hovertemplate='%{y}<br>Range: %{x:.2%}<extra></extra>'
        ))
        
        # Add point estimates
        fig.add_trace(go.Scatter(
            x=[trad_point],
            y=['Traditional 95% CI'],
            mode='markers',
            marker=dict(color=self.colors['primary'], size=12, symbol='diamond'),
            name='Traditional Estimate',
            hovertemplate='Point Estimate: %{x:.2%}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[bayes_point],
            y=['Bayesian 95% CI'],
            mode='markers',
            marker=dict(color=self.colors['secondary'], size=12, symbol='diamond'),
            name='Bayesian Estimate',
            hovertemplate='Point Estimate: %{x:.2%}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color=self.colors['dark'],
            opacity=0.5
        )
        
        fig.update_layout(
            title='Comparison of Confidence/Credible Intervals',
            xaxis_title='Effect Size',
            xaxis_tickformat='.1%',
            showlegend=True,
            height=300,
            yaxis=dict(showgrid=False)
        )
        
        return fig
