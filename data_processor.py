import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DataProcessor:
    """
    Handles data loading, validation, cleaning, and preprocessing for A/B testing analysis.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.required_columns = ['user id', 'test group', 'converted', 'total ads', 'most ads day', 'most ads hour']
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the input data structure and content.
        
        Returns:
            Dict containing validation results and any errors found
        """
        errors = []
        
        # Check if data is not empty
        if self.data.empty:
            errors.append("Dataset is empty")
            return {'is_valid': False, 'errors': errors}
        
        # Check for required columns
        missing_columns = []
        for col in self.required_columns:
            if col not in self.data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check data types and content
        if 'converted' in self.data.columns:
            # Check if converted column has valid boolean-like values
            valid_converted_values = {True, False, 'True', 'False', 1, 0, '1', '0'}
            invalid_converted = set(self.data['converted'].unique()) - valid_converted_values
            if invalid_converted:
                errors.append(f"Invalid values in 'converted' column: {invalid_converted}")
        
        # Check for minimum sample size per group
        if 'test group' in self.data.columns and len(errors) == 0:
            group_counts = self.data['test group'].value_counts()
            small_groups = group_counts[group_counts < 10].index.tolist()
            if small_groups:
                errors.append(f"Groups with insufficient sample size (<10): {small_groups}")
        
        # Check for at least 2 test groups
        if 'test group' in self.data.columns and len(errors) == 0:
            if self.data['test group'].nunique() < 2:
                errors.append("Dataset must contain at least 2 test groups")
        
        return {'is_valid': len(errors) == 0, 'errors': errors}
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the data for analysis.
        
        Returns:
            Cleaned DataFrame ready for analysis
        """
        df = self.data.copy()
        
        # Remove any unnamed columns (index columns from CSV)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        
        # Convert 'converted' to boolean
        if 'converted' in df.columns:
            df['converted'] = df['converted'].astype(str).str.lower()
            df['converted'] = df['converted'].map({
                'true': True, 'false': False, '1': True, '0': False,
                '1.0': True, '0.0': False
            }).fillna(df['converted'].astype(bool))
        
        # Clean test group names
        if 'test group' in df.columns:
            df['test group'] = df['test group'].astype(str).str.strip().str.lower()
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['total ads', 'most ads hour']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize day names
        if 'most ads day' in df.columns:
            df['most ads day'] = df['most ads day'].astype(str).str.strip().str.title()
        
        # Remove rows with missing critical data
        critical_columns = ['test group', 'converted']
        df = df.dropna(subset=critical_columns)
        
        # Remove duplicate user IDs if any
        if 'user id' in df.columns:
            df = df.drop_duplicates(subset=['user id'])
        
        return df
    
    def get_group_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for each test group.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with group statistics
        """
        stats = []
        
        for group in df['test group'].unique():
            group_data = df[df['test group'] == group]
            
            stat_dict = {
                'Test Group': group,
                'Total Users': len(group_data),
                'Conversions': group_data['converted'].sum(),
                'Conversion Rate': group_data['converted'].mean(),
                'Avg Total Ads': group_data['total ads'].mean(),
                'Median Total Ads': group_data['total ads'].median(),
                'Std Total Ads': group_data['total ads'].std(),
                'Min Total Ads': group_data['total ads'].min(),
                'Max Total Ads': group_data['total ads'].max()
            }
            stats.append(stat_dict)
        
        return pd.DataFrame(stats).round(4)
    
    def get_segment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform segment analysis by day and hour.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with segment analysis results
        """
        segments = []
        
        # Day-wise analysis
        for day in df['most ads day'].unique():
            if pd.notna(day):
                day_data = df[df['most ads day'] == day]
                for group in day_data['test group'].unique():
                    group_day_data = day_data[day_data['test group'] == group]
                    
                    if len(group_day_data) > 0:
                        segments.append({
                            'Segment Type': 'Day',
                            'Segment': day,
                            'Test Group': group,
                            'Users': len(group_day_data),
                            'Conversions': group_day_data['converted'].sum(),
                            'Conversion Rate': group_day_data['converted'].mean(),
                            'Avg Ads': group_day_data['total ads'].mean()
                        })
        
        # Hour-wise analysis (peak hours)
        peak_hours = df['most ads hour'].value_counts().head(6).index
        for hour in peak_hours:
            if pd.notna(hour):
                hour_data = df[df['most ads hour'] == hour]
                for group in hour_data['test group'].unique():
                    group_hour_data = hour_data[hour_data['test group'] == group]
                    
                    if len(group_hour_data) > 0:
                        segments.append({
                            'Segment Type': 'Hour',
                            'Segment': f"{int(hour)}:00",
                            'Test Group': group,
                            'Users': len(group_hour_data),
                            'Conversions': group_hour_data['converted'].sum(),
                            'Conversion Rate': group_hour_data['converted'].mean(),
                            'Avg Ads': group_hour_data['total ads'].mean()
                        })
        
        segment_df = pd.DataFrame(segments)
        if not segment_df.empty:
            segment_df = segment_df.round(4).sort_values(['Segment Type', 'Segment', 'Test Group'])
        
        return segment_df
    
    def get_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Analyze temporal patterns in the data.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary containing temporal analysis results
        """
        patterns = {}
        
        # Hour-wise conversion rates
        hourly_conv = df.groupby(['most ads hour', 'test group'])['converted'].agg(['count', 'sum', 'mean']).reset_index()
        hourly_conv.columns = ['Hour', 'Test Group', 'Users', 'Conversions', 'Conversion Rate']
        patterns['hourly'] = hourly_conv
        
        # Day-wise conversion rates
        daily_conv = df.groupby(['most ads day', 'test group'])['converted'].agg(['count', 'sum', 'mean']).reset_index()
        daily_conv.columns = ['Day', 'Test Group', 'Users', 'Conversions', 'Conversion Rate']
        patterns['daily'] = daily_conv
        
        return patterns
    
    def filter_by_criteria(self, df: pd.DataFrame, criteria: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter data based on specified criteria.
        
        Args:
            df: DataFrame to filter
            criteria: Dictionary of filtering criteria
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        if 'min_ads' in criteria:
            filtered_df = filtered_df[filtered_df['total ads'] >= criteria['min_ads']]
        
        if 'max_ads' in criteria:
            filtered_df = filtered_df[filtered_df['total ads'] <= criteria['max_ads']]
        
        if 'days' in criteria:
            filtered_df = filtered_df[filtered_df['most ads day'].isin(criteria['days'])]
        
        if 'hours' in criteria:
            filtered_df = filtered_df[filtered_df['most ads hour'].isin(criteria['hours'])]
        
        if 'test_groups' in criteria:
            filtered_df = filtered_df[filtered_df['test group'].isin(criteria['test_groups'])]
        
        return filtered_df
