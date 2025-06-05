import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from statistical_tests import StatisticalTests
from bayesian_analysis import BayesianAnalysis
from visualization import Visualizer

# Page configuration
st.set_page_config(
    page_title="Bayesian A/B Testing Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🧪 Bayesian A/B Testing for Marketing Campaign Effectiveness")
st.markdown("---")

# Sidebar for file upload and configuration
st.sidebar.header("📊 Data Upload & Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Marketing Campaign CSV",
    type=['csv'],
    help="Upload your A/B testing dataset with columns: user_id, test_group, converted, total_ads, most_ads_day, most_ads_hour"
)

if uploaded_file is not None:
    try:
        # Load and process data
        with st.spinner("Loading and processing data..."):
            df = pd.read_csv(uploaded_file)
            processor = DataProcessor(df)
            
            # Validate data
            validation_results = processor.validate_data()
            
            if not validation_results['is_valid']:
                st.error("❌ Data validation failed:")
                for error in validation_results['errors']:
                    st.error(f"• {error}")
                st.stop()
            
            # Clean and prepare data
            df_clean = processor.clean_data()
            
        # Display dataset overview
        st.header("📈 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", f"{len(df_clean):,}")
        with col2:
            st.metric("Test Groups", df_clean['test group'].nunique())
        with col3:
            st.metric("Conversion Rate", f"{df_clean['converted'].mean():.2%}")
        with col4:
            st.metric("Total Conversions", f"{df_clean['converted'].sum():,}")
        
        # Display data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        # Sidebar configuration
        st.sidebar.header("⚙️ Analysis Configuration")
        
        # Filter by test groups
        available_groups = df_clean['test group'].unique()
        if len(available_groups) >= 2:
            group_a = st.sidebar.selectbox("Group A (Control)", available_groups, index=0)
            group_b = st.sidebar.selectbox("Group B (Treatment)", available_groups, index=1)
        else:
            st.error("Dataset must contain at least 2 test groups for comparison")
            st.stop()
        
        # Bayesian prior configuration
        st.sidebar.subheader("🎯 Bayesian Prior Settings")
        prior_alpha = st.sidebar.number_input("Prior Alpha (successes)", min_value=0.1, value=1.0, step=0.1)
        prior_beta = st.sidebar.number_input("Prior Beta (failures)", min_value=0.1, value=1.0, step=0.1)
        
        # Confidence level
        confidence_level = st.sidebar.slider("Confidence Level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
        
        # Filter data for selected groups
        df_filtered = df_clean[df_clean['test group'].isin([group_a, group_b])].copy()
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Exploratory Data Analysis", 
            "🔬 Traditional A/B Testing", 
            "🧠 Bayesian Analysis", 
            "📈 Visualizations", 
            "💡 Business Recommendations"
        ])
        
        # Initialize analysis objects
        visualizer = Visualizer(df_filtered)
        stat_tests = StatisticalTests(df_filtered, group_a, group_b)
        bayesian = BayesianAnalysis(df_filtered, group_a, group_b, prior_alpha, prior_beta)
        
        with tab1:
            st.header("📊 Exploratory Data Analysis")
            
            # Group statistics
            group_stats = processor.get_group_statistics(df_filtered)
            st.subheader("Group Statistics Summary")
            st.dataframe(group_stats, use_container_width=True)
            
            # Conversion rates by group
            st.subheader("Conversion Rates by Test Group")
            conv_rates = df_filtered.groupby('test group')['converted'].agg(['count', 'sum', 'mean']).round(4)
            conv_rates.columns = ['Total Users', 'Conversions', 'Conversion Rate']
            st.dataframe(conv_rates, use_container_width=True)
            
            # Distribution analysis
            st.subheader("Ad Exposure Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(visualizer.plot_ad_exposure_distribution(), use_container_width=True)
            
            with col2:
                st.plotly_chart(visualizer.plot_conversion_by_group(), use_container_width=True)
            
            # Time-based analysis
            st.subheader("Peak Conversion Times")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(visualizer.plot_conversion_by_hour(), use_container_width=True)
            
            with col2:
                st.plotly_chart(visualizer.plot_conversion_by_day(), use_container_width=True)
        
        with tab2:
            st.header("🔬 Traditional A/B Testing")
            
            # Perform statistical tests
            test_results = stat_tests.perform_all_tests(confidence_level)
            
            # Display results
            st.subheader("Statistical Test Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    f"{group_a} Conversion Rate", 
                    f"{test_results['group_a_rate']:.4f}",
                    f"{test_results['group_a_conversions']}/{test_results['group_a_users']} users"
                )
            
            with col2:
                st.metric(
                    f"{group_b} Conversion Rate", 
                    f"{test_results['group_b_rate']:.4f}",
                    f"{test_results['group_b_conversions']}/{test_results['group_b_users']} users"
                )
            
            # Test results table
            results_df = pd.DataFrame({
                'Test': ['Z-Test', 'Chi-Square Test'],
                'Statistic': [test_results['z_stat'], test_results['chi2_stat']],
                'P-Value': [test_results['z_pvalue'], test_results['chi2_pvalue']],
                'Significant': [
                    test_results['z_pvalue'] < (1 - confidence_level),
                    test_results['chi2_pvalue'] < (1 - confidence_level)
                ]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Effect size
            st.subheader("Effect Size Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Absolute Difference", f"{test_results['effect_size']:.4f}")
            with col2:
                st.metric("Relative Lift", f"{test_results['relative_lift']:.2%}")
            with col3:
                st.metric("Confidence Interval", f"[{test_results['ci_lower']:.4f}, {test_results['ci_upper']:.4f}]")
        
        with tab3:
            st.header("🧠 Bayesian Analysis")
            
            # Perform Bayesian analysis
            bayesian_results = bayesian.perform_analysis()
            
            # Display prior information
            st.subheader("Prior Configuration")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Prior Alpha (successes):** {prior_alpha}")
            with col2:
                st.info(f"**Prior Beta (failures):** {prior_beta}")
            
            # Posterior distributions
            st.subheader("Posterior Distributions")
            st.plotly_chart(visualizer.plot_posterior_distributions(bayesian_results), use_container_width=True)
            
            # Key metrics
            st.subheader("Bayesian Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"{group_b} > {group_a} Probability", 
                    f"{bayesian_results['prob_b_better']:.1%}"
                )
            
            with col2:
                st.metric(
                    "Expected Lift", 
                    f"{bayesian_results['expected_lift']:.2%}"
                )
            
            with col3:
                st.metric(
                    f"Credible Interval ({confidence_level:.0%})",
                    f"[{bayesian_results['ci_lower']:.1%}, {bayesian_results['ci_upper']:.1%}]"
                )
            
            with col4:
                st.metric(
                    "Practical Significance",
                    "Yes" if bayesian_results['practical_significance'] else "No"
                )
            
            # Detailed results
            st.subheader("Detailed Posterior Statistics")
            posterior_df = pd.DataFrame({
                'Group': [group_a, group_b],
                'Posterior Alpha': [bayesian_results['posterior_a_alpha'], bayesian_results['posterior_b_alpha']],
                'Posterior Beta': [bayesian_results['posterior_a_beta'], bayesian_results['posterior_b_beta']],
                'Mean': [bayesian_results['posterior_a_mean'], bayesian_results['posterior_b_mean']],
                'Std Dev': [bayesian_results['posterior_a_std'], bayesian_results['posterior_b_std']]
            })
            st.dataframe(posterior_df, use_container_width=True)
        
        with tab4:
            st.header("📈 Advanced Visualizations")
            
            # Heatmaps
            st.subheader("Conversion Heatmaps")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(visualizer.plot_conversion_heatmap_hour_day(), use_container_width=True)
            
            with col2:
                st.plotly_chart(visualizer.plot_conversion_heatmap_by_group(), use_container_width=True)
            
            # Distribution comparisons
            st.subheader("Distribution Comparisons")
            st.plotly_chart(visualizer.plot_ad_exposure_by_group(), use_container_width=True)
            
            # Segment analysis
            st.subheader("Segment Analysis")
            segment_analysis = processor.get_segment_analysis(df_filtered)
            st.dataframe(segment_analysis, use_container_width=True)
        
        with tab5:
            st.header("💡 Business Recommendations")
            
            # Generate recommendations
            recommendations = bayesian.generate_business_recommendations(
                test_results, bayesian_results, confidence_level
            )
            
            # Display primary recommendation
            st.subheader("🎯 Primary Recommendation")
            if recommendations['primary_action'] == 'implement':
                st.success(f"**IMPLEMENT {group_b.upper()} STRATEGY**")
                st.success(recommendations['primary_reason'])
            elif recommendations['primary_action'] == 'continue_testing':
                st.warning("**CONTINUE TESTING**")
                st.warning(recommendations['primary_reason'])
            else:
                st.error("**DO NOT IMPLEMENT**")
                st.error(recommendations['primary_reason'])
            
            # Key insights
            st.subheader("📊 Key Insights")
            for insight in recommendations['key_insights']:
                st.info(f"• {insight}")
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            risk_level = recommendations['risk_level']
            if risk_level == 'Low':
                st.success(f"**Risk Level: {risk_level}**")
            elif risk_level == 'Medium':
                st.warning(f"**Risk Level: {risk_level}**")
            else:
                st.error(f"**Risk Level: {risk_level}**")
            
            for risk in recommendations['risks']:
                st.warning(f"• {risk}")
            
            # Next steps
            st.subheader("🚀 Recommended Next Steps")
            for i, step in enumerate(recommendations['next_steps'], 1):
                st.write(f"{i}. {step}")
            
            # Revenue impact simulation
            st.subheader("💰 Revenue Impact Simulation")
            
            col1, col2 = st.columns(2)
            with col1:
                revenue_per_conversion = st.number_input(
                    "Revenue per Conversion ($)", 
                    min_value=0.0, 
                    value=100.0, 
                    step=10.0
                )
            with col2:
                projected_users = st.number_input(
                    "Projected Monthly Users", 
                    min_value=1000, 
                    value=10000, 
                    step=1000
                )
            
            if st.button("Calculate Revenue Impact"):
                revenue_impact = bayesian.calculate_revenue_impact(
                    bayesian_results, projected_users, revenue_per_conversion
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Expected Monthly Lift", 
                        f"${revenue_impact['expected_monthly_lift']:,.2f}"
                    )
                with col2:
                    st.metric(
                        "90% Confidence Lower Bound", 
                        f"${revenue_impact['conservative_estimate']:,.2f}"
                    )
                with col3:
                    st.metric(
                        "Annual Revenue Potential", 
                        f"${revenue_impact['annual_potential']:,.2f}"
                    )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check your file format and try again.")

else:
    # Landing page when no file is uploaded
    st.markdown("""
    ## Welcome to the Bayesian A/B Testing Dashboard! 👋
    
    This tool helps you analyze marketing campaign effectiveness using both traditional statistical methods and Bayesian inference.
    
    ### 📊 What this tool provides:
    
    - **Comprehensive EDA**: Explore conversion rates, ad exposure patterns, and temporal trends
    - **Traditional A/B Testing**: Z-tests and Chi-square tests with confidence intervals
    - **Bayesian Analysis**: Beta-Binomial modeling with posterior distributions
    - **Business Recommendations**: Actionable insights with risk assessment
    - **Revenue Impact**: Simulate potential financial outcomes
    
    ### 📁 Expected Data Format:
    
    Your CSV file should contain these columns:
    - `user_id`: Unique identifier for each user
    - `test_group`: Group assignment (e.g., 'ad', 'control', 'psa')
    - `converted`: Boolean indicating if user converted (True/False)
    - `total_ads`: Number of ads shown to the user
    - `most_ads_day`: Day of week with highest ad exposure
    - `most_ads_hour`: Hour of day with highest ad exposure
    
    ### 🚀 Getting Started:
    
    1. Upload your marketing campaign CSV file using the sidebar
    2. Configure your analysis parameters
    3. Explore the different analysis tabs
    4. Review business recommendations
    
    **Ready to start? Upload your data file in the sidebar to begin!**
    """)
    
    # Sample data info
    st.subheader("📋 Sample Data Structure")
    sample_data = pd.DataFrame({
        'user_id': [1069124, 1119715, 1144181],
        'test_group': ['ad', 'ad', 'control'],
        'converted': [False, True, False],
        'total_ads': [130, 93, 21],
        'most_ads_day': ['Monday', 'Tuesday', 'Tuesday'],
        'most_ads_hour': [20, 22, 18]
    })
    st.dataframe(sample_data, use_container_width=True)
